"""
train.py
========
Modeli SIFIRDAN egitir. GTX 1660 Ti (6 GB) icin optimize edilmis:
  * mixed precision (Turing'de fp16 + GradScaler; bf16 destekleniyorsa bf16)
  * gradient checkpointing (config'ten)
  * gradient accumulation (kucuk batch ile buyuk efektif batch)
  * cosine LR + warmup, gradient clipping
  * checkpoint kaydet/devam et

Kullanim:
    python train.py --preset small-100m --data data/bin
    python train.py --preset tiny-30m  --data data/bin --max-steps 2000
    python train.py --resume checkpoints/ckpt.pt        # devam et
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, fields
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import GPTConfig, get_config
from model import GPT


# ---------------------------------------------------------------------------
def build_cfg_from_dict(d: dict) -> GPTConfig:
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in d.items() if k in valid})


def build_data_signature(data_dir: str) -> dict:
    """Dagitik egitimde tum makinelerin ayni veri hazirligini gordugunu dogrular."""
    names = ("meta.json", "train.bin", "val.bin")
    paths = {name: os.path.join(data_dir, name) for name in names}
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        return {
            "ok": False,
            "error": f"Eksik veri dosyasi: {', '.join(missing)} ({os.path.abspath(data_dir)})",
        }

    try:
        with open(paths["meta.json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return {
            "ok": False,
            "error": f"meta.json okunamadi: {e} ({os.path.abspath(paths['meta.json'])})",
        }

    return {
        "ok": True,
        "meta": meta,
        "files": {
            "train.bin": os.path.getsize(paths["train.bin"]),
            "val.bin": os.path.getsize(paths["val.bin"]),
        },
    }


def checked_data_signature(data_dir: str, dist_info: dict) -> tuple[dict, dict]:
    local_sig = build_data_signature(data_dir)
    signatures = [local_sig]
    if dist_info["enabled"]:
        signatures = [None] * dist_info["world_size"]
        dist.all_gather_object(signatures, local_sig)

    errors = []
    for rank, sig in enumerate(signatures):
        if not sig or not sig.get("ok"):
            errors.append(f"rank {rank}: {sig.get('error', 'bilinmeyen veri hatasi') if sig else 'imza yok'}")

    if not errors:
        ref = signatures[0]
        for rank, sig in enumerate(signatures[1:], start=1):
            if sig["meta"] != ref["meta"] or sig["files"] != ref["files"]:
                errors.append(
                    f"rank {rank}: data/bin imzasi rank 0 ile uyusmuyor "
                    f"(rank0={ref['files']}, rank{rank}={sig['files']})"
                )

    if errors:
        if dist_info["rank"] == 0:
            print("[data] Dagitik veri on kontrolu basarisiz:")
            for err in errors:
                print(f"  - {err}")
        if dist_info["enabled"]:
            dist.destroy_process_group()
        raise SystemExit("Tum rank'lerde ayni meta.json/train.bin/val.bin hazir olmali.")

    return local_sig["meta"], local_sig["files"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type_of(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device.startswith("mps"):
        return "mps"
    return "cpu"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def setup_distributed(requested_device: str, requested_backend: str, timeout_minutes: int) -> tuple[dict, str]:
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    enabled = world_size > 1

    device = default_device() if requested_device == "auto" else requested_device
    device_type = device_type_of(device)

    if enabled:
        if not dist.is_available():
            raise SystemExit("torch.distributed kullanilamiyor; PyTorch kurulumunu kontrol et.")
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit("--device cuda secildi ama CUDA gorunmuyor.")
            cuda_index = local_rank % max(1, torch.cuda.device_count())
            torch.cuda.set_device(cuda_index)
            device = f"cuda:{cuda_index}"
            device_type = "cuda"

        # Cross-platform LAN icin varsayilan gloo. Homojen Linux/CUDA kumesinde
        # istersen --dist-backend nccl ile daha hizli backend secebilirsin.
        backend = "gloo" if requested_backend == "auto" else requested_backend
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    else:
        backend = "none"

    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": backend,
    }, device


def unwrap_model(model):
    raw = model.module if hasattr(model, "module") else model
    return raw._orig_mod if hasattr(raw, "_orig_mod") else raw


def get_lr(step, cfg: GPTConfig):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.lr_decay_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.lr_decay_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small-100m")
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--out", default=None, help="checkpoint klasoru (vars: config.out_dir)")
    ap.add_argument("--resume", default=None, help="devam edilecek ckpt yolu")
    ap.add_argument("--adapter", action="store_true",
                    help="base modeli dondurup sadece kucuk neural adapter egit")
    ap.add_argument("--adapter-dim", type=int, default=64,
                    help="adapter bottleneck boyutu")
    ap.add_argument("--adapter-dropout", type=float, default=0.0)
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--adapter-out", default=None,
                    help="adapter checkpoint klasoru (vars: cfg.out_dir)")
    ap.add_argument("--adapter-resume", default=None,
                    help="devam edilecek adapter.pt yolu")
    ap.add_argument("--adapter-continue-step", action="store_true",
                    help="adapter-resume icindeki step sayacindan devam et; varsayilan sadece agirliklari yukler")
    ap.add_argument("--adapter-lr", type=float, default=None,
                    help="adapter egitim learning-rate (vars: 1e-3)")
    ap.add_argument("--reset-best", action="store_true",
                    help="Devam ederken en-iyi-val sayacini sifirla. SFT gibi FARKLI "
                         "bir veri setine gecerken SART: yoksa eski (dusuk) best_val "
                         "yuzunden yeni ckpt.pt hic kaydedilmez.")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr-decay-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile (Linux'ta hizli)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps veya cpu")
    ap.add_argument("--dist-backend", default="auto", choices=("auto", "gloo", "nccl"))
    ap.add_argument("--dist-timeout-minutes", type=int, default=60)
    args = ap.parse_args()

    dist_info, device = setup_distributed(args.device, args.dist_backend, args.dist_timeout_minutes)
    is_main = dist_info["rank"] == 0
    world_size = dist_info["world_size"]

    def log(msg: str):
        if is_main:
            print(msg)

    torch.manual_seed(args.seed + dist_info["rank"])
    np.random.seed(args.seed + dist_info["rank"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta, data_files = checked_data_signature(args.data, dist_info)
    np_dtype = np.dtype(meta["dtype"])
    log(f"[data] train.bin={data_files['train.bin']:,} bayt | "
        f"val.bin={data_files['val.bin']:,} bayt | dtype={np_dtype.name}")

    # --- config ---
    start_step = 0
    best_val = float("inf")
    if args.adapter and not args.resume:
        raise SystemExit("--adapter icin --resume checkpoints/ckpt.pt gerekli.")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = build_cfg_from_dict(ckpt["config"])
        start_step = 0 if args.adapter else ckpt.get("step", 0)
        best_val = float("inf") if args.adapter else ckpt.get("best_val", float("inf"))
        if args.adapter:
            cfg.adapter_enabled = True
            cfg.adapter_dim = max(1, args.adapter_dim)
            cfg.adapter_dropout = max(0.0, args.adapter_dropout)
            cfg.adapter_scale = args.adapter_scale
            cfg.learning_rate = args.adapter_lr or 1e-3
            cfg.weight_decay = 0.0
        if args.reset_best:
            best_val = float("inf")
            log("[train] best_val sifirlandi (yeni veri seti icin temiz ckpt.pt).")
        log(f"[train] devam ediliyor: {args.resume} (step={start_step})")
    else:
        cfg = get_config(args.preset)
        ckpt = None

    # veriye gore vocab'i sabitle
    cfg.vocab_size = meta["vocab_size"]
    if args.out:
        cfg.out_dir = args.out
    if args.adapter and args.adapter_out:
        cfg.out_dir = args.adapter_out
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
        cfg.lr_decay_steps = max(1, args.max_steps)   # decay'i kosu uzunluguyla esle
    if args.lr_decay_steps is not None:
        cfg.lr_decay_steps = args.lr_decay_steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.grad_accum:
        cfg.grad_accum_steps = args.grad_accum
    if args.adapter:
        cfg.warmup_steps = min(cfg.warmup_steps, max(1, cfg.max_steps // 10))

    os.makedirs(cfg.out_dir, exist_ok=True)
    device_type = device_type_of(device)
    if dist_info["enabled"]:
        log(f"[dist] backend={dist_info['backend']} world_size={world_size} "
            f"(rank 0 checkpoint/log yazar)")

    # --- precision ---
    if device_type == "cuda":
        # Turing (1660 Ti, cc 7.5) bf16'yi sadece EMULE eder (yavas). Gercek bf16
        # hizlandirmasi yalniz Ampere+ (cc >= 8.0). Yeni torch is_bf16_supported()
        # emulasyonu da True sayar; bu yuzden cc'yi de kontrol edip fp16'ya zorla.
        major = torch.cuda.get_device_capability()[0]
        bf16_ok = torch.cuda.is_bf16_supported() and major >= 8
        ptdtype = torch.bfloat16 if bf16_ok else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=ptdtype)
        # Windows + torch 2.12 CUDA'da fused AdamW ile GradScaler bazen
        # "Expected grad_scale and found_inf to be None" hatasina dusuyor.
        # Autocast hizini koruyup scaler'i kapatmak kisa SFT kosularini stabil tutar.
        use_scaler = (ptdtype == torch.float16) and sys.platform != "win32"
        log(f"[train] GPU: {torch.cuda.get_device_name()} | "
            f"autocast={ptdtype} | scaler={use_scaler}")
    elif device_type == "mps":
        ptdtype = torch.float32
        ctx = nullcontext()
        use_scaler = False
        log("[train] Apple MPS kullaniliyor; mixed precision kapali.")
    else:
        ptdtype = torch.float32
        ctx = nullcontext()
        use_scaler = False
        log("[train] UYARI: CUDA/MPS yok, CPU'da egitim cok yavastir (sadece test).")

    # torch.amp.GradScaler takma adi 2.3.0+ ile geldi; eski surumlerde fallback
    _GradScaler = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    scaler = _GradScaler(enabled=use_scaler)

    # --- veri yukleyici (memmap) ---
    val_fallback_logged = False

    def get_batch(split):
        nonlocal val_fallback_logged
        # her cagrida memmap'i yeniden ac (bellek sizintisini onler)
        fp = os.path.join(args.data, f"{split}.bin")
        data = np.memmap(fp, dtype=np_dtype, mode="r")
        max_start = len(data) - cfg.block_size - 1
        if max_start < 1:
            if split == "val":
                train_fp = os.path.join(args.data, "train.bin")
                train_data = np.memmap(train_fp, dtype=np_dtype, mode="r")
                train_max_start = len(train_data) - cfg.block_size - 1
                if train_max_start >= 1:
                    if not val_fallback_logged:
                        log(f"[data] val.bin cok kucuk; eval icin train.bin kullaniliyor "
                            f"(block_size={cfg.block_size}).")
                        val_fallback_logged = True
                    data = train_data
                    max_start = train_max_start
                else:
                    raise SystemExit(f"{split}.bin ve train.bin cok kucuk (block_size={cfg.block_size}).")
            if max_start < 1:
                raise SystemExit(f"{split}.bin cok kucuk (block_size={cfg.block_size}).")
        ix = torch.randint(max_start, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # --- model ---
    model = GPT(cfg).to(device)
    if ckpt is not None:
        incompatible = model.load_state_dict(ckpt["model"], strict=not args.adapter)
        if args.adapter:
            missing = [key for key in incompatible.missing_keys if key.startswith("adapter.")]
            unexpected = [key for key in incompatible.unexpected_keys if key.startswith("adapter.")]
            if missing:
                log(f"[adapter] yeni adapter baslatildi ({len(missing)} tensor).")
            if unexpected:
                log(f"[adapter] base checkpoint icinde beklenmeyen adapter tensorleri: {len(unexpected)}")
    if args.adapter_resume:
        adapter_ckpt = torch.load(args.adapter_resume, map_location=device, weights_only=False)
        state = adapter_ckpt.get("adapter") or adapter_ckpt.get("adapter_state_dict")
        if state is None:
            raise SystemExit(f"adapter state bulunamadi: {args.adapter_resume}")
        model.adapter.load_state_dict(state)
        loaded_step = adapter_ckpt.get("step", 0)
        if args.adapter_continue_step:
            start_step = loaded_step
            best_val = adapter_ckpt.get("best_val", best_val)
        log(f"[adapter] agirlik yuklendi: {args.adapter_resume} "
            f"(ckpt_step={loaded_step}, run_start={start_step})")
    if args.adapter:
        model.freeze_base_model(train_adapter=True)
    n = model.num_params()
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[train] model parametreleri: {n/1e6:.1f}M  "
        f"(preset={args.preset}, vocab={cfg.vocab_size}, ctx={cfg.block_size})")
    if args.adapter:
        log(f"[adapter] trainable parametre: {trainable_n:,} "
            f"(dim={cfg.adapter_dim}, lr={cfg.learning_rate:g})")

    optimizer = model.configure_optimizers(cfg, device_type)
    if ckpt is not None and "optimizer" in ckpt and not args.adapter:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            log(f"[train] optimizer durumu yuklenemedi ({e}); sifirdan.")

    if args.compile:
        log("[train] torch.compile ediliyor (ilk adim yavas olabilir)...")
        model = torch.compile(model)

    if dist_info["enabled"]:
        ddp_kwargs = {}
        if device_type == "cuda":
            cuda_index = torch.cuda.current_device()
            ddp_kwargs = {"device_ids": [cuda_index], "output_device": cuda_index}
        model = DDP(model, **ddp_kwargs)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = get_batch(split)
                with ctx:
                    _, loss, _ = model(x, y)
                losses[k] = loss.item()
            mean_loss = losses.mean().item()
            if dist_info["enabled"]:
                reduce_device = device if dist_info["backend"] == "nccl" else "cpu"
                loss_t = torch.tensor(mean_loss, dtype=torch.float64, device=reduce_device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                mean_loss = (loss_t / world_size).item()
            out[split] = mean_loss
        model.train()
        return out

    # --- egitim dongusu ---
    model.train()
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps * world_size
    t0 = time.time()
    running = None
    saved_best_adapter = False

    log(f"[train] efektif batch = {cfg.batch_size} x {cfg.grad_accum_steps} x {world_size} "
        f"= {cfg.batch_size * cfg.grad_accum_steps * world_size} | {tokens_per_step:,} token/adim")

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum_steps):
            sync_gradients = micro == cfg.grad_accum_steps - 1
            sync_ctx = model.no_sync() if dist_info["enabled"] and not sync_gradients else nullcontext()
            with sync_ctx:
                x, y = get_batch("train")
                with ctx:
                    _, loss, _ = model(x, y)
                    loss = loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        lossf = loss.item() * cfg.grad_accum_steps
        running = lossf if running is None else 0.9 * running + 0.1 * lossf

        if is_main and step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * cfg.log_interval / dt if step > start_step else 0
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device_type == "cuda" else 0
            print(f"adim {step:>6} | loss {lossf:6.3f} (ort {running:6.3f}) | "
                  f"lr {lr:.2e} | {tps:7.0f} tok/s | VRAM {mem:4.2f} GB")

        if step > start_step and step % cfg.eval_interval == 0:
            ev = estimate_loss()
            log(f"  >> eval | train {ev['train']:.3f} | val {ev['val']:.3f}")
            if is_main and ev["val"] < best_val:
                best_val = ev["val"]
                if args.adapter:
                    save_adapter(model, optimizer, cfg, step, best_val, cfg.out_dir, "adapter.pt", args.resume)
                    saved_best_adapter = True
                    log(f"  >> en iyi val ({best_val:.3f}) kaydedildi -> {cfg.out_dir}/adapter.pt")
                else:
                    save_ckpt(model, optimizer, cfg, step, best_val, cfg.out_dir, "ckpt.pt")
                    log(f"  >> en iyi val ({best_val:.3f}) kaydedildi -> {cfg.out_dir}/ckpt.pt")

        if is_main and step > start_step and step % cfg.save_interval == 0:
            if args.adapter:
                save_adapter(model, optimizer, cfg, step, best_val, cfg.out_dir, "adapter_last.pt", args.resume)
            else:
                save_ckpt(model, optimizer, cfg, step, best_val, cfg.out_dir, "ckpt_last.pt")

    # son kayit
    if is_main:
        if args.adapter:
            save_adapter(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "adapter_last.pt", args.resume)
            final_adapter = os.path.join(cfg.out_dir, "adapter.pt")
            if not saved_best_adapter:
                save_adapter(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "adapter.pt", args.resume)
                log("[adapter] final adapter adapter.pt olarak kaydedildi.")
            log(f"[adapter] bitti. adapter checkpoint'leri: {cfg.out_dir}/")
        else:
            save_ckpt(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "ckpt_last.pt")
            # eval hic calismadiysa (kisa kosu) ckpt.pt olusmamis olabilir -> garanti et
            final_ckpt = os.path.join(cfg.out_dir, "ckpt.pt")
            if not os.path.exists(final_ckpt):
                save_ckpt(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "ckpt.pt")
                log("[train] ckpt.pt yoktu, son model ckpt.pt olarak da kaydedildi.")
            log(f"[train] bitti. checkpoint'ler: {cfg.out_dir}/")

    if dist_info["enabled"]:
        dist.barrier()
        dist.destroy_process_group()


def save_ckpt(model, optimizer, cfg, step, best_val, out_dir, name):
    raw = unwrap_model(model)  # DDP / torch.compile sarmali
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val": best_val,
    }, os.path.join(out_dir, name))


def save_adapter(model, optimizer, cfg, step, best_val, out_dir, name, base_ckpt):
    raw = unwrap_model(model)  # DDP / torch.compile sarmali
    if raw.adapter is None:
        raise RuntimeError("adapter yok; save_adapter cagrilamaz")
    torch.save({
        "adapter": raw.adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "adapter_config": {
            "adapter_dim": cfg.adapter_dim,
            "adapter_dropout": cfg.adapter_dropout,
            "adapter_scale": cfg.adapter_scale,
            "n_embd": cfg.n_embd,
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
        },
        "base_checkpoint": base_ckpt,
        "step": step,
        "best_val": best_val,
        "note": "base LLM donduruldu; sadece neural adapter egitildi",
    }, os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
