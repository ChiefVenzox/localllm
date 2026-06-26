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
import time
from contextlib import nullcontext
from dataclasses import asdict, fields

import numpy as np
import torch

from config import GPTConfig, get_config
from model import GPT


# ---------------------------------------------------------------------------
def build_cfg_from_dict(d: dict) -> GPTConfig:
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in d.items() if k in valid})


def load_meta(data_dir: str) -> dict:
    p = os.path.join(data_dir, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"meta.json yok: {p}\nOnce: python -m data.prepare_data")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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
    ap.add_argument("--reset-best", action="store_true",
                    help="Devam ederken en-iyi-val sayacini sifirla. SFT gibi FARKLI "
                         "bir veri setine gecerken SART: yoksa eski (dusuk) best_val "
                         "yuzunden yeni ckpt.pt hic kaydedilmez.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr-decay-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile (Linux'ta hizli)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta = load_meta(args.data)
    np_dtype = np.dtype(meta["dtype"])

    # --- config ---
    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = build_cfg_from_dict(ckpt["config"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        if args.reset_best:
            best_val = float("inf")
            print("[train] best_val sifirlandi (yeni veri seti icin temiz ckpt.pt).")
        print(f"[train] devam ediliyor: {args.resume} (step={start_step})")
    else:
        cfg = get_config(args.preset)
        ckpt = None

    # veriye gore vocab'i sabitle
    cfg.vocab_size = meta["vocab_size"]
    if args.out:
        cfg.out_dir = args.out
    if args.max_steps:
        cfg.max_steps = args.max_steps
        cfg.lr_decay_steps = args.max_steps   # decay'i kosu uzunluguyla esle
    if args.lr_decay_steps:
        cfg.lr_decay_steps = args.lr_decay_steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.grad_accum:
        cfg.grad_accum_steps = args.grad_accum

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"

    # --- precision ---
    if device_type == "cuda":
        # Turing (1660 Ti, cc 7.5) bf16'yi sadece EMULE eder (yavas). Gercek bf16
        # hizlandirmasi yalniz Ampere+ (cc >= 8.0). Yeni torch is_bf16_supported()
        # emulasyonu da True sayar; bu yuzden cc'yi de kontrol edip fp16'ya zorla.
        major = torch.cuda.get_device_capability()[0]
        bf16_ok = torch.cuda.is_bf16_supported() and major >= 8
        ptdtype = torch.bfloat16 if bf16_ok else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=ptdtype)
        use_scaler = (ptdtype == torch.float16)
        print(f"[train] GPU: {torch.cuda.get_device_name(0)} | "
              f"autocast={ptdtype} | scaler={use_scaler}")
    else:
        ptdtype = torch.float32
        ctx = nullcontext()
        use_scaler = False
        print("[train] UYARI: CUDA yok, CPU'da egitim cok yavastir (sadece test).")

    # torch.amp.GradScaler takma adi 2.3.0+ ile geldi; eski surumlerde fallback
    _GradScaler = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    scaler = _GradScaler(enabled=use_scaler)

    # --- veri yukleyici (memmap) ---
    def get_batch(split):
        # her cagrida memmap'i yeniden ac (bellek sizintisini onler)
        fp = os.path.join(args.data, f"{split}.bin")
        data = np.memmap(fp, dtype=np_dtype, mode="r")
        max_start = len(data) - cfg.block_size - 1
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
        model.load_state_dict(ckpt["model"])
    n = model.num_params()
    print(f"[train] model parametreleri: {n/1e6:.1f}M  "
          f"(preset={args.preset}, vocab={cfg.vocab_size}, ctx={cfg.block_size})")

    optimizer = model.configure_optimizers(cfg, device_type)
    if ckpt is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[train] optimizer durumu yuklenemedi ({e}); sifirdan.")

    if args.compile:
        print("[train] torch.compile ediliyor (ilk adim yavas olabilir)...")
        model = torch.compile(model)

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
            out[split] = losses.mean().item()
        model.train()
        return out

    # --- egitim dongusu ---
    model.train()
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps
    t0 = time.time()
    running = None

    print(f"[train] efektif batch = {cfg.batch_size} x {cfg.grad_accum_steps} "
          f"= {cfg.batch_size * cfg.grad_accum_steps} | {tokens_per_step:,} token/adim")

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum_steps):
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

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * cfg.log_interval / dt if step > start_step else 0
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device_type == "cuda" else 0
            print(f"adim {step:>6} | loss {lossf:6.3f} (ort {running:6.3f}) | "
                  f"lr {lr:.2e} | {tps:7.0f} tok/s | VRAM {mem:4.2f} GB")

        if step > start_step and step % cfg.eval_interval == 0:
            ev = estimate_loss()
            print(f"  >> eval | train {ev['train']:.3f} | val {ev['val']:.3f}")
            if ev["val"] < best_val:
                best_val = ev["val"]
                save_ckpt(model, optimizer, cfg, step, best_val, cfg.out_dir, "ckpt.pt")
                print(f"  >> en iyi val ({best_val:.3f}) kaydedildi -> {cfg.out_dir}/ckpt.pt")

        if step > start_step and step % cfg.save_interval == 0:
            save_ckpt(model, optimizer, cfg, step, best_val, cfg.out_dir, "ckpt_last.pt")

    # son kayit
    save_ckpt(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "ckpt_last.pt")
    # eval hic calismadiysa (kisa kosu) ckpt.pt olusmamis olabilir -> garanti et
    final_ckpt = os.path.join(cfg.out_dir, "ckpt.pt")
    if not os.path.exists(final_ckpt):
        save_ckpt(model, optimizer, cfg, cfg.max_steps, best_val, cfg.out_dir, "ckpt.pt")
        print("[train] ckpt.pt yoktu, son model ckpt.pt olarak da kaydedildi.")
    print(f"[train] bitti. checkpoint'ler: {cfg.out_dir}/")


def save_ckpt(model, optimizer, cfg, step, best_val, out_dir, name):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model  # torch.compile sarmali
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val": best_val,
    }, os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
