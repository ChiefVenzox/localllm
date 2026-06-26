"""
scripts/vram_probe.py
=====================
Bir preset'in EGITIM sirasinda ne kadar VRAM yedigini gercek GPU'da olcer.
Veri/tokenizer gerektirmez: rastgele token batch'i ile birkac ileri+geri adim
calistirip torch.cuda.max_memory_allocated degerini raporlar.

Kullanim:
    python scripts/vram_probe.py --preset small-100m
    python scripts/vram_probe.py --preset small-100m --batch-size 4
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

# proje kokunu import yoluna ekle (scripts/ alt klasorunden calistirilinca)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config        # noqa: E402
from model import GPT                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small-100m")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="gradient checkpointing'i kapat (daha hizli, daha cok VRAM)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA yok; bu arac GPU VRAM'i olcer. --device gerekmez.")

    over = {}
    if args.batch_size:
        over["batch_size"] = args.batch_size
    if args.block_size:
        over["block_size"] = args.block_size
    if args.no_checkpoint:
        over["gradient_checkpointing"] = False
    cfg = get_config(args.preset, **over)

    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    model = GPT(cfg).to(dev)
    opt = model.configure_optimizers(cfg, "cuda")
    # Turing'de bf16 emule (yavas) -> sadece Ampere+ (cc>=8) icin bf16 kullan
    bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
    ptdtype = torch.bfloat16 if bf16 else torch.float16
    _GradScaler = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    scaler = _GradScaler(enabled=(ptdtype == torch.float16))

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), device=dev)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.block_size), device=dev)

    model.train()
    print(f"preset={args.preset} | params={model.num_params()/1e6:.0f}M | "
          f"batch={cfg.batch_size} | ctx={cfg.block_size} | "
          f"grad_ckpt={cfg.gradient_checkpointing} | dtype={ptdtype}")
    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU toplam: {total/1e9:.2f} GB | bos (baslangic): {free_before/1e9:.2f} GB")

    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=ptdtype):
            _, loss, _ = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    print(f"--> ZIRVE VRAM (allocated): {peak:.2f} GB | reserved: {reserved:.2f} GB")
    print(f"--> son loss: {loss.item():.3f}")
    if reserved < (total / 1e9) - 0.3:
        print("[OK] Bu ayar 6 GB'a SIGIYOR.")
    else:
        print("[DIKKAT] Sinira cok yakin/asabilir; --batch-size dusur.")


if __name__ == "__main__":
    main()
