"""
generate.py
===========
Egitilmis checkpoint'i yukleyip metin/sohbet uretir. Hem terminal sohbeti hem
de web sunucusu (server/app.py) buradaki fonksiyonlari kullanir.

Kullanim:
    python generate.py --prompt "Bir varmis bir yokmus"      # metin tamamlama
    python generate.py --chat                                # terminal sohbeti
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import fields
from typing import Iterator, List, Dict

import torch

from config import GPTConfig
from model import GPT
from tokenizer import load_tokenizer
from chat_template import encode_chat, DEFAULT_SYSTEM


def _cfg_from_dict(d: dict) -> GPTConfig:
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in d.items() if k in valid})


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _cfg_from_dict(ckpt["config"])
    cfg.gradient_checkpointing = False  # uretimde gerek yok
    cfg.dropout = 0.0
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _decode_stream(model, tok, ids: List[int], device: str,
                   max_new_tokens: int, temperature: float,
                   top_k: int, top_p: float, stop_ids=None) -> Iterator[str]:
    """Token id listesinden baslayip metni parca parca (UTF-8 guvenli) yield eder."""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_ids = stop_ids if stop_ids is not None else tok.stop_ids
    generated: List[int] = []
    prev_text = ""
    for tok_tensor in model.generate_stream(
        idx, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, eos_token_id=None, use_cache=True,
    ):
        tid = int(tok_tensor.item())
        if tid in stop_ids:
            break
        generated.append(tid)
        text = tok.decode(generated, skip_special=True)
        if text.endswith("�"):   # yarim UTF-8 (replacement char); sonraki token'i bekle
            continue
        new = text[len(prev_text):]
        prev_text = text
        if new:
            yield new


def generate_text(model, tok, prompt: str, device: str, **kw) -> Iterator[str]:
    ids = [tok.eot_id] + tok.encode(prompt)
    yield from _decode_stream(model, tok, ids, device, **kw)


def chat_stream(model, tok, messages: List[Dict[str, str]], device: str,
                system: str | None = DEFAULT_SYSTEM, **kw) -> Iterator[str]:
    ids = encode_chat(tok, messages, add_generation_prompt=True, system=system)
    # Baglama sigmazsa EN ESKI tokenlardan kirp ama prompt'u asla yok etme:
    # uretim icin en az 1 slot kalsin (model zaten block_size'a gelince durur).
    # Onceki "block_size - max_new_tokens" formulu, max_new_tokens >= block_size
    # iken prompt'u tek tokena dusururdu (sohbeti bozan hata).
    max_ctx = model.block_size - 1
    if len(ids) > max_ctx:
        ids = ids[len(ids) - max_ctx:]
    yield from _decode_stream(model, tok, ids, device, **kw)


def main():
    # Windows konsolu (cp1254) UTF-8/replacement char basamaz -> cokmeyi onle
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    model, cfg = load_model(args.ckpt, args.device)
    print(f"[generate] model yuklendi ({model.num_params()/1e6:.0f}M, "
          f"ctx={cfg.block_size}, device={args.device})")
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, top_p=args.top_p)

    if args.chat:
        print("Sohbet modu. Cikmak icin 'cik' yaz.\n")
        history: List[Dict[str, str]] = []
        while True:
            try:
                user = input("Sen: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in ("cik", "exit", "quit"):
                break
            history.append({"role": "user", "content": user})
            print("AI: ", end="", flush=True)
            pieces = []
            for piece in chat_stream(model, tok, history, args.device, **gen_kw):
                print(piece, end="", flush=True)
                pieces.append(piece)
            print("\n")
            history.append({"role": "assistant", "content": "".join(pieces)})
    else:
        prompt = args.prompt or "Bir varmis bir yokmus,"
        print(prompt, end="", flush=True)
        for piece in generate_text(model, tok, prompt, args.device, **gen_kw):
            print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
