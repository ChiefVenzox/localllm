"""
data/prepare_data.py
====================
Ham metni token id'lerine cevirip egitim icin ikili (binary) dosyalara yazar.
Iki mod:

  --mode pretrain   (.txt dosyalari)  her dosya/paragraf bir dokuman; aralara
                    <|endoftext|> konur.
  --mode chat       (.jsonl dosyalari) her satir {"messages":[...]} ; sohbet
                    sablonuyla kodlanir.

Cikti: <out>/train.bin, <out>/val.bin (memmap), <out>/meta.json

Kullanim:
    python -m data.prepare_data --mode pretrain --input data/raw
    python -m data.prepare_data --mode chat --input data/chat --out data/chat_bin
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm

from tokenizer import load_tokenizer
from chat_template import encode_chat, DEFAULT_SYSTEM


def list_files(inputs, exts):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            for e in exts:
                files += glob.glob(os.path.join(item, "**", f"*{e}"), recursive=True)
        elif os.path.isfile(item):
            files.append(item)
    return sorted(set(files))


def iter_pretrain_docs(files):
    """Her .txt dosyasini blank-line ile dokumanlara boler."""
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            buf = []
            for line in f:
                if line.strip() == "" and buf:
                    yield "".join(buf).strip()
                    buf = []
                else:
                    buf.append(line)
            if buf:
                yield "".join(buf).strip()


def iter_chat_convos(files):
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj["messages"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "chat"], default="pretrain")
    ap.add_argument("--input", nargs="+", default=["data/raw"])
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/bin")
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--system", default=None,
                    help="chat modunda her konusmaya eklenecek sistem mesaji")
    ap.add_argument("--default-system", action="store_true",
                    help="chat modunda Bilge persona'sini (DEFAULT_SYSTEM) ekle")
    args = ap.parse_args()
    system = DEFAULT_SYSTEM if args.default_system else args.system

    tok = load_tokenizer(args.tokenizer)
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    print(f"[data] tokenizer vocab={tok.vocab_size}, dtype={np.dtype(dtype).name}")

    exts = [".txt"] if args.mode == "pretrain" else [".jsonl"]
    files = list_files(args.input, exts)
    if not files:
        raise SystemExit(f"Girdi bulunamadi ({exts}): {args.input}")
    print(f"[data] {len(files)} dosya, mod={args.mode}")

    all_ids = []
    if args.mode == "pretrain":
        for doc in tqdm(iter_pretrain_docs(files), desc="kodlaniyor"):
            if not doc:
                continue
            all_ids.extend(tok.encode(doc))
            all_ids.append(tok.eot_id)
    else:
        for messages in tqdm(iter_chat_convos(files), desc="kodlaniyor"):
            ids = encode_chat(tok, messages, add_generation_prompt=False,
                              system=system)
            all_ids.extend(ids)
            all_ids.append(tok.eot_id)

    if not all_ids:
        raise SystemExit("Hic token uretilemedi (girdi bos olabilir).")

    arr = np.array(all_ids, dtype=dtype)
    n = len(arr)
    n_val = max(1, int(n * args.val_ratio)) if n > 100 else 0
    train, val = arr[: n - n_val], arr[n - n_val:]

    os.makedirs(args.out, exist_ok=True)
    train.tofile(os.path.join(args.out, "train.bin"))
    if n_val:
        val.tofile(os.path.join(args.out, "val.bin"))
    else:
        # cok kucuk veri: val = train (sadece duman testi icin)
        train.tofile(os.path.join(args.out, "val.bin"))

    meta = {
        "mode": args.mode,
        "dtype": np.dtype(dtype).name,
        "vocab_size": tok.vocab_size,
        "n_tokens_total": int(n),
        "n_train": int(len(train)),
        "n_val": int(n_val),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[data] toplam {n:,} token  (train={len(train):,}, val={n_val:,})")
    print(f"[data] yazildi -> {args.out}/  (train.bin, val.bin, meta.json)")


if __name__ == "__main__":
    main()
