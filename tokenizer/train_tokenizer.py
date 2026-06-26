"""
tokenizer/train_tokenizer.py
============================
KENDI BPE tokenizer'imizi sifirdan egitir (Turkce + Ingilizce + kod).
Byte-level BPE kullaniriz: her UTF-8 byte'i bildigi icin Turkce karakterler
(c, g, i, o, s, u ...) ve kaynak kod sembolleri sorunsuz islenir; <unk> yok.

Kullanim:
    python -m tokenizer.train_tokenizer --input data/raw --vocab-size 32000
    python -m tokenizer.train_tokenizer --input a.txt b.txt --vocab-size 16000

Cikti: tokenizer/tokenizer.json
"""
from __future__ import annotations
import argparse
import glob
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from config import SPECIAL_TOKENS


def collect_files(inputs):
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files += glob.glob(os.path.join(item, "**", "*.txt"), recursive=True)
            files += glob.glob(os.path.join(item, "**", "*.jsonl"), recursive=True)
        elif os.path.isfile(item):
            files.append(item)
    files = sorted(set(files))
    if not files:
        raise SystemExit(
            f"Egitim metni bulunamadi: {inputs}\n"
            f"Once data/raw/ icine .txt dosyalari koy "
            f"(ornek: python -m data.make_sample_data)."
        )
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=["data/raw"],
                    help="Klasor(ler) veya dosya(lar)")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--output", default="tokenizer/tokenizer.json")
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    files = collect_files(args.input)
    print(f"[tokenizer] {len(files)} dosya uzerinde egitiliyor "
          f"(vocab={args.vocab_size}) ...")

    tk = Tokenizer(BPE(unk_token=None))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tk.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=list(SPECIAL_TOKENS),   # id 0..4 sirayla atanir
        initial_alphabet=ByteLevel.alphabet(), # 256 byte -> her seyi kodlayabilir
        show_progress=True,
    )
    tk.train(files, trainer)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tk.save(args.output)
    print(f"[tokenizer] kaydedildi -> {args.output}  "
          f"(gercek vocab={tk.get_vocab_size()})")
    # dogrulama
    sample = "Merhaba dunya! def selamla(): print('hello')  # Turkce: cgiosu"
    enc = tk.encode(sample)
    print(f"[test] '{sample}'")
    print(f"       -> {len(enc.ids)} token -> geri: {tk.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()
