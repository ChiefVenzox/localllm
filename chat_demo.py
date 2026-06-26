"""
chat_demo.py
============
Egitilmis modeli HIZLICA, etkilesimsiz (non-interactive) denemek icin kucuk bir
surucu. Bir soru listesini sirayla modele sorar ve cevaplari yazar. Boylece
"chatbot calisiyor mu?" sorusunu tek komutla gorebilirsin.

Kullanim:
    python chat_demo.py
    python chat_demo.py --ckpt checkpoints/ckpt.pt --temperature 0.0
"""
from __future__ import annotations
import argparse
import sys

import torch

from generate import load_model, chat_stream
from tokenizer import load_tokenizer
from chat_template import DEFAULT_SYSTEM

# Egitim verisindeki ornek sorular (model bunlari ezberlemis olmali) +
# birkac varyasyon (genelleme ne kadar zayif onu da gormek icin).
DEFAULT_QUESTIONS = [
    "Merhaba, nasilsin?",
    "Python'da iki sayiyi toplayan bir fonksiyon yaz.",
    "What is a language model?",
    "Fibonacci dizisini hesaplayan kod verir misin?",
]


def main():
    # Windows konsolu Turkce/UTF-8 basabilsin
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (ezberi birebir verir), >0 = cesitlilik")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--system", default=None,
                    help="Sistem mesaji. SFT verisi sistemsizdi -> varsayilan: yok.")
    ap.add_argument("--default-system", action="store_true",
                    help="Sunucunun gonderdigi DEFAULT_SYSTEM ile dene (uyum testi).")
    ap.add_argument("--questions", nargs="*", default=None)
    args = ap.parse_args()
    if args.default_system:
        args.system = DEFAULT_SYSTEM

    tok = load_tokenizer(args.tokenizer)
    model, cfg = load_model(args.ckpt, args.device)
    print(f"[demo] model: {model.num_params()/1e6:.1f}M | ctx={cfg.block_size} | "
          f"device={args.device} | temp={args.temperature}\n")

    questions = args.questions or DEFAULT_QUESTIONS
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=0, top_p=1.0)

    for q in questions:
        messages = [{"role": "user", "content": q}]
        answer = "".join(chat_stream(model, tok, messages, args.device,
                                     system=args.system, **gen_kw))
        print(f"Sen: {q}")
        print(f"AI : {answer.strip()}")
        print("-" * 60)


if __name__ == "__main__":
    main()
