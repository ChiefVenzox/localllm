"""
data/make_sample_data.py
=========================
KUCUK bir ornek korpus uretir (Turkce + Ingilizce + Python kodu) ve kucuk bir
sohbet (.jsonl) dosyasi yazar. AMACI: boru hattini (tokenizer -> veri -> egitim
-> sohbet) ucdan uca test etmek. GERCEK egitim icin yetersizdir; data/raw/
icine kendi buyuk metinlerini koymalisin (Wikipedia dumpu, kitaplar, kod vb.).

Kullanim:
    python -m data.make_sample_data
"""
from __future__ import annotations
import json
import os

TR = [
    "Merhaba, bugun hava cok guzel ve disarida yuruyuse cikmak istiyorum.",
    "Yapay zeka, bilgisayarlarin insan gibi ogrenmesini saglayan bir alandir.",
    "Turkiye'nin baskenti Ankara'dir, en kalabalik sehri ise Istanbul'dur.",
    "Kitap okumak hem bilgi verir hem de hayal gucunu gelistirir.",
    "Kahvalti gunun en onemli ogunudur diye dusunuyorum.",
    "Bu aksam arkadaslarimla sinemaya gitmeyi planliyoruz.",
    "Ogrenmek istedigim sey, dil modellerinin nasil egitildigidir.",
    "Yaz tatilinde deniz kenarinda dinlenmek cok keyifli olur.",
]
EN = [
    "Hello, the weather is nice today and I want to go for a walk.",
    "Artificial intelligence is a field that lets computers learn like humans.",
    "Reading books gives you knowledge and improves your imagination.",
    "A language model predicts the next token given the previous ones.",
    "Training a neural network requires data, compute, and patience.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "I enjoy solving problems and writing clean, readable code.",
    "Summer holidays by the sea are relaxing and full of joy.",
]
CODE = [
    "def topla(a, b):\n    \"\"\"Iki sayiyi toplar.\"\"\"\n    return a + b",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "for i in range(10):\n    if i % 2 == 0:\n        print(f'{i} cifttir')\n    else:\n        print(f'{i} tektir')",
    "class Sayac:\n    def __init__(self):\n        self.deger = 0\n    def artir(self):\n        self.deger += 1\n        return self.deger",
    "import math\n\ndef daire_alani(r):\n    return math.pi * r ** 2",
    "names = ['ali', 'veli', 'ayse']\nfor name in names:\n    print(name.capitalize())",
]

CHAT = [
    {"messages": [
        {"role": "user", "content": "Merhaba, nasilsin?"},
        {"role": "assistant", "content": "Merhaba! Ben bir yapay zeka asistaniyim, gayet iyiyim. Sana nasil yardimci olabilirim?"},
    ]},
    {"messages": [
        {"role": "user", "content": "Python'da iki sayiyi toplayan bir fonksiyon yaz."},
        {"role": "assistant", "content": "Tabii:\n\ndef topla(a, b):\n    return a + b\n\nBu fonksiyon a ve b'yi alip toplamlarini dondurur."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is a language model?"},
        {"role": "assistant", "content": "A language model is a system that predicts the next token in a sequence, learning patterns from text data."},
    ]},
    {"messages": [
        {"role": "user", "content": "Fibonacci dizisini hesaplayan kod verir misin?"},
        {"role": "assistant", "content": "Elbette:\n\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
    ]},
]


def main(repeat: int = 200):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/chat", exist_ok=True)

    # pretrain korpusu: paragraflar bos satirla ayrilir
    paras = []
    pool = TR + EN + CODE
    for r in range(repeat):
        for i, p in enumerate(pool):
            # cesitlilik icin sirayi degistir
            paras.append(p)
    text = "\n\n".join(paras)
    with open("data/raw/sample.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # chat verisi
    with open("data/chat/sample.jsonl", "w", encoding="utf-8") as f:
        for r in range(repeat // 4 + 1):
            for c in CHAT:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    nbytes = os.path.getsize("data/raw/sample.txt")
    print(f"[sample] data/raw/sample.txt yazildi (~{nbytes/1024:.0f} KB)")
    print(f"[sample] data/chat/sample.jsonl yazildi")
    print("[sample] NOT: bu sadece DUMAN TESTI icindir; gercek egitim icin "
          "data/raw/ icine kendi buyuk metinlerini ekle.")


if __name__ == "__main__":
    main()
