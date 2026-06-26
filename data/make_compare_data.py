"""
data/make_compare_data.py
========================
Sayi KIYASLAMA verisi: "X mi Y mi buyuk/kucuk" tarzi sorulari sistematik uretir.
Amac: modelin kiyasi TOPLAMA ile karistirmamasi (matematik deseni baskin cikiyordu).
Bu yuzden cevap formati toplamadan KASITLI olarak farkli ("daha buyuktur/kucuktur").

Cikti: data/chat_compare/compare.jsonl

Kullanim:
    python -m data.make_compare_data
"""
from __future__ import annotations
import json
import os
import random

random.seed(11)

MAXN = 20   # 0..20 arasi sayilar


def _convo(q, a):
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main():
    os.makedirs("data/chat_compare", exist_ok=True)
    seen, out = set(), []

    def add(q, a):
        k = q.lower()
        if k not in seen:
            seen.add(k)
            out.append(_convo(q, a))

    for x in range(0, MAXN + 1):
        for y in range(0, MAXN + 1):
            if x == y:
                continue
            big, small = max(x, y), min(x, y)
            # BUYUK sorulari
            ans_b = f"Dusunelim: {big}, {small}'ten buyuktur. Sonuc: {big} daha buyuktur."
            for q in random.sample([
                f"{x} mi {y} mi daha buyuk?",
                f"{x} mi {y} mi buyuk?",
                f"Hangisi daha buyuk, {x} mi {y} mi?",
                f"{x} ile {y}'den hangisi buyuk?",
            ], 2):
                add(q, ans_b)
            # KUCUK sorulari (daha az: cesitlilik icin %40)
            if random.random() < 0.4:
                ans_s = f"Dusunelim: {small}, {big}'ten kucuktur. Sonuc: {small} daha kucuktur."
                q = random.choice([
                    f"{x} mi {y} mi daha kucuk?",
                    f"Hangisi daha kucuk, {x} mi {y} mi?",
                ])
                add(q, ans_s)

    # cok-fazlaysa makul sayiya indir (denge icin)
    random.shuffle(out)
    out = out[:700]

    path = "data/chat_compare/compare.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[compare] {len(out)} kiyas konusmasi yazildi -> {path}")
    for c in out[:3]:
        print("  S:", c["messages"][0]["content"], "| C:", c["messages"][1]["content"])


if __name__ == "__main__":
    main()
