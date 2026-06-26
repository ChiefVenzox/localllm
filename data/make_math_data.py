"""
data/make_math_data.py
======================
Bilge'nin temel aritmetigi GERCEKTEN bilmesi icin deterministik bir matematik
sohbet seti uretir (toplama/cikarma/carpma + birkac problem cumlesi), hepsi
"Dusunelim: ... Sonuc: ..." adim-adim formatinda. Kucuk model bunlari ezberler
ve gorulen sayilar icin dogru cevap verir.

Cikti: data/chat_math/math.jsonl

Kullanim:
    python -m data.make_math_data
"""
from __future__ import annotations
import json
import os
import random

random.seed(7)

ADD_MAX = 12      # 0..12 toplama (13x13 = 169 cift)
SUB_MAX = 12      # eksilen 0..12
MUL_MAX = 10      # 0..10 carpma


def _convo(q, a):
    return {"messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def addition():
    out = []
    for x in range(0, ADD_MAX + 1):
        for y in range(0, ADD_MAX + 1):
            z = x + y
            ans = f"Dusunelim: {x} ile {y}'yi toplariz. {x} + {y} = {z}. Sonuc: {z}."
            qs = [f"{x} ile {y} toplami kac eder?", f"{x} + {y} kac eder?",
                  f"{x} arti {y} kac?", f"{x} ve {y}'yi toplar misin?"]
            for q in random.sample(qs, 2):
                out.append(_convo(q, ans))
    return out


def subtraction():
    out = []
    for x in range(0, SUB_MAX + 1):
        for y in range(0, x + 1):           # negatif olmasin
            z = x - y
            ans = f"Dusunelim: {x}'ten {y} cikaririz. {x} - {y} = {z}. Sonuc: {z}."
            qs = [f"{x} eksi {y} kac eder?", f"{x} - {y} kac?",
                  f"{x}'ten {y} cikarsak kac kalir?"]
            for q in random.sample(qs, 2):
                out.append(_convo(q, ans))
    return out


def multiplication():
    out = []
    for x in range(0, MUL_MAX + 1):
        for y in range(0, MUL_MAX + 1):
            z = x * y
            ans = f"Dusunelim: {x} kere {y} = {z}. Sonuc: {z}."
            qs = [f"{x} carpi {y} kac eder?", f"{x} x {y} kac?",
                  f"{x} kere {y} kac eder?"]
            for q in random.sample(qs, 2):
                out.append(_convo(q, ans))
    return out


def word_problems():
    """Problem cumleleri: format genellesin diye birkac sablon."""
    out = []
    nouns = [("elma", "elma"), ("kalem", "kalem"), ("kitap", "kitap"),
             ("kus", "kus"), ("sayi", "sayi"), ("lira", "lira")]
    for a in range(2, 13):
        for b in range(1, 9):
            for sing, _ in random.sample(nouns, 2):
                z = a + b
                q = f"Bir yerde {a} {sing} var, {b} tane daha eklersek kac {sing} olur?"
                ans = (f"Dusunelim: Once {a} {sing} vardi, uzerine {b} ekledik. "
                       f"{a} + {b} = {z}. Sonuc: {z} {sing}.")
                out.append(_convo(q, ans))
    # cikarma problemleri
    for a in range(3, 13):
        for b in range(1, a):
            sing = random.choice(nouns)[0]
            z = a - b
            q = f"Elimde {a} {sing} vardi, {b} tanesini verdim. Kac {sing} kaldi?"
            ans = (f"Dusunelim: {a} {sing}'den {b} tanesini cikaririz. "
                   f"{a} - {b} = {z}. Sonuc: {z} {sing}.")
            out.append(_convo(q, ans))
    return out


def main():
    os.makedirs("data/chat_math", exist_ok=True)
    data = []
    data += addition()
    data += subtraction()
    data += multiplication()
    data += word_problems()
    # tekille (ayni ilk soru tekrar etmesin)
    seen, uniq = set(), []
    random.shuffle(data)
    for c in data:
        k = c["messages"][0]["content"].lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    path = "data/chat_math/math.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in uniq:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[math] {len(uniq)} matematik konusmasi yazildi -> {path}")
    # ornek
    for c in uniq[:3]:
        print("  S:", c["messages"][0]["content"], "| C:", c["messages"][1]["content"])


if __name__ == "__main__":
    main()
