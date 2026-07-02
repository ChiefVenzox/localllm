"""
data/make_chat_tokens.py
========================
Tokenizer ile olculen hedef token sayisina gore sentetik sohbet verisi uretir.

Varsayilan cikti:
    data/chat_100k/auto_100k.jsonl

Kullanim:
    python -m data.make_chat_tokens --target-tokens 100000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from chat_template import DEFAULT_SYSTEM, encode_chat
from data.make_chat_50k import (
    SEED,
    build_code,
    build_compare,
    build_everyday,
    build_knowledge,
    build_language,
    build_logic,
    build_math,
    build_safety,
    build_social,
    build_units,
    fmt_num,
)
from tokenizer import load_tokenizer


BUILDERS = [
    (build_math, 24),
    (build_compare, 12),
    (build_units, 10),
    (build_logic, 12),
    (build_language, 12),
    (build_everyday, 10),
    (build_social, 8),
    (build_code, 7),
    (build_safety, 5),
    (build_knowledge, 4),
]


def make_row(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def row_token_count(tok, row: dict, system: str | None) -> int:
    ids = encode_chat(tok, row["messages"], add_generation_prompt=False, system=system)
    return len(ids) + 1  # eot


def build_pool(target_tokens: int, tok, system: str | None) -> list[tuple[dict, int]]:
    rows: list[dict] = []
    seen: set[str] = set()
    measured: list[tuple[dict, int]] = []
    measured_count = 0

    while measured_count < target_tokens * 1.25:
        before = len(rows)
        for builder, count in BUILDERS:
            builder(rows, seen, count)
        if len(rows) == before:
            raise SystemExit("Yeni sohbet ornegi uretilemedi; sablon havuzu tukendi.")

        for row in rows[before:]:
            tokens = row_token_count(tok, row, system)
            measured.append((row, tokens))
            measured_count += tokens
            if measured_count >= target_tokens * 1.25:
                break

    return measured


def build_large_row(i: int, rng: random.Random) -> dict:
    """Buyuk veri icin RAM dostu, deterministik sentetik sohbet satiri uretir."""
    bucket = rng.randrange(100)

    if bucket < 24:
        op = rng.choice(["toplama", "cikarma", "carpma", "bolme"])
        a = rng.randint(-1_000_000, 1_000_000)
        b = rng.randint(1, 50_000)
        if op == "toplama":
            q = rng.choice([
                "{a} + {b} kac eder?",
                "{a} ile {b} toplamini hesapla.",
                "{a} arti {b} sonucunu kisa yaz.",
            ]).format(a=a, b=b)
            ans = f"{a} + {b} = {a + b}. Sonuc: {a + b}."
        elif op == "cikarma":
            q = rng.choice([
                "{a} - {b} kac eder?",
                "{a} sayisindan {b} cikarsa kac kalir?",
                "{a} eksi {b} sonucunu hesapla.",
            ]).format(a=a, b=b)
            ans = f"{a} - {b} = {a - b}. Sonuc: {a - b}."
        elif op == "carpma":
            a = rng.randint(-10_000, 10_000)
            b = rng.randint(-500, 500)
            q = rng.choice([
                "{a} x {b} kac eder?",
                "{a} carpi {b} sonucunu bul.",
                "{a} ile {b}'nin carpimini hesapla.",
            ]).format(a=a, b=b)
            ans = f"{a} x {b} = {a * b}. Sonuc: {a * b}."
        else:
            b = rng.randint(1, 500)
            z = rng.randint(-5000, 5000)
            a = b * z
            q = rng.choice([
                "{a} / {b} kac eder?",
                "{a} sayisini {b}'ye bol.",
                "{a} icinde {b} kac kere var?",
            ]).format(a=a, b=b)
            ans = f"{a} / {b} = {z}. Sonuc: {z}."
        return make_row(q, ans)

    if bucket < 36:
        nums = [rng.randint(-50_000, 50_000) for _ in range(4)]
        kind = rng.choice(["sirala", "karsilastir", "tek_cift", "toplam", "ortalama"])
        if kind == "sirala":
            q = "Bu sayilari kucukten buyuge sirala: " + ", ".join(map(str, nums)) + "."
            ans = "Sirali hali: " + ", ".join(map(str, sorted(nums))) + "."
        elif kind == "karsilastir":
            a, b = nums[:2]
            rel = "esittir" if a == b else ("buyuktur" if a > b else "kucuktur")
            q = f"{a} ve {b} sayilarini karsilastir."
            ans = f"Karsilastirma: {a}, {b}'ye gore {rel}."
        elif kind == "tek_cift":
            n = nums[0]
            q = f"{n} tek mi cift mi?"
            ans = f"{n} {'cift' if n % 2 == 0 else 'tek'} sayidir."
        elif kind == "toplam":
            q = f"{nums[0]}, {nums[1]}, {nums[2]} sayilarinin toplami kac?"
            ans = f"Toplam: {nums[0]} + {nums[1]} + {nums[2]} = {sum(nums[:3])}."
        else:
            vals = nums[:3]
            avg = sum(vals) / 3
            q = f"{vals[0]}, {vals[1]}, {vals[2]} sayilarinin ortalamasi kac?"
            ans = f"Ortalama = {sum(vals)} / 3 = {fmt_num(avg)}."
        return make_row(q, ans)

    if bucket < 47:
        conversions = [
            ("cm", "m", lambda n: n / 100),
            ("m", "cm", lambda n: n * 100),
            ("km", "m", lambda n: n * 1000),
            ("g", "kg", lambda n: n / 1000),
            ("kg", "g", lambda n: n * 1000),
            ("dakika", "saniye", lambda n: n * 60),
            ("saat", "dakika", lambda n: n * 60),
            ("gun", "saat", lambda n: n * 24),
        ]
        src, dst, fn = rng.choice(conversions)
        n = rng.randint(1, 200_000)
        q = f"{n} {src} kac {dst} eder?"
        ans = f"{n} {src} = {fmt_num(fn(n))} {dst}."
        return make_row(q, ans)

    if bucket < 60:
        tasks = [
            "ders calisma", "oda toplama", "spor yapma", "kitap okuma",
            "mail yazma", "sunuma hazirlanma", "gunu planlama",
            "dosyalari duzenleme", "proje takibi", "kod inceleme",
        ]
        contexts = [
            "sabah", "ogle arasi", "aksam", "hafta sonu", "is cikisi",
            "enerjim azken", "odaklanmam gerekirken", "son teslimden once",
        ]
        duration = rng.choice([10, 15, 20, 25, 30, 45, 60, 90])
        task = rng.choice(tasks)
        context = rng.choice(contexts)
        q = f"{context} {duration} dakikada {task} icin uygulanabilir bir plan hazirla."
        ans = (
            f"{duration} dakikalik plan:\n"
            f"1. Ilk {max(2, duration // 6)} dakikada hedefi netlestir.\n"
            "2. Orta bolumde tek ise odaklan.\n"
            "3. Son 2 dakikada sonucu kontrol edip toparla."
        )
        return make_row(q, ans)

    if bucket < 71:
        subjects = [
            "toplanti notlari", "alisveris listesi", "odev plani",
            "spor programi", "kitap ozeti", "proje fikri",
            "seyahat hazirligi", "gunluk hedefler", "mail taslagi",
            "sunum konusu", "hata raporu", "egitim notu",
        ]
        tone = rng.choice(["sakin", "resmi", "samimi", "net", "kibar"])
        verb = rng.choice(["daha kisa yaz", "baslik oner", "uc maddeye bol", "daha net hale getir"])
        subject = rng.choice(subjects)
        raw = f"{subject} icin duzenli ve anlasilir bir plan hazirlamak istiyorum"
        q = f"Su cumleyi {tone} bir dille {verb}: {raw}."
        if verb == "baslik oner":
            ans = f"Baslik onerisi: {subject.title()} Icin Net Plan."
        elif verb == "uc maddeye bol":
            ans = f"- Hedefi belirle.\n- Gerekli adimlari sirala.\n- Son kontrol icin zaman ayir."
        elif verb == "daha kisa yaz":
            ans = f"{subject.title()} icin net bir plan hazirlamak istiyorum."
        else:
            ans = f"Amacim {subject} icin anlasilir, uygulanabilir ve sirali bir plan hazirlamak."
        return make_row(q, ans)

    if bucket < 80:
        funcs = [
            ("iki sayiyi toplayan", "return a + b", "topla"),
            ("listedeki sayilarin toplamini bulan", "return sum(sayilar)", "liste_toplam"),
            ("bir metni ters ceviren", "return metin[::-1]", "ters_cevir"),
            ("sayinin cift olup olmadigini soyleyen", "return n % 2 == 0", "cift_mi"),
            ("listenin en buyuk elemanini bulan", "return max(sayilar)", "en_buyuk"),
            ("kelime sayisini bulan", "return len(metin.split())", "kelime_say"),
        ]
        desc, body, base_name = rng.choice(funcs)
        name = f"{base_name}_{i}"
        if "sayilar" in body:
            code = f"def {name}(sayilar):\n    {body}"
        elif "metin" in body:
            code = f"def {name}(metin):\n    {body}"
        elif " n " in f" {body} ":
            code = f"def {name}(n):\n    {body}"
        else:
            code = f"def {name}(a, b):\n    {body}"
        q = f"Python'da {desc} {name} adli fonksiyon yaz."
        ans = f"Tabii:\n\n```python\n{code}\n```\n\nBu fonksiyon istenen isi dogrudan yapar."
        return make_row(q, ans)

    if bucket < 88:
        topics = [
            "bugunku dolar kuru", "yarinki hava durumu", "son dakika haberleri",
            "anlik bitcoin fiyati", "bugunku mac skoru", "guncel borsa yorumu",
            "ilac kullanimi", "yatirim karari", "hukuki dilekce", "saglik teshisi",
        ]
        topic = rng.choice(topics)
        q = f"{topic} hakkinda kesin ve guncel bilgi ver."
        ans = (
            f"{topic} guncel veya uzmanlik gerektiren bilgi olabilir. "
            "Emin olmadan uydurmam; guvenilir ve guncel bir kaynaktan kontrol etmek gerekir."
        )
        return make_row(q, ans)

    if bucket < 95:
        facts = [
            ("Su dongusu", "Su dongusu; buharlasma, yogunlasma ve yagis adimlariyla suyun dogada dolasmasidir."),
            ("Fotosentez", "Fotosentez, bitkilerin isik enerjisiyle su ve karbondioksitten besin uretmesidir."),
            ("Algoritma", "Algoritma, bir problemi cozmek icin izlenen sirali adimlar butunudur."),
            ("Fonksiyon", "Fonksiyon, belirli bir isi yapan ve tekrar kullanilabilen kod parcasidir."),
            ("Hipotez", "Hipotez, test edilebilir bir aciklama veya tahmindir."),
        ]
        name, fact = rng.choice(facts)
        q = rng.choice([
            f"{name} nedir?",
            f"{name} konusunu kisaca acikla.",
            f"{name} icin tek cumlelik aciklama yap.",
        ])
        return make_row(q, fact)

    q = rng.choice([
        "Bilmedigin konuda nasil cevap vermelisin?",
        "Kullanicinin sorusu belirsizse ne yaparsin?",
        "Cevap verirken nasil bir uslup kullanmalisin?",
        "Yerel bir sohbet modeli olarak amacin nedir?",
    ])
    answers = {
        "Bilmedigin konuda nasil cevap vermelisin?":
            "Emin degilsem bunu acikca soylerim, uydurmam ve kontrol edilebilir bir yol oneririm.",
        "Kullanicinin sorusu belirsizse ne yaparsin?":
            "Once anladigim kismi netlestirir, gerekiyorsa kisa bir takip sorusu sorarim.",
        "Cevap verirken nasil bir uslup kullanmalisin?":
            "Kisa, net, sakin ve kullanicinin amacina odakli bir uslup kullanirim.",
        "Yerel bir sohbet modeli olarak amacin nedir?":
            "Amacim kullanicinin isini yerelde, anlasilir ve guvenilir cevaplarla kolaylastirmaktir.",
    }
    return make_row(q, answers[q])


def write_stream(target_tokens: int, tok, system: str | None, out_path: str, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = 0
    rows = 0
    t0 = time.time()
    next_report = 5_000_000
    with open(out_path, "w", encoding="utf-8") as handle:
        while total < target_tokens:
            row = build_large_row(rows + 1, rng)
            tokens = row_token_count(tok, row, system)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += tokens
            rows += 1
            if total >= next_report:
                elapsed = max(0.001, time.time() - t0)
                print(
                    f"[chat-tokens] {total:,}/{target_tokens:,} token | "
                    f"{rows:,} sohbet | {total / elapsed:,.0f} tok/s",
                    flush=True,
                )
                next_report += 5_000_000
    return rows, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=100_000)
    ap.add_argument("--out", default="data/chat_100k/auto_100k.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--seed", type=int, default=SEED + 100_000)
    ap.add_argument("--system", default=None)
    ap.add_argument("--default-system", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="buyuk hedeflerde sohbetleri RAM'de tutmadan satir satir yaz")
    ap.add_argument("--stream-threshold", type=int, default=5_000_000,
                    help="hedef bu token sayisini gecerse otomatik stream modu kullan")
    args = ap.parse_args()

    if args.target_tokens < 1_000:
        raise SystemExit("--target-tokens en az 1000 olmali.")

    random.seed(args.seed)
    tok = load_tokenizer(args.tokenizer)
    system = DEFAULT_SYSTEM if args.default_system else args.system

    if args.stream or args.target_tokens >= args.stream_threshold:
        rows, total = write_stream(args.target_tokens, tok, system, args.out, args.seed)
        print(
            f"[chat-tokens] {rows:,} sohbet yazildi, "
            f"yaklasik {total:,} token -> {args.out}"
        )
        return

    measured = build_pool(args.target_tokens, tok, system)
    random.shuffle(measured)

    selected: list[dict] = []
    total = 0
    for row, tokens in measured:
        selected.append(row)
        total += tokens
        if total >= args.target_tokens:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[chat-tokens] {len(selected):,} sohbet yazildi, "
        f"yaklasik {total:,} token -> {args.out}"
    )


if __name__ == "__main__":
    main()
