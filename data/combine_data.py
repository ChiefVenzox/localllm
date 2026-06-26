"""
data/combine_data.py
====================
Tum sohbet veri kaynaklarini DENGELI sekilde tek dosyada birlestirir. Her kaynak
icin bir HEDEF SAYI verilir: kaynakta daha az varsa tekrar (oversample), fazlaysa
rastgele alt-kume (subsample). Boylece matematik gibi cok ornekli kaynaklar
digerlerini ezmez; kimlik/kod gibi kritik ama az kaynaklar guclendirilir.

Cikti: data/chat_all/all.jsonl

Kullanim:
    python -m data.combine_data
"""
from __future__ import annotations
import json
import os
import random

random.seed(13)

# (dosya, hedef konusma sayisi)
SOURCES = [
    ("data/chat_bilge/bilge.jsonl", 460),        # temel persona/sohbet (hepsi)
    ("data/chat_identity/identity.jsonl", 284),  # kimlik (142 varyant x2 -> guclu)
    ("data/chat_math/math.jsonl", 1003),         # matematik (HEPSI -- her islem egitimde olsun)
    ("data/chat_knowledge/knowledge.jsonl", 409),  # genel bilgi (hepsi)
    ("data/chat_code/code.jsonl", 316),          # kod (158 x2 -> guclu)
    ("data/chat_round3/round3.jsonl", 367),      # cok-turlu + kiyas/mantik + Turkiye + tavsiye
    ("data/chat_compare/compare.jsonl", 700),    # sayi KIYAS (matematikle karistirmasin)
    ("data/chat_felsefe/felsefe.jsonl", 340),    # felsefe/etik/soyut + Bilge oz-dusunce
]


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        print(f"[combine] UYARI: yok -> {path}")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    combined = []
    for path, target in SOURCES:
        items = read_jsonl(path)
        if not items:
            continue
        if target <= len(items):
            chosen = random.sample(items, target)            # subsample
        else:
            chosen = items + random.choices(items, k=target - len(items))  # oversample
        combined.extend(chosen)
        print(f"[combine] {os.path.basename(path):20s} {len(items):4d} mevcut -> {target:4d} alindi")

    random.shuffle(combined)
    os.makedirs("data/chat_all", exist_ok=True)
    path = "data/chat_all/all.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in combined:
            f.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")
    print(f"[combine] TOPLAM {len(combined)} konusma -> {path}")


if __name__ == "__main__":
    main()
