"""
data/build_intent_data.py
========================
Bilge'nin kategorili sohbet verisinden NIYET sinifkandirma seti uretir. Her
kategori dosyasindaki kullanici sorusu -> o kategorinin etiketi. Ayrica Bilge'nin
HIC bilmedigi seyler icin bir "kapsam_disi" sinifi sentezler (gercek-ama-kapsam-
disi sorular + sacma metin). Cikti: data/intent/intent.jsonl  ({text, label})

Kullanim: python -m data.build_intent_data
"""
from __future__ import annotations
import json
import os
import random

random.seed(21)

SOURCES = {
    "matematik": "data/chat_math/math.jsonl",
    "kiyas":     "data/chat_compare/compare.jsonl",
    "kimlik":    "data/chat_identity/identity.jsonl",
    "kod":       "data/chat_code/code.jsonl",
    "bilgi":     "data/chat_knowledge/knowledge.jsonl",
    "felsefe":   "data/chat_felsefe/felsefe.jsonl",
    # not: round3 (cok-turlu/turkiye/empati...) cok karisik -> tutarli imza yok,
    # niyet siniflandirmasini bozuyordu; cikarildi.
}
PER_CLASS = 350     # sinif basina hedef (denge)

# Bilge'nin egitiminde OLMAYAN, gercek ama kapsam-disi sorular
OOS_REAL = [
    "Kuantum fiziği nedir?", "Görelilik teorisini açıkla", "DNA nasıl kopyalanır?",
    "Bitcoin fiyatı bugün ne kadar?", "Borsada hangi hisseyi almalıyım?",
    "Dolar kaç TL oldu?", "Bugün hava nasıl olacak?", "Yarın yağmur yağacak mı?",
    "En yakın restoran nerede?", "Şu an saat kaç?", "Bugün maç kaç kaç bitti?",
    "Galatasaray dün kazandı mı?", "Son dakika haberleri neler?",
    "Bana baklava tarifi ver", "Mantı nasıl yapılır?", "Lazanya tarifi nedir?",
    "Başım ağrıyor ne yapmalıyım?", "Bu ilacı kullanabilir miyim?",
    "Karın ağrım var, hangi hastaneye gitmeliyim?", "Vergi beyannamesi nasıl verilir?",
    "Ev kredisi faizi ne kadar?", "Pasaport nasıl çıkarılır?",
    "Java ile bir sunucu nasıl yazılır?", "Rust dilini öğret bana",
    "Photoshop'ta arka plan nasıl silinir?", "Excel'de pivot tablo nasıl yapılır?",
    "Napolyon hangi yıl öldü?", "Osmanlı kaç yıl sürdü?",
    "Everest dağı kaç metre?", "Amazon nehri nerede?", "Japonya'nın para birimi nedir?",
    "Albert Einstein kimdir?", "Mona Lisa'yı kim yaptı?", "Beethoven kimdir?",
    "Araba lastiği nasıl değişir?", "Bisiklet zinciri nasıl takılır?",
    "Köpeğim neden havlıyor?", "Kediler neden mırlar?", "Bebek neden ağlar?",
    "Uzaya nasıl gidilir?", "Roket nasıl çalışır?", "Nükleer enerji tehlikeli mi?",
    "Bana bir şiir yaz", "Bir aşk hikayesi anlat", "Rap şarkı sözü yaz",
    "Hangi diziyi izlemeliyim?", "En iyi film hangisi?",
]

# Sacma / anlamsiz metin (kapsam-disi'nin bir parcasi)
_GIBBERISH_WORDS = ["zırt", "pırt", "lorem", "flarp", "günok", "blez", "kweez",
                    "xanto", "mırfıl", "şponk", "drumel", "vızıl", "krint", "yablo",
                    "fnort", "glub", "trex", "munzo", "kaplo", "sertik"]


def read_first_users(path):
    out = []
    if not os.path.exists(path):
        print(f"[intent] UYARI yok: {path}")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("messages", [])
            if msgs and msgs[0].get("role") == "user":
                t = msgs[0]["content"].strip()
                if t:
                    out.append(t)
    return out


def gibberish():
    n = random.randint(2, 6)
    return " ".join(random.choice(_GIBBERISH_WORDS) for _ in range(n))


def build_oos(n):
    items = list(OOS_REAL)
    while len(items) < n:
        items.append(gibberish())
    random.shuffle(items)
    return items[:n]


def main():
    rows = []
    for label, path in SOURCES.items():
        texts = read_first_users(path)
        random.shuffle(texts)
        texts = texts[:PER_CLASS]
        rows += [{"text": t, "label": label} for t in texts]
        print(f"[intent] {label:10s} {len(texts):4d}")

    oos = build_oos(PER_CLASS)
    rows += [{"text": t, "label": "kapsam_disi"} for t in oos]
    print(f"[intent] {'kapsam_disi':10s} {len(oos):4d}")

    random.shuffle(rows)
    os.makedirs("data/intent", exist_ok=True)
    with open("data/intent/intent.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[intent] TOPLAM {len(rows)} ornek -> data/intent/intent.jsonl")


if __name__ == "__main__":
    main()
