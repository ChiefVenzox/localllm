"""
train_intent.py
===============
Bilge'nin "niyet/kapsam hakemi"ni egitir: sifirdan Tsetlin Makinesi, kategorili
veriden gelen mesajlari sinifkandirir (matematik/kimlik/kod/bilgi/felsefe/kiyas/
sohbet/kapsam_disi). Egitim GRADYANSIZ, CPU'da, ve OKUNABILIR kurallar uretir.

Kullanim:
    python -m data.build_intent_data   # once veriyi uret
    python train_intent.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

from tsetlin.featurize import Vocabulary
from tsetlin.machine import MultiClassTsetlinMachine

OUT_DIR = "checkpoints/intent"


def load(path):
    texts, labels = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            o = json.loads(line)
            texts.append(o["text"]); labels.append(o["label"])
    return texts, labels


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = np.random.default_rng(0)
    texts, labels = load("data/intent/intent.jsonl")
    label_names = sorted(set(labels))
    lab2id = {l: i for i, l in enumerate(label_names)}
    y = np.array([lab2id[l] for l in labels])

    # train/test bol
    idx = rng.permutation(len(texts))
    n_te = max(1, int(0.15 * len(texts)))
    te, tr = idx[:n_te], idx[n_te:]
    tr_texts = [texts[i] for i in tr]; te_texts = [texts[i] for i in te]
    ytr, yte = y[tr], y[te]

    # ozellikler
    vocab = Vocabulary.build(tr_texts, max_features=400, min_df=2)
    Xtr = vocab.transform(tr_texts)
    Xte = vocab.transform(te_texts)
    print(f"[intent] {len(tr_texts)} egitim / {len(te_texts)} test | "
          f"{vocab.size} ozellik | {len(label_names)} sinif: {label_names}")

    # egit (daha cok clause + daha yuksek T/s + daha cok epoch -> daha iyi ayrim)
    tm = MultiClassTsetlinMachine(n_classes=len(label_names), n_features=vocab.size,
                                  n_clauses=200, N=100, T=25, s=5.0, seed=0)
    tm.fit(Xtr, ytr, epochs=50, eval_every=10, Xte=Xte, yte=yte)

    # sinif bazli dogruluk
    pred = tm.predict(Xte)
    print("\n[sinif bazli test dogrulugu]")
    for i, name in enumerate(label_names):
        m = yte == i
        if m.any():
            print(f"  {name:12s}: {(pred[m] == i).mean():.2f}  ({m.sum()} ornek)")

    # okunabilir kurallar
    print("\n[ogrenilen mantik kurallari (ornek)]")
    rules = tm.get_rules(vocab.words, top_clauses=2, max_lits=4)
    for i, name in enumerate(label_names):
        for r in rules[i]:
            print(f"  EGER {r}  ->  {name}")

    # kaydet
    os.makedirs(OUT_DIR, exist_ok=True)
    tm.save(os.path.join(OUT_DIR, "tm"))
    vocab.save(os.path.join(OUT_DIR, "vocab.json"))
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=False)
    print(f"\n[intent] model kaydedildi -> {OUT_DIR}/ (tm.npz, vocab.json, labels.json)")

    # hizli deneme
    print("\n[canli deneme]")
    for q in ["12 ile 8 toplami kac eder?", "adin ne senin", "python'da liste nasil yapilir",
              "etik nedir", "5 mi 9 mu buyuk", "kuantum fizigi nedir", "zirt pirt flarp"]:
        cid, sc = tm.decision(vocab.transform_one(q))
        print(f"  '{q}' -> {label_names[cid]}  (skor {sc[cid]})")


if __name__ == "__main__":
    main()
