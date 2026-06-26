"""
tsetlin/intent_gate.py
=====================
Egitilmis Tsetlin niyet/kapsam hakemini yukler ve bir mesaji siniflandirir.
Bilge'nin sunucusu bunu KAPI olarak kullanir: mesaj kapsam-disi ya da
guvensizse "bilmiyorum, ogret" der; degilse Bilge cevaplar.

Cikti (classify): {label, score, in_scope, rule, all_scores}
  - label     : tahmin edilen niyet (matematik/kimlik/.../kapsam_disi)
  - in_scope  : Bilge bu konuyu biliyor mu (kapsam_disi degil + yeterli marj)
  - rule      : kararin OKUNABILIR gerekcesi ("EGER 'adin' VE ... -> kimlik")
"""
from __future__ import annotations
import json
import os

import numpy as np

from tsetlin.featurize import Vocabulary
from tsetlin.machine import MultiClassTsetlinMachine


class IntentGate:
    def __init__(self, model_dir="checkpoints/intent", margin=2):
        self.tm = MultiClassTsetlinMachine.load(os.path.join(model_dir, "tm.npz"))
        self.vocab = Vocabulary.load(os.path.join(model_dir, "vocab.json"))
        with open(os.path.join(model_dir, "labels.json"), encoding="utf-8") as f:
            self.labels = json.load(f)
        self.margin = margin
        # kurallari bir kez hesapla (her sorguda degil)
        self._rules = self.tm.get_rules(self.vocab.words, top_clauses=1, max_lits=4)

    def classify(self, text: str) -> dict:
        x = self.vocab.transform_one(text)
        cid, scores = self.tm.decision(x)
        top = int(scores[cid])
        second = int(np.partition(scores, -2)[-2]) if len(scores) > 1 else -10**9
        label = self.labels[cid]
        in_scope = (label != "kapsam_disi") and (top > 0) and (top - second >= self.margin)
        rule = self._rules.get(cid, [None])[0] if self._rules.get(cid) else None
        return {
            "label": label,
            "score": top,
            "in_scope": bool(in_scope),
            "rule": rule,
            "all_scores": {self.labels[i]: int(scores[i]) for i in range(len(self.labels))},
        }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    gate = IntentGate()
    for q in ["adin ne", "12 + 8 kac eder", "python fonksiyon yaz", "etik nedir",
              "5 mi 8 mi buyuk", "kuantum fizigi nedir", "asdf zırt pırt"]:
        r = gate.classify(q)
        print(f"'{q}' -> {r['label']} (skor {r['score']}, kapsamda={r['in_scope']})")
        if r["rule"]:
            print(f"      gerekce: EGER {r['rule']}")
