"""
tsetlin/featurize.py
===================
Metni Tsetlin Makinesi'nin anlayacagi IKILI (0/1) ozellik vektorune cevirir.
Yaklasim: kelime-torbasi (bag-of-words). Her ozellik = "su kelime mesajda var mi?"

  * Diakritik KATLAMA: "adın" ve "adin" ayni ozellige gider (kullanici nasil
    yazarsa yazsin).
  * Sayilar tek bir <num> belirtecine indirgenir ("12+8" -> "<num> + <num>")
    -> "sayi iceriyor mu" matematik/kiyas icin guclu bir sinyal.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from typing import List

# Turkce -> ASCII (hem buyuk hem kucuk), sonra lower()
_FOLD = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "İ": "I", "ö": "o", "ş": "s", "ü": "u",
    "â": "a", "î": "i", "û": "u",
    "Ç": "C", "Ğ": "G", "Ö": "O", "Ş": "S", "Ü": "U", "Â": "A", "Î": "I", "Û": "U",
})
_TOKEN_RE = re.compile(r"<num>|[a-z]{2,}")


def tokenize(text: str) -> List[str]:
    text = text.translate(_FOLD).lower()
    text = re.sub(r"\d+", " <num> ", text)
    return _TOKEN_RE.findall(text)


def terms(text: str) -> List[str]:
    """Tek kelimeler (unigram) + ardisik ikililer (bigram). Bigram'lar 'etik nedir'
    gibi anlamsal ayrimi yakalar (generik 'nedir'den ayrisir)."""
    toks = tokenize(text)
    out = list(toks)
    out += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return out


class Vocabulary:
    def __init__(self, words: List[str]):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}

    @property
    def size(self) -> int:
        return len(self.words)

    @classmethod
    def build(cls, texts: List[str], max_features: int = 400, min_df: int = 2):
        df = Counter()
        for t in texts:
            for w in set(tokenize(t)):           # tek kelime (unigram) daha temiz sonuc verdi
                df[w] += 1
        common = [w for w, c in df.most_common() if c >= min_df][:max_features]
        return cls(common)

    def transform_one(self, text: str):
        import numpy as np
        v = np.zeros(self.size, dtype=np.int8)
        for w in set(tokenize(text)):
            j = self.index.get(w)
            if j is not None:
                v[j] = 1
        return v

    def transform(self, texts: List[str]):
        import numpy as np
        X = np.zeros((len(texts), self.size), dtype=np.int8)
        for i, t in enumerate(texts):
            for w in set(tokenize(t)):
                j = self.index.get(w)
                if j is not None:
                    X[i, j] = 1
        return X

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"words": self.words}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["words"])
