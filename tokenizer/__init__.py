"""
tokenizer paketi
================
KENDI egittigimiz BPE tokenizer'i yukleyip kullanmak icin sarmalayici (wrapper).
`tokenizers` kutuphanesini kullaniriz (bu bir TOKENIZER araci, LLM degil) ama
kelime dagarcigini (vocab) tamamen kendi verimiz uzerinde egitiriz.
"""
from __future__ import annotations
import os
from typing import List

from tokenizers import Tokenizer as _HFTokenizer

from config import SPECIAL_TOKENS


class Tokenizer:
    """tokenizer.json dosyasini yukler; ozel token id'lerini hazirda tutar."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tokenizer bulunamadi: {path}\n"
                f"Once: python -m tokenizer.train_tokenizer ile egit."
            )
        self.tk = _HFTokenizer.from_file(path)
        self.path = path
        self.vocab_size = self.tk.get_vocab_size()

        # ozel token id'leri
        self.eot_id = self.token_to_id("<|endoftext|>")
        self.user_id = self.token_to_id("<|user|>")
        self.assistant_id = self.token_to_id("<|assistant|>")
        self.system_id = self.token_to_id("<|system|>")
        self.end_id = self.token_to_id("<|end|>")
        # uretimi durduracak token id'leri
        self.stop_ids = {self.eot_id, self.end_id}

    def token_to_id(self, tok: str) -> int:
        i = self.tk.token_to_id(tok)
        if i is None:
            raise ValueError(f"Ozel token tokenizer'da yok: {tok}")
        return i

    def encode(self, text: str) -> List[int]:
        """Duz metni token id'lerine cevirir (ozel token islemez)."""
        return self.tk.encode(text).ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special)


def load_tokenizer(path: str = "tokenizer/tokenizer.json") -> Tokenizer:
    return Tokenizer(path)
