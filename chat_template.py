"""
chat_template.py
================
Sohbeti tek bir token dizisine ceviren format. Ozel tokenlarla tur (turn)
sinirlari belirlenir. Hem egitim (chat verisi) hem de uretim (sunucu) ayni
formati kullanir; boylece model formati ogrenir.

Format (token bazli):
    <|system|>  ...sistem metni...   <|end|>
    <|user|>    ...kullanici metni... <|end|>
    <|assistant|> ...asistan metni... <|end|>
    ...
Uretimde dizinin sonuna <|assistant|> eklenir ve modelin devami uretmesi beklenir;
uretim <|end|> veya <|endoftext|> gorulunce durur.
"""
from __future__ import annotations
from typing import List, Dict

DEFAULT_SYSTEM = (
    "Sen Bilge'sin: kullanicinin kendi bilgisayarinda calisan, sifirdan egitilmis "
    "yardimsever bir Turkce yapay zeka asistani. Kibar ve samimisin; gerektiginde "
    "adim adim dusunursun. Sohbet eder, sorulari yanitlar ve kucuk kod ornekleri yazarsin."
)

_ROLE_ATTR = {
    "system": "system_id",
    "user": "user_id",
    "assistant": "assistant_id",
}


def encode_chat(tokenizer, messages: List[Dict[str, str]],
                add_generation_prompt: bool = True,
                system: str | None = None) -> List[int]:
    """
    messages: [{"role": "user"/"assistant"/"system", "content": "..."}]
    add_generation_prompt=True -> sona <|assistant|> ekler (uretim icin).
    """
    ids: List[int] = []

    # sistem mesaji (verilmisse veya messages icinde yoksa varsayilan)
    has_system = any(m["role"] == "system" for m in messages)
    if system is not None and not has_system:
        ids.append(tokenizer.system_id)
        ids.extend(tokenizer.encode(system))
        ids.append(tokenizer.end_id)

    for m in messages:
        role = m["role"]
        if role not in _ROLE_ATTR:
            raise ValueError(f"Bilinmeyen rol: {role}")
        ids.append(getattr(tokenizer, _ROLE_ATTR[role]))
        ids.extend(tokenizer.encode(m["content"]))
        ids.append(tokenizer.end_id)

    if add_generation_prompt:
        ids.append(tokenizer.assistant_id)
    return ids
