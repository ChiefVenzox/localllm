"""
online_learn.py
===============
"Konustukca kendini egit": calisan (canli) modeli, sohbet sirasinda gelen
ornek konusmalarla ANINDA, birkac gradyan adimiyla ince-ayar eder. Boylece
kullanici bir cevabi duzeltip "ogret" dediginde model o an ogrenir.

Tasarim:
  * Kayip yalnizca ASISTAN token'larina uygulanir (maskeli SFT) -> model
    "verilen soruya su cevabi uret" iliskisini ogrenir, soruyu ezberlemez.
  * Dusuk LR + az adim -> tek ornek yuzunden her seyi unutmayi (catastrophic
    forgetting) sinirlar.
  * REHEARSAL: daha once ogretilen birkac ornek de tekrar oynatilir; boylece
    eski duzeltmeler korunur.
  * Ogretilen her ornek data/chat_live/learned.jsonl'e yazilir (kalici);
    istenirse model checkpoint'i de diske kaydedilir (save()).

NOT: Bu GERCEK bir surekli-ogrenme (continual learning) mekanizmasidir ama kucuk
modelde sinirlidir: cok sik/celiskili ogretim modeli bozabilir. Kalici, saglam
ogrenme icin ogretilen ornekleri toplayip ara ara train.py ile yeniden egitmek
en saglamidir (learned.jsonl bunun icin saklanir).
"""
from __future__ import annotations
import json
import os
import random
import threading
from collections import deque
from typing import Dict, List, Optional

import torch

from chat_template import DEFAULT_SYSTEM


_ROLE_ATTR = {"system": "system_id", "user": "user_id", "assistant": "assistant_id"}


def encode_supervised(tok, messages: List[Dict[str, str]],
                      system: Optional[str] = DEFAULT_SYSTEM):
    """Konusmayi token id'lerine cevirir + her token icin 'asistan ciktisi mi'
    maskesi dondurur. mask[i]=True ise ids[i] bir asistan-cevap token'idir
    (icerik veya asistan turunun <|end|>'i)."""
    ids: List[int] = []
    mask: List[bool] = []

    def add(tid: int, trainable: bool):
        ids.append(tid)
        mask.append(trainable)

    has_system = any(m["role"] == "system" for m in messages)
    if system and not has_system:
        add(tok.system_id, False)
        for t in tok.encode(system):
            add(t, False)
        add(tok.end_id, False)

    for m in messages:
        role = m["role"]
        if role not in _ROLE_ATTR:
            continue
        add(getattr(tok, _ROLE_ATTR[role]), False)   # rol isareti egitilmez
        trainable = (role == "assistant")
        for t in tok.encode(m["content"]):
            add(t, trainable)
        add(tok.end_id, trainable)                    # asistan ise <|end|>'i de ogret
    return ids, mask


class OnlineLearner:
    def __init__(self, model, tok, device: str, *,
                 lr: float = 8e-5, weight_decay: float = 0.0,
                 grad_clip: float = 1.0,
                 log_path: str = "data/chat_live/learned.jsonl",
                 base_data: str = "data/chat_bilge/bilge.jsonl",
                 rehearsal: int = 4, buffer_size: int = 400):
        self.model = model
        self.tok = tok
        self.device = device
        self.grad_clip = grad_clip
        self.log_path = log_path
        self.rehearsal = rehearsal
        self.system = DEFAULT_SYSTEM
        self._lock = threading.Lock()           # sunucuda istekler arasi guvenlik
        self.buffer: deque = deque(maxlen=buffer_size)
        self.base_pool: list = []               # unutmayi onlemek icin TEMEL veri

        # online ince-ayar fp32'de yapilir (tek ornekte fp16'dan daha stabil)
        self.model.gradient_checkpointing = False
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay,
        )
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._load_base(base_data)
        self._preload_buffer()

    # -- dahili --
    def _load_base(self, path: str):
        """Temel egitim konusmalarini rehearsal havuzuna al (unutmayi onler)."""
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if obj.get("messages"):
                            self.base_pool.append(obj["messages"])
        except Exception:
            pass

    def _rehearse(self):
        """Temel veri + onceki ogretimlerden birkac ornegi tekrar oynat."""
        pool = self.base_pool + list(self.buffer)
        if not pool:
            return
        for ex in random.sample(pool, min(self.rehearsal, len(pool))):
            self._one_step(ex)

    def _preload_buffer(self):
        """Onceki oturumda ogretilenleri tampona al (rehearsal icin)."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if obj.get("messages"):
                            self.buffer.append(obj["messages"])
        except Exception:
            pass

    def _one_step(self, messages) -> Optional[float]:
        ids, mask = encode_supervised(self.tok, messages, self.system)
        if len(ids) < 2 or not any(mask[1:]):
            return None
        n = min(len(ids), self.model.block_size + 1)
        ids, mask = ids[-n:], mask[-n:]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor(
            [[ids[i + 1] if mask[i + 1] else -1 for i in range(len(ids) - 1)]],
            dtype=torch.long, device=self.device,
        )
        _, loss, _ = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    def _log(self, messages):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # -- ana API --
    def teach(self, messages: List[Dict[str, str]], steps: int = 6) -> Optional[Dict]:
        """Bir konusmayi (en az bir asistan turu olmali) modele ogretir."""
        with self._lock:
            was_training = self.model.training
            self.model.train()
            try:
                first = self._one_step(messages)
                if first is None:
                    return None
                last = first
                self._rehearse()                 # ilk adimdan sonra da sabitle
                for _ in range(max(0, steps - 1)):
                    last = self._one_step(messages)
                    # her yeni-ornek adimindan sonra TEMEL veri + tampondan tekrar
                    self._rehearse()
            finally:
                if not was_training:
                    self.model.eval()
            self.buffer.append(messages)
            self._log(messages)
            return {"loss_before": round(first, 4),
                    "loss_after": round(last, 4),
                    "steps": steps,
                    "ogrenilen_toplam": len(self.buffer)}

    def _lm_step(self, ids: List[int]) -> Optional[float]:
        """Ham metin uzerinde DENETIMSIZ next-token adimi (maskesiz)."""
        n = min(len(ids), self.model.block_size + 1)
        if n < 8:
            return None
        ids = ids[:n]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
        _, loss, _ = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    def study_text(self, text: str, passes: int = 1) -> Optional[Dict]:
        """Bir metin parcasini 'okuyarak' ogrenir (next-token), arada chat
        ornekleriyle rehearsal yaparak Bilge'nin sohbet davranisini korur."""
        ids = self.tok.encode(text)
        if len(ids) < 8:
            return None
        B = self.model.block_size
        windows = [ids[i:i + B + 1] for i in range(0, max(1, len(ids) - 1), B)]
        windows = [w for w in windows if len(w) >= 8]
        if not windows:
            return None
        with self._lock:
            was = self.model.training
            self.model.train()
            first = last = None
            try:
                for _ in range(max(1, passes)):
                    for w in windows:
                        loss = self._lm_step(w)
                        if loss is None:
                            continue
                        first = loss if first is None else first
                        last = loss
                        self._rehearse()
            finally:
                if not was:
                    self.model.eval()
        if first is None:
            return None
        return {"loss_before": round(first, 4), "loss_after": round(last, 4),
                "windows": len(windows)}

    def teach_pair(self, user: str, assistant: str, system: Optional[str] = None,
                   steps: int = 6) -> Optional[Dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
        return self.teach(msgs, steps=steps)

    def save(self, path: str):
        """Guncel (ogrenmis) modeli checkpoint olarak diske yazar."""
        from dataclasses import asdict
        with self._lock:
            raw = getattr(self.model, "_orig_mod", self.model)
            torch.save({
                "model": raw.state_dict(),
                "config": asdict(self.model.cfg),
                "step": -1,
                "best_val": float("inf"),
                "note": "online_learn ile guncellendi",
            }, path)
        return path
