"""
model/gpt.py
============
Sifirdan yazilmis, modern bir decoder-only Transformer (GPT tarzi).
Hicbir hazir model/agirlik kullanilmaz. Mimari bilesenleri:

  * RMSNorm            (LayerNorm yerine, daha hizli/stabil)
  * RoPE               (Rotary Position Embeddings - ogrenilen konum yok)
  * SwiGLU MLP         (GELU/ReLU yerine)
  * GQA                (Grouped-Query Attention - KV bellek tasarrufu)
  * SDPA               (PyTorch flash/efficient attention cekirdegi)
  * KV-cache           (hizli uretim/sohbet icin)
  * Gradient checkpointing (VRAM tasarrufu)
  * Weight tying       (giris embedding == cikis lm_head)

Tasarim GTX 1660 Ti (6 GB) gibi dusuk-VRAM kartlar gozetilerek yapildi.
"""
from __future__ import annotations
import math
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import GPTConfig


# ---------------------------------------------------------------------------
#  RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32'de hesapla, sonra geri dondur (stabilite icin)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
#  RoPE yardimcilari
# ---------------------------------------------------------------------------
def build_rope_cache(head_dim: int, max_seq: int, theta: float):
    """cos/sin tablolarini onceden hesaplar. Sekil: (max_seq, head_dim)."""
    assert head_dim % 2 == 0, "head_dim cift olmali (RoPE icin)"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)            # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (max_seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:   (B, n_head, T, head_dim)
    # cos: (T, head_dim) -> broadcast (1,1,T,head_dim)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, D) -> (B, n_kv*n_rep, T, D)  (GQA icin K/V kopyalama)."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, T, D)
        .reshape(B, H * n_rep, T, D)
    )


# ---------------------------------------------------------------------------
#  Dikkat (Attention)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head, n_kv_head'in tam kati olmali"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        kv_dim = self.n_kv_head * self.head_dim
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, attn_mask, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE'yi YENI tokenlara uygula (gecmis K zaten kendi konumunda donmustu)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None and past_kv[0] is not None:
            pk, pv = past_kv
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        new_past = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        dp = self.dropout if self.training else 0.0
        if attn_mask is None:
            # hizli yol: tam nedensel (training / bostan prefill)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        else:
            # acik maske (KV-cache ile prefill/decode)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.wo(y))
        return y, new_past


# ---------------------------------------------------------------------------
#  SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg._ffn_hidden()
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class NeuralAdapter(nn.Module):
    """Final hidden state icin sifir-etkili residual bottleneck adapter."""
    def __init__(self, n_embd: int, bottleneck: int = 64, dropout: float = 0.0,
                 scale: float = 1.0, rms_eps: float = 1e-5):
        super().__init__()
        bottleneck = max(1, int(bottleneck))
        self.norm = RMSNorm(n_embd, rms_eps)
        self.down = nn.Linear(n_embd, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = float(scale)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        delta = self.up(F.silu(self.down(self.norm(x))))
        return x + self.drop(delta) * self.scale


# ---------------------------------------------------------------------------
#  Transformer blogu (pre-norm)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask, past_kv=None, use_cache=False):
        h, new_past = self.attn(self.attn_norm(x), cos, sin, attn_mask, past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_past


# ---------------------------------------------------------------------------
#  GPT
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layer = cfg.n_layer
        self.block_size = cfg.block_size
        head_dim = cfg.n_embd // cfg.n_head

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, cfg.rms_eps)
        self.adapter = NeuralAdapter(
            cfg.n_embd,
            cfg.adapter_dim,
            cfg.adapter_dropout,
            cfg.adapter_scale,
            cfg.rms_eps,
        ) if cfg.adapter_enabled else None
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE tablolari (egitilmez tampon)
        cos, sin = build_rope_cache(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)
        if self.adapter is not None:
            nn.init.zeros_(self.adapter.up.weight)
        # residual projeksiyonlarini derinlige gore olcekle (GPT-2 init)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def attach_adapter(self, bottleneck: int | None = None,
                       dropout: float | None = None,
                       scale: float | None = None):
        """Egitilmis base modele sonradan kucuk adapter tak."""
        if self.adapter is not None:
            return self.adapter
        bottleneck = int(bottleneck or self.cfg.adapter_dim or 64)
        dropout = self.cfg.adapter_dropout if dropout is None else float(dropout)
        scale = self.cfg.adapter_scale if scale is None else float(scale)
        ref = next(self.parameters())
        self.adapter = NeuralAdapter(
            self.cfg.n_embd,
            bottleneck,
            dropout,
            scale,
            self.cfg.rms_eps,
        ).to(device=ref.device, dtype=ref.dtype)
        self.cfg.adapter_enabled = True
        self.cfg.adapter_dim = bottleneck
        self.cfg.adapter_dropout = dropout
        self.cfg.adapter_scale = scale
        return self.adapter

    def freeze_base_model(self, train_adapter: bool = True):
        """Base LLM'i dondur; sadece adapter trainable kalsin."""
        for p in self.parameters():
            p.requires_grad = False
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad = train_adapter

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kvs=None):
        B, T = idx.shape
        use_cache = past_kvs is not None
        past_len = 0
        if use_cache and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)

        assert past_len + T <= self.block_size, (
            f"baglam tasti: {past_len + T} > block_size={self.block_size}"
        )

        cos = self.rope_cos[past_len:past_len + T]
        sin = self.rope_sin[past_len:past_len + T]

        x = self.drop(self.tok_emb(idx))

        # maske: gecmis varsa acik maske kur, yoksa is_causal hizli yolu (None)
        if past_len > 0:
            q_pos = torch.arange(past_len, past_len + T, device=idx.device)
            k_pos = torch.arange(0, past_len + T, device=idx.device)
            attn_mask = (q_pos[:, None] >= k_pos[None, :])  # (T, past+T) bool
        else:
            attn_mask = None

        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if use_cache else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, npkv = checkpoint(
                    block, x, cos, sin, attn_mask, pkv, use_cache,
                    use_reentrant=False,
                )
            else:
                x, npkv = block(x, cos, sin, attn_mask, pkv, use_cache)
            if use_cache:
                new_past.append(npkv)

        x = self.norm_f(x)
        if self.adapter is not None:
            x = self.adapter(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss, None

        # uretimde sadece son konumun logit'i yeter
        logits = self.lm_head(x[:, [-1], :])
        return logits, None, new_past

    # ---- optimizer kurulumu (weight decay gruplari) ----
    def configure_optimizers(self, cfg: GPTConfig, device_type: str):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:          # matrisler -> weight decay
                decay.append(p)
            else:                     # norm/bias -> decay yok
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if cfg.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(groups, lr=cfg.learning_rate,
                                          betas=(cfg.beta1, cfg.beta2))
                print("[optim] bitsandbytes AdamW8bit kullaniliyor")
                return opt
            except Exception as e:
                print(f"[optim] 8-bit optimizer yuklenemedi ({e}); torch AdamW'ye geciliyor")
        # Windows + bazi PyTorch CUDA surumlerinde fused AdamW, GradScaler ile
        # "grad_scale/found_inf" assertion'ina dusebiliyor. Linux CUDA'da hizli
        # yolu koru, Windows'ta klasik AdamW daha guvenilir.
        use_fused = device_type == "cuda" and sys.platform != "win32"
        opt = torch.optim.AdamW(groups, lr=cfg.learning_rate,
                                betas=(cfg.beta1, cfg.beta2), fused=use_fused)
        return opt

    # -----------------------------------------------------------------
    #  Uretim (KV-cache'li)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 top_p=None, eos_token_id=None, use_cache=True):
        for tok in self.generate_stream(idx, max_new_tokens, temperature,
                                         top_k, top_p, eos_token_id, use_cache):
            idx = torch.cat((idx, tok), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                        top_p=None, eos_token_id=None, use_cache=True):
        """Her adimda uretilen token id'sini (sekil (B,1)) yield eder."""
        self.eval()
        device = idx.device
        # prefill: baglam block_size'i asarsa sondan kirp
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
        past_kvs = [None] * self.n_layer if use_cache else None
        cur_len = idx.size(1)

        for step in range(max_new_tokens):
            if use_cache:
                idx_cond = idx if step == 0 else next_id
            else:
                idx_cond = idx[:, -self.block_size:]

            logits, _, past_kvs = self(idx_cond, past_kvs=past_kvs)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                logits = self._filter_logits(logits, top_k, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            if not use_cache:
                idx = torch.cat((idx, next_id), dim=1)
            cur_len += 1
            yield next_id

            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
            if cur_len >= self.block_size:
                break  # RoPE tablosu/baglam siniri

    @staticmethod
    def _filter_logits(logits, top_k, top_p):
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        return logits
