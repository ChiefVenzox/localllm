"""
config.py
=========
Modelin TUM mimari/egitim ayarlari burada. Tek yerden olceklenir:
parametre sayisini buyutmek/kucultmek icin sadece bir preset secersin.

Hicbir hazir/acik-kaynak model agirligi KULLANILMAZ. Mimari tamamen bizim;
asagidaki sayilarla model sifirdan insa edilir ve egitilir.

Kullanim:
    from config import get_config
    cfg = get_config("small-100m")   # 1660 Ti icin onerilen
    cfg = get_config("tiny-30m")     # hizli deneme
    cfg = get_config("xl-1b")        # SADECE buyuk GPU/bulut
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json


# ---- Sohbet sablonu icin ozel tokenlar (tokenizer ile paylasilir) ---------
SPECIAL_TOKENS = [
    "<|endoftext|>",   # metin/dizi sonu  (id 0 olacak sekilde egitilir)
    "<|user|>",        # kullanici turu basi
    "<|assistant|>",   # asistan turu basi
    "<|system|>",      # sistem mesaji basi
    "<|end|>",         # bir konusma turunun sonu
]


@dataclass
class GPTConfig:
    # ---- Mimari ----
    vocab_size: int = 32000        # tokenizer ile AYNI olmali (egitimden sonra otomatik guncellenir)
    block_size: int = 1024         # baglam (context) uzunlugu
    n_layer: int = 12              # transformer blok sayisi
    n_head: int = 12               # sorgu (query) basi sayisi
    n_kv_head: int = 12            # anahtar/deger basi sayisi (GQA icin < n_head yapilabilir)
    n_embd: int = 768              # gizli boyut (model genisligi)
    ffn_mult: float = 8 / 3        # SwiGLU ara katman carpani (~2.667)
    rope_theta: float = 10000.0    # RoPE taban frekansi
    dropout: float = 0.0           # kucuk verilerde 0.1 deneyebilirsin
    rms_eps: float = 1e-5
    tie_embeddings: bool = True    # giris embedding ile cikis (lm_head) agirligini paylas

    # ---- Egitim (6 GB dostu varsayilanlar) ----
    batch_size: int = 8            # GPU'ya ayni anda giren ornek sayisi
    grad_accum_steps: int = 16     # efektif batch = batch_size * grad_accum_steps = 128
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 20000
    lr_decay_steps: int = 20000    # genelde max_steps ile ayni
    gradient_checkpointing: bool = True   # VRAM <-> hiz takasi (6GB'da acik tut)
    use_8bit_optimizer: bool = False      # bitsandbytes varsa True yapilabilir

    # ---- Degerlendirme / kayit ----
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 10
    save_interval: int = 1000
    out_dir: str = "checkpoints"

    def n_params(self) -> int:
        """Yaklasik parametre sayisi (raporlama icin)."""
        v, d, L = self.vocab_size, self.n_embd, self.n_layer
        head_dim = d // self.n_head
        kv_dim = self.n_kv_head * head_dim
        # dikkat (attention): q + k + v + o projeksiyonlari
        attn = d * d + 2 * d * kv_dim + d * d
        # SwiGLU MLP: gate + up + down
        hidden = self._ffn_hidden()
        mlp = 3 * d * hidden
        per_layer = attn + mlp
        embed = v * d  # tie_embeddings ise tek sayilir
        total = embed + L * per_layer
        if not self.tie_embeddings:
            total += v * d
        return total

    def _ffn_hidden(self) -> int:
        # SwiGLU ara boyutu, 64'un katina yuvarlanir (verimlilik icin)
        h = int(self.ffn_mult * self.n_embd)
        return ((h + 63) // 64) * 64

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ---- Hazir presetler ------------------------------------------------------
_PRESETS = {
    # ~4M: ucdan uca DEMO/duman testi. CPU'da bile dakikalar, GPU'da saniyeler.
    # Amac: tokenizer -> veri -> egitim -> sohbet zincirini ve entropi (loss)
    # dususunu hizlica gostermek. Gercek kalite beklenmemeli.
    "nano-demo": dict(
        n_layer=4, n_head=4, n_kv_head=2, n_embd=256, block_size=128,
        vocab_size=4096, batch_size=32, grad_accum_steps=1,
        learning_rate=3e-4, warmup_steps=100, max_steps=1500,
        lr_decay_steps=1500, eval_interval=200, eval_iters=20,
        log_interval=20, save_interval=500, gradient_checkpointing=False,
    ),
    # ~57M: "Bilge" sohbet asistani icin. 1660 Ti'da (grad-checkpointing ile)
    # egitilebilir; kucuk veride zengin sohbeti ezberleyip paraphrase'lere bir
    # miktar genelleyecek kapasite. Kullanici tercihi: 50-60M.
    "bilge-60m": dict(
        n_layer=12, n_head=10, n_kv_head=2, n_embd=640, block_size=256,
        vocab_size=8000, batch_size=16, grad_accum_steps=2,
        learning_rate=5e-4, min_lr=5e-5, warmup_steps=40, max_steps=600,
        lr_decay_steps=600, eval_interval=99999, eval_iters=10,
        log_interval=20, save_interval=100, gradient_checkpointing=False,
        dropout=0.0,
    ),
    # ~107M: "Bilge" buyuk surum. 1660 Ti'da grad-checkpointing ile SIGAR ama
    # ~2 kat yavas egitilir (saatler). Daha cok kapasite -> daha cesitli veriyi
    # tutar ve biraz daha iyi genelleme. Kullanici tercihi: 100M.
    "bilge-100m": dict(
        n_layer=16, n_head=12, n_kv_head=4, n_embd=768, block_size=256,
        vocab_size=8000, batch_size=8, grad_accum_steps=4,
        learning_rate=3e-4, min_lr=3e-5, warmup_steps=100, max_steps=2500,
        lr_decay_steps=2500, eval_interval=99999, eval_iters=10,
        log_interval=20, save_interval=100, gradient_checkpointing=True,
        dropout=0.0,
    ),
    # ~17M: saniyeler/dakikalar icinde sonuc. Sadece boru hattini test icin.
    "tiny-30m": dict(
        n_layer=8, n_head=8, n_kv_head=8, n_embd=512, block_size=512,
        vocab_size=16000, batch_size=16, grad_accum_steps=8, max_steps=10000,
        lr_decay_steps=10000,
    ),
    # ~110M: 1660 Ti (6GB) icin ONERILEN denge.
    "small-100m": dict(
        n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024,
        vocab_size=32000, batch_size=8, grad_accum_steps=16, max_steps=40000,
        lr_decay_steps=40000,
    ),
    # ~350M: 6GB'da YALNIZCA gradient_checkpointing + 8-bit optimizer + kucuk
    # batch ile zorlanarak egitilebilir; cok yavas olur.
    "medium-350m": dict(
        n_layer=24, n_head=16, n_kv_head=4, n_embd=1024, block_size=1024,
        vocab_size=32000, batch_size=2, grad_accum_steps=64, max_steps=60000,
        lr_decay_steps=60000, use_8bit_optimizer=True,
    ),
    # ~1.1B: 1660 Ti'da EGITILEMEZ. Buyuk GPU / bulut icin hedef config.
    "xl-1b": dict(
        n_layer=22, n_head=16, n_kv_head=4, n_embd=2048, block_size=2048,
        vocab_size=32000, batch_size=4, grad_accum_steps=64, max_steps=200000,
        lr_decay_steps=200000, use_8bit_optimizer=True, learning_rate=2e-4,
    ),
}

DEFAULT_PRESET = "small-100m"


def get_config(preset: str = DEFAULT_PRESET, **overrides) -> GPTConfig:
    """Preset adindan GPTConfig uretir. overrides ile tek tek alan ezilebilir."""
    if preset not in _PRESETS:
        raise ValueError(
            f"Bilinmeyen preset: {preset!r}. Secenekler: {list(_PRESETS)}"
        )
    params = dict(_PRESETS[preset])
    params.update(overrides)
    return GPTConfig(**params)


def list_presets() -> None:
    print(f"{'preset':<14}{'~params':>12}{'n_layer':>9}{'n_embd':>8}{'ctx':>7}")
    print("-" * 50)
    for name in _PRESETS:
        c = get_config(name)
        p = c.n_params()
        human = f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.2f}B"
        print(f"{name:<14}{human:>12}{c.n_layer:>9}{c.n_embd:>8}{c.block_size:>7}")


if __name__ == "__main__":
    list_presets()
