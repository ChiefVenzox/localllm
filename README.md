# yerelLLM — Sıfırdan Yerel Dil Modeli (GTX 1660 Ti / 6 GB)

Bu proje, **hazır/açık-kaynak bir model kullanmadan**, tamamen kendi kodumuzla
yazılmış bir decoder-only Transformer'ı (GPT tarzı) **sıfırdan eğitir** ve bir
**web sohbet arayüzünde** çalıştırır. Türkçe + İngilizce, sohbet + kod amaçlı.

> **Donanım gerçeği:** 6 GB VRAM'de *kullanılabilir* bir 1B model **sıfırdan
> eğitilemez** (bellek + veri + zaman yetmez). Bu yüzden kod **ölçeklenebilir**
> yazıldı: şimdi 1660 Ti'da gerçekten eğitilebilen **~110M** (`small-100m`)
> modeli eğitiyoruz; ileride büyük GPU/bulut bulursan **aynı kodla** `xl-1b`
> presetine geçebilirsin. Küçük modelden ChatGPT kalitesi beklenmemeli — bu
> kendi modelin, öğretici ve tamamen senin kontrolünde.

---

## "Bilge" — sohbet asistanı, kendi kendine öğrenme ve web

Bu proje üzerine **Bilge** adlı bir Türkçe sohbet asistanı kuruldu: yerel,
sıfırdan eğitilmiş, ~57M parametreli (`bilge-60m` preset, 1660 Ti'da eğitilebilir)
bir model. Bilge **ezber-tabanlıdır** — verdiğin örnekleri öğrenir; bilmediğini
uyduramaz. Bu yüzden iki güçlü yeteneği vardır:

**1) Konuştukça öğrenir (online learning).** Web arayüzünde her cevabın altında
**👍 Bunu öğret** ve **✏️ Düzelt & öğret** düğmeleri var. Yanlış bir cevabı
düzeltip öğretirsin, Bilge o an birkaç gradyan adımıyla öğrenir
(`online_learn.py`). Temel veriden *rehearsal* yapıldığı için yeni bir şey
öğretmek eski bilgiyi bozmaz. "Öğrenileni kaydet" ile kalıcı yaparsın.

**2) Web'den kendi kendine eğitilir.** `web_sources.json`'a izin verdiğin siteleri
yazarsın; Bilge bunları gezip (robots.txt'e saygılı, hız/boyut sınırlı, sadece
stdlib) ana metni çıkarır, otomatik Soru-Cevap üretir ve yerelde kendini eğitir
(`web_learn.py`, `/api/web/study`, arayüzde **🌐 Web'den öğren**). Otomatik/zamanlı
da çalışabilir (`enabled: true`, `interval_minutes`).

### Eğitim verisi üreteçleri (`data/`)
Bilge'nin bilgisi, kod ile üretilen veri setlerinden gelir:

| Üreteç | İçerik |
|---|---|
| `make_math_data.py` | Aritmetik (toplama/çıkarma/çarpma + problem), "Düşünelim:/Sonuç:" |
| `make_identity_data.py` | Kimlik soruları, çok-varyantlı (şapkalı/şapkasız/büyük-küçük) |
| `make_sample_data.py` | Küçük örnek korpus (duman testi) |
| *(workflow)* | Genel bilgi, kod, çok-turlu diyalog, kıyas/mantık → `build_bilge_data.py` ile jsonl |
| `combine_data.py` | Tüm kaynakları **dengeli** birleştirir (az olanı çoğaltır, çok olanı seyreltir) |

### Bilge'yi kullan
```bash
# Terminalden sohbet:
python generate.py --chat --ckpt checkpoints/ckpt.pt
# Hızlı toplu deneme:
python chat_demo.py --default-system
# Web arayüzü (öğret + web öğrenme dahil):
python -m server.app --ckpt checkpoints/ckpt.pt   # -> http://127.0.0.1:8000
```

### Dürüst sınırlar
- ~57M ezber modeli: öğretilen aralıkta matematik (örn. 12+8) ve öğretilen
  konuları iyi bilir; **eğitim dışı** sorularda uydurabilir veya benzer iki
  cevabı karıştırabilir.
- Web'den **ham metin "okuma"** küçük modeli bozar → varsayılan kapalı; **Soru-Cevap**
  öğrenme güvenlidir ve cevaplar kısa tutulur. Kalite tavanı model boyutudur;
  daha büyük modelde (aynı kod) belirgin artar.

---

## Mimari (hepsi bizim kodumuz)

| Bileşen | Seçim | Neden |
|---|---|---|
| Konum kodlama | **RoPE** | Öğrenilen konum yok, uzunluğa esnek |
| Normalizasyon | **RMSNorm** | LayerNorm'dan hızlı/stabil |
| MLP | **SwiGLU** | Modern, daha iyi kalite |
| Dikkat | **GQA + SDPA** | VRAM tasarrufu + flash/efficient çekirdek |
| Üretim | **KV-cache** | Hızlı sohbet |
| Bellek | **grad checkpointing + fp16 + grad accum** | 6 GB'a sığar |
| Tokenizer | **kendi Byte-level BPE'miz** | TR/EN/kod + `<unk>` yok |

`model/gpt.py`, `config.py`, `train.py`, `tokenizer/`, `data/`, `generate.py`,
`server/` — hepsi tek bir `GPTConfig` sözleşmesine bağlı.

---

## Kurulum

GTX 1660 Ti (Turing, SM 7.5) için CUDA derlemeli PyTorch. Ayrı CUDA Toolkit
**gerekmez** — wheel CUDA runtime'ı içinde taşır; sadece güncel NVIDIA sürücüsü yeter.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> `cu126` (CUDA 12.6) 1660 Ti'da doğrulandı. Sürücün CUDA 13.x'e kadar
> destekliyorsa `cu128`/`cu124` de çalışır (sürücüler geriye dönük uyumlu).
> Aynı wheel hem GPU (`--device cuda`) hem CPU (`--device cpu`) çalıştırır;
> ayrı "CPU torch" gerekmez.

GPU'yu doğrula:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Bir preset'in 6 GB'a sığıp sığmadığını ölç (gerçek GPU'da):
```bash
python scripts/vram_probe.py --preset small-100m
# 1660 Ti'da olculen: zirve ~4.3 GB allocated / ~5.4 GB reserved -> SIGIYOR
```

---

## Hızlı Başlangıç (uçtan uca duman testi)

Aşağıdaki adımlar küçük örnek veriyle tüm boru hattını çalıştırır:

```bash
# 1) Örnek korpus üret (TR + EN + kod + sohbet)
python -m data.make_sample_data

# 2) Kendi tokenizer'ımızı eğit
python -m tokenizer.train_tokenizer --input data/raw --vocab-size 16000

# 3) Veriyi token bin'lerine çevir
python -m data.prepare_data --mode pretrain --input data/raw --out data/bin

# 4) Modeli eğit (hızlı deneme için tiny + az adım)
python train.py --preset tiny-30m --data data/bin --max-steps 500

# 5) Terminalden dene
python generate.py --chat --ckpt checkpoints/ckpt.pt

# 6) Web arayüzünü aç
python -m server.app --ckpt checkpoints/ckpt.pt
#   -> tarayıcı: http://127.0.0.1:8000
```

---

## Gerçek Eğitim (1660 Ti için önerilen akış)

1. **Veri topla.** `data/raw/` içine bol miktarda `.txt` koy (Türkçe Wikipedia
   dökümü, kitaplar, makaleler + İngilizce metin + kaynak kod). Ne kadar çok,
   o kadar iyi. Paragraflar boş satırla ayrılır.
2. **Tokenizer** (32k önerilir):
   ```bash
   python -m tokenizer.train_tokenizer --input data/raw --vocab-size 32000
   ```
3. **Veri hazırla:**
   ```bash
   python -m data.prepare_data --mode pretrain --input data/raw --out data/bin
   ```
4. **Ön-eğitim (pretrain):**
   ```bash
   python train.py --preset small-100m --data data/bin
   ```
   Yarıda kesersen devam et: `python train.py --resume checkpoints/ckpt_last.pt --data data/bin`
5. **Sohbet için ince ayar (SFT, opsiyonel ama önerilir):** Modelin "asistan"
   gibi davranması için sohbet verisiyle eğitim sürdür:
   ```bash
   python -m data.prepare_data --mode chat --input data/chat --out data/chat_bin
   python train.py --resume checkpoints/ckpt.pt --data data/chat_bin --max-steps 5000 --reset-best
   ```
   > `--reset-best` **şart**: farklı bir veri setine (sohbet) geçtiğin için
   > ön-eğitimden taşınan düşük `best_val` aksi halde yeni `ckpt.pt`'nin
   > kaydedilmesini engeller (en iyi sohbet checkpoint'i diske yazılmaz).
6. **Çalıştır:** `python -m server.app`

### Sohbet verisi formatı (`data/chat/*.jsonl`)
Her satır bir konuşma:
```json
{"messages":[{"role":"user","content":"Merhaba"},{"role":"assistant","content":"Selam! Nasıl yardımcı olabilirim?"}]}
```

---

## Presetler

```bash
python config.py   # tüm presetleri ve ~parametre sayılarını listeler
```

| preset | ~params | not |
|---|---|---|
| `tiny-30m` | ~17M | saniyeler/dakikalar; sadece test |
| `small-100m` | ~110M | **1660 Ti için önerilen** |
| `medium-350m` | ~350M | 6 GB'da zorlanır, çok yavaş |
| `xl-1b` | ~1.1B | **sadece büyük GPU/bulut** |

Ölçeği büyütmek için tek değişiklik: `--preset xl-1b`. Kod aynı kalır.

---

## VRAM neden 1B'yi kaldırmıyor?

Eğitimde her parametre kabaca **~10-16 byte** ister (ağırlık + gradyan + fp32
master + Adam momentleri). 1B × ~10 byte ≈ **10 GB** — daha aktivasyonları bile
saymadan 6 GB'ı aşar. Ayrıca işe yarar 1B model için ~20 milyar token gerekir;
bu 1660 Ti'da haftalar/aylar sürer. `~110M` ise 6 GB'a rahat sığar ve makul
sürede ilerler.

---

## Sorun Giderme

- **`CUDA out of memory`** → `--batch-size`'ı düşür (örn. 4 veya 2),
  `--grad-accum`'u artır; `config.py`'de `gradient_checkpointing=True` olduğundan
  emin ol; `block_size`'ı 512'ye indir.
- **`bitsandbytes` yüklenmiyor (Windows)** → sorun değil, `use_8bit_optimizer`
  otomatik olarak normal AdamW'ye düşer.
- **bf16 hatası** → 1660 Ti bf16 desteklemez; kod otomatik fp16 + GradScaler kullanır.
- **Çıktı anlamsız** → küçük modeller az veriyle böyledir; daha çok veri + daha
  çok adım + (sohbet için) SFT aşaması gerekir.
