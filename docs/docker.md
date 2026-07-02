# Docker ile localllm

Bu kurulum image içine büyük veri veya checkpoint koymaz. Kod image içinde,
`./data`, `./checkpoints` ve `./state` host volume olarak container içine bağlanır.

## Sistem gereksinimi

Hızlı kontrol:

```bash
make doctor
make docker-doctor
```

Minimumlar:

| Alan | Minimum | Önerilen |
|---|---:|---:|
| Docker API | 20 GB boş disk | 40 GB+ |
| GPU Docker | NVIDIA driver + NVIDIA Container Toolkit | 60 GB+ boş disk |
| RAM | 8 GB | 16-32 GB |
| Adapter eğitimi | 4 GB VRAM | 6 GB+ VRAM |
| 200M veri | `data/chat_200m_plus_bin` | `32 GB RAM` ile daha rahat |

GPU profilleri için `docker info` içinde NVIDIA runtime görünmelidir. Görünmüyorsa
`api` CPU'da çalışır, ama `api-gpu`/`worker-gpu` için Container Toolkit gerekir.

## CPU/API

```bash
export YERELLM_API_TOKEN="uzun-rastgele-token"
docker compose build api
docker compose up -d api
docker compose logs -f api
```

Arayüz:

```text
http://127.0.0.1:8000/
```

## GPU/API

NVIDIA Container Toolkit kurulu olmalı. Aynı image CUDA PyTorch wheel ile build edilir.

```bash
docker compose --profile gpu up -d --build api-gpu
docker compose logs -f api-gpu
```

## Worker container

API container'a bağlı CPU/auto worker:

```bash
docker compose --profile worker up -d worker
```

GPU worker:

```bash
docker compose --profile gpu --profile gpu-worker up -d worker-gpu
```

## Eğitim komutları

Hazır 200M sohbet verisiyle kısa adapter eğitimi:

```bash
docker compose --profile train run --rm trainer train \
  --resume checkpoints/ckpt.pt \
  --adapter \
  --adapter-out checkpoints \
  --data data/chat_200m_plus_bin \
  --max-steps 250 \
  --batch-size 4 \
  --grad-accum 4 \
  --reset-best \
  --device cuda
```

Mevcut adapter üzerine devam etmek istersen komuta ayrıca
`--adapter-resume checkpoints/adapter.pt` ekle.

Veri hazırlama:

```bash
docker compose run --rm api prepare-data \
  --mode chat \
  --input data/chat_200m \
  --out data/chat_200m_bin \
  --no-chat-sync
```

## Ortam dosyası

İstersen `.env.docker.example` dosyasını `.env` olarak kopyalayıp port, checkpoint
worker adlarını ve `YERELLM_API_TOKEN` değerini oradan yönetebilirsin.
