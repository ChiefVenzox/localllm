# LAN uzerinden coklu cihaz egitimi

Bu repo artik PyTorch Distributed Data Parallel (DDP) ile ayni egitim kosusuna
birden fazla makineyi katabilir. Ana makine checkpoint ve log yazar; diger
makineler ayni modelin gradyanlarini senkronize eder.

## Ne zaman kullanmali?

- Iki veya daha fazla masaustu/GPU ayni LAN icindeyse uygundur.
- Windows + macOS karisik ortam icin varsayilan backend `gloo`dur.
- Senkron DDP en yavas cihazi bekler. MacBook CPU ile katilirsa egitim hizini
  dusurebilir; buna ragmen test, kucuk preset veya veri/akislari dogrulamak icin
  ise yarar.
- Homojen Linux/CUDA GPU kumesinde `--dist-backend nccl` daha hizli olabilir.

## Her cihazda bir kez kurulum

```bash
git clone https://github.com/ChiefVenzox/localllm.git
cd localllm
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

MacBook:

```bash
source .venv/bin/activate
pip install torch
pip install -r requirements.txt
```

Tum makinelerde ayni tokenizer ve ayni `data/bin` bulunmali. Repo icindeki ham
veriyi kullaniyorsan her cihazda sunu calistir:

```bash
python -m tokenizer.train_tokenizer --input data/raw --vocab-size 32000
python -m data.prepare_data --mode pretrain --input data/raw --out data/bin
```

Kendi buyuk verin repo disindaysa `data/bin/train.bin`, `data/bin/val.bin` ve
`data/bin/meta.json` dosyalarini tum cihazlara ayni sekilde kopyala.

## 3 cihazlik ornek

Varsayim:

- Ana masaustu IP: `192.168.1.20`
- Ana masaustu: `node-rank 0`
- Diger masaustu: `node-rank 1`
- MacBook: `node-rank 2`
- Toplam node: `3`

Ana Windows masaustu:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\lan_train.ps1 -NodeRank 0 -MasterAddr 192.168.1.20 -Nodes 3 -Device cuda -Preset small-100m -Data data/bin
```

Diger Windows masaustu:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\lan_train.ps1 -NodeRank 1 -MasterAddr 192.168.1.20 -Nodes 3 -Device cuda -Preset small-100m -Data data/bin
```

MacBook:

```bash
NODE_RANK=2 MASTER_ADDR=192.168.1.20 NNODES=3 DEVICE=cpu PRESET=small-100m DATA=data/bin bash scripts/lan_train.sh
```

MacBook tarafinda MPS denemek istersen `DEVICE=mps` yapabilirsin. PyTorch
surumune gore DDP+MPS destek durumu degisebilir; hata alirsan `DEVICE=cpu`
ile devam et.

## Ayni makinede hizli duman testi

Iki sureci tek makinede denemek icin iki terminal ac.

Terminal 1:

```powershell
$env:MASTER_ADDR="127.0.0.1"; $env:MASTER_PORT="29511"; $env:WORLD_SIZE="2"; $env:RANK="0"; $env:LOCAL_RANK="0"; $env:USE_LIBUV="0"
python train.py --preset nano-demo --data data/bin --device cpu --max-steps 5 --batch-size 2 --grad-accum 1 --dist-backend gloo
```

Terminal 2:

```powershell
$env:MASTER_ADDR="127.0.0.1"; $env:MASTER_PORT="29511"; $env:WORLD_SIZE="2"; $env:RANK="1"; $env:LOCAL_RANK="1"; $env:USE_LIBUV="0"
python train.py --preset nano-demo --data data/bin --device cpu --max-steps 5 --batch-size 2 --grad-accum 1 --dist-backend gloo
```

## Firewall ve ag notlari

- Ana makinede TCP `29500` portu acik olmali.
- Tum cihazlar ayni Wi-Fi/LAN aginda olmali ve ana IP'ye ulasabilmeli.
- VPN, misafir Wi-Fi veya Windows Defender Firewall baglantiyi kesebilir.
- Windows PyTorch wheel'lerinde `libuv` kapali olabilir. Scriptler bunu otomatik
  `USE_LIBUV=0` yapar; elle calistirirsan ayni env degerini ver.
- Farkli port istersen scriptlere `-MasterPort 29501` veya `MASTER_PORT=29501`
  ver.

## Devam etme

Checkpoint sadece `node-rank 0` tarafinda yazilir:

```bash
checkpoints/ckpt.pt
checkpoints/ckpt_last.pt
```

Egitimi yeniden baslatirken `--resume checkpoints/ckpt_last.pt` kullan. Bu dosya
tum cihazlarda ayni path altinda bulunmali; gerekirse ana makineden digerlerine
kopyala.
