# Senkron yerel LLM yönergesi

Bu proje için tek önerilen ağ mimarisi:

- FastAPI ana sunucu: `http://SUNUCU_IP:8000`
- Ana arayüz: `http://SUNUCU_IP:8000/`
- Cihaz paneli: `http://SUNUCU_IP:8000/cluster`
- Filebrowser/veri paylaşımı varsa: `http://SUNUCU_IP:8081`
- Worker `--server` değeri: her zaman `http://SUNUCU_IP:8000`

`8081` worker adresi değildir. Eski ayrı panel akışı bu kurulumda kullanılmaz.

## 1. Ana sunucu

Ana makinede repo klasöründe:

```bash
export YERELLM_API_TOKEN="uzun-rastgele-token"
python -m server.app --host 0.0.0.0 --port 8000 --api-token "$YERELLM_API_TOKEN"
```

Systemd ile çalışan kurulumda servis zaten bunu yapar:

```bash
systemctl --user status localllm-chat-server.service
```

Beklenen sağlık cevabı:

```bash
curl http://127.0.0.1:8000/api/health
```

## 2. Her cihazı worker olarak bağla

Her Mac/Windows/Linux cihazda aynı repo ve sanal ortam hazır olmalı. En kolay
çalışma şekli setup scriptidir:

```bash
python scripts/setup_worker.py --server http://SUNUCU_IP:8000 --name cihaz-adi --token "$YERELLM_API_TOKEN"
```

Windows örneği:

```powershell
py -3 scripts\setup_worker.py --server http://SUNUCU_IP:8000 --name windows-rtx --token "%YERELLM_API_TOKEN%"
```

Script `.yerellm_worker.env` dosyasını yazar ve worker'ı başlatır. Aynı cihazda
sonra yeniden başlatmak için:

```bash
python scripts/setup_worker.py
```

Sadece ayar dosyasını yazmak istersen:

```bash
python scripts/setup_worker.py --server http://SUNUCU_IP:8000 --name cihaz-adi --token "$YERELLM_API_TOKEN" --save-only
```

Sunucu makinenin kendisi de worker olacaksa sabit kimlik kullan:

```bash
python scripts/setup_worker.py \
  --server http://127.0.0.1:8000 \
  --name server-local \
  --node-id server-local \
  --role server-local \
  --repo /home/hakan/localllm \
  --device cuda \
  --token "$YERELLM_API_TOKEN"
```

Setup script varsayılan olarak yapılandırılmış eğitim işlerini açar. Bu şunlara
izin verir:

- patch/dosya senkronu
- yapılandırılmış chat fine-tune işi
- yapılandırılmış DDP eğitim işi
- worker tarafında chat inference işi

Rastgele shell komutu çalıştırmaz. Genel shell komutu için ayrıca
`--allow-remote-commands` gerekir; normal kullanımda kapalı bırak.
Eğitim işlerini de kapatmak istersen setup scriptine `--no-training-jobs` ekle.

## 3. Senkron kontrolü

Ana makinede:

```bash
curl http://127.0.0.1:8000/api/nodes
curl http://127.0.0.1:8000/api/sync/status
```

Panelden:

1. `http://SUNUCU_IP:8000/` adresini aç.
2. `Cihazlar` bölümünde cihazların online olduğunu kontrol et.
3. `Senkron` bölümünden seçili cihazlara patch senkronu gönder.
4. `Eğitim` bölümünden yapılandırılmış chat eğitim işi kuyruğa al.
5. Sohbet sonucunu worker adapter'ıyla görmek için ana panelde cevap kaynağını
   `Otomatik worker/server` veya `Worker modeli` seç.

## 4. Veri ve checkpoint kuralı

Tüm cihazlarda aynı temel dosyalar bulunmalı:

- `tokenizer/tokenizer.json`
- `checkpoints/ckpt.pt`
- opsiyonel adapter: `checkpoints/adapter.pt`
- eğitim için hazırlanmış veri: ör. `data/chat_200m_plus_bin`

Büyük veri ve checkpoint dosyalarını Docker image içine koyma. Host volume,
Filebrowser veya manuel kopya ile paylaş.

Worker chat eğitimi varsayılan olarak temiz adapter başlatır. Eski adapter
üzerinden devam etmek istersen `adapter_resume` değerini API payload'ında açıkça
gönder; panel varsayılanı eski adapter'ı otomatik kullanmaz.

## 5. Üç cihaz için pratik düzen

Önerilen kurulum:

- Ana Linux sunucu: FastAPI + panel + registry + gerekirse CUDA worker
- Güçlü GPU cihazı: `setup_worker.py` ile eğitim/senkron worker
- Zayıf CPU/MPS cihazı: küçük deneme, veri hazırlama veya ayrı adapter işi

Senkron DDP eğitimde yavaş cihaz tüm koşuyu yavaşlatır. Aynı DDP koşusunda
benzer hızdaki GPU'ları birlikte kullan; CPU/MPS makineleri ayrı işler için tut.

## 6. Derleme ve tanılama

```bash
python -m compileall -q chat_demo.py chat_template.py config.py generate.py online_learn.py quick_intents.py train.py train_intent.py web_learn.py web_lookup.py web_search.py data model scripts server tokenizer tsetlin worker docker
python scripts/system_check.py --mode all
```

Docker kullanacaksan:

```bash
make docker-doctor
docker compose config
```

`8000` portu ana servis tarafından kullanılıyorsa Docker API için
`YERELLM_PORT` değerini değiştir.
