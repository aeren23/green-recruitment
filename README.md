# Green Hiring Pipeline

## Proje

Green Hiring Pipeline, özgeçmiş ön eleme sürecini kademeli (tiered), veri açısından minimal ve ölçülebilir kılan tek servisli bir FastAPI uygulamasıdır. Aynı kod, benchmark CLI'ı (`bench/`) tarafından da çalıştırılır. İddia şudur: aynı veri ve aynı makinede, kademeli ve veri minimizasyonlu bir hat, geleneksel bir hatta göre başvuru başına daha az enerji, depolama ve LLM çağrısı harcar ve kısa liste kalitesi kabul edilebilir kalır. Bu iddia `bench/` ile ölçülür; demo uygulama aynı mantığı canlı gösterir, rapordaki sayılar benchmark'tan gelir. Arayüz Türkçedir; kod, komutlar ve tanımlayıcılar İngilizcedir.

Tasarım `docs/SPEC.md` dosyasındadır, görev sırası ve kabul kriterleri `docs/TASKS.md` dosyasındadır.

## Kurulum

Gereksinimler: Windows 11, Python 3.13, CUDA destekli PyTorch (makinede önceden kurulu; ölçümlerde `torch 2.6.0+cu124` kullanıldı), NVIDIA GPU ve Ollama.

### 1. Sanal ortam ve bağımlılıklar

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e .
```

`--system-site-packages` sayesinde sanal ortam sistemdeki CUDA destekli torch'u görür. `torch` bağımlılık listesinde yoktur; kurulum sırasında yeniden kurulmamalı ve sürümü değiştirilmemelidir. Kurulumdan sonra kontrol edin:

```powershell
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` `True` dönmelidir.

### 2. Ortam dosyası

```powershell
Copy-Item .env.example .env
```

`.env` içindeki `HASH_SECRET` ve `EMPLOYER_TOKEN` değerlerini rastgele değerlerle değiştirin:

```powershell
.venv/Scripts/python -c "import secrets; print(secrets.token_hex(32))"
```

`HASH_SECRET` değişirse tüm `raw_hash` değerleri değişir ve mevcut parse önbelleği işe yaramaz. Bu değerler repoya ya da README'ye yazılmamalıdır.

### 3. Embedding modelini bir kez indirin

```powershell
.venv/Scripts/python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small')"
```

`.env` içindeki `EMBED_LOCAL_FILES_ONLY=true` ayarı modeli yalnızca yerel önbellekten yükler; uygulama açılışta ağa çıkmaz. Önbellekte model yoksa uygulama hata mesajı verir.

### 4. Ollama ve yerel LLM

Ollama'yı resmi sitesinden kurun, ardından:

```powershell
ollama pull qwen2.5:3b
ollama ps
```

`ollama ps` çıktısındaki `PROCESSOR` sütunu `100% GPU` göstermelidir. Ollama `http://localhost:11434` adresinde çalışmalıdır (`OLLAMA_URL`). Ollama çalışmıyorsa uygulama çalışmaya devam eder, yalnızca özet alanı boş kalır (SPEC §6.6).

### 5. Veri seti (yalnızca benchmark için)

Kaggle `snehaanbhawal/resume-dataset` veri setini `data/raw/` altında şu yapıda yerleştirin:

```
data/raw/Resume/Resume.csv
data/raw/data/data/<KATEGORİ>/<ID>.pdf
```

`data/` klasörü git dışındadır. Yapıyı kontrol edin (kök dizin `--root` ile verilebilir):

```powershell
.venv/Scripts/python -m bench dataset-stats
```

Beklenen değerler: 24 kategori, 2484 kayıt, 2484 PDF, 0 eksik PDF.

## Çalıştırma

### Demo ilanları

```powershell
.venv/Scripts/python -m app.seed
```

Üç demo ilanı oluşturur: Backend Developer, Accountant ve HR Specialist.

### Uygulama

```powershell
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Embedding modeli açılışta (lifespan) bir kez yüklenir. Varsayılan port 8000 üzerinde şu sayfalar açılır:

| Sayfa | Adres | Kim |
|---|---|---|
| Açık ilanlar | http://127.0.0.1:8000/ | aday |
| İşveren girişi | http://127.0.0.1:8000/employer/login | işveren (`EMPLOYER_TOKEN`) |
| Yeşil panel | http://127.0.0.1:8000/green | herkes |

### Beş dakikalık demo

Demo CV'si `docs/demo/demo-cv.pdf` dosyasıdır. Akış SPEC §3'e göre:

1. İşveren olarak Backend Developer ilanını açın; sert filtreler ve ağırlıklı gereksinimler görünür.
2. Aday olarak `demo-cv.pdf` dosyasını yükleyin, ön izlemeyi düzeltip onaylayın. "Dosyanız saklanmadı" mesajı görünür.
3. Aynı CV'yi Accountant ilanına yükleyin. Ön izleme önceki onaylı veriden dolar ve "yeniden işlenmedi" der.
4. Aynı CV'yi Backend Developer ilanına tekrar yükleyin. `DUPLICATE_APPLICATION` ile engellenir.
5. İşveren olarak sıralı listeyi (anonim) açın, bir adayın matrisine bakın ve shortlist yapın. İletişim bilgisi yalnızca bu aşamada görünür.
6. İlanı kapatın. Saklama kuralları çalışır ve panel sayaçları güncellenir.
7. `/green` sayfasında karşılaştırma tablosu, sayaçlar ve sayfa ağırlığı görünür.

Canlı demo ölçüm yapmaz; web katmanı her zaman ölçüm izleyicisi olmadan çalışır. Panelin benchmark tablosu `data/bench/bench.db` dosyasını salt okunur açar.

## Testler

```powershell
.venv/Scripts/python -m pytest -q
```

Testler GPU ya da Ollama gerektirmez; embedding ve LLM çağrıları testlerde mock'lanır. Birim testleri zorunludur: `ingest/normalize`, `ingest/sections`, `scoring/matrix`, `scoring/filters` ve `privacy/retention`.

## Benchmark ve ölçüm protokolü

Komutlar:

```powershell
.venv/Scripts/python -m bench idle --seconds 120
.venv/Scripts/python -m bench run --pipeline green --n 50
.venv/Scripts/python -m bench run --pipeline baseline --n 50
.venv/Scripts/python -m bench compare
.venv/Scripts/python -m bench dataset-stats
.venv/Scripts/python -m bench parse-report
```

- `idle`: boşta güç (W) ölçer. Net enerji bu değerle düzeltilir.
- `run --pipeline green|baseline`: bir koşu yapar. Diğer senaryo parametreleri için `--help` kullanın. K1'de kullanılan senaryo: `seed=42`, `jobs_per_cv=2`, `resubmit_rate=0.1`, `top_k=5`.
- `compare`: koşuları `scenario_json` değerine göre gruplar, grup başına ortalama ± standart sapma hesaplar ve `data/bench/report.md` ile `report.json` dosyalarını yazar.
- `dataset-stats`: veri setinin yapısını doğrular.
- `parse-report`: CV ayrıştırma raporunu üretir (ayrıntı için `--help`).

Ölçüm protokolü:

1. Güç planı "Dengeli", bilgisayar prizde.
2. OneDrive ve diğer uygulamalar kapalı. Arka plan yükü düşük tutulur (K1'de %1'in altı).
3. Git ağacı temiz olmalıdır. Koşunun commit hash'i kaydedilir.
4. Boşta güç koşulardan önce ve sonra birkaç kez ölçülür (K1'de önce üç, sonra bir ölçüm; `idle --seconds 120`).
5. Yeşil ve baseline koşuları dönüşümlü çalıştırılır. Her biri 3 kez tekrarlanır; K1'de toplam 6 koşu yapıldı.
6. `compare` raporuna koşul tablosu (güç planı, OneDrive, git durumu, boşta güç) eklenir. Nihai koşullara uymayan koşular raporda uyarılır.

Ölçüm yöntemi: CodeCarbon `OfflineEmissionsTracker` (`tracking_mode=machine`, ülke TUR), her aşama `stage("<ad>")` bağlam yöneticisiyle ölçülür. CPU enerjisi Windows Energy Meter Interface (RAPL) sayacından, GPU enerjisi NVML sayacından gelir; RAM enerjisi yüke bağlı olmayan 10 W sabit modeldir.

Çıktılar `data/bench/` altına yazılır: `bench.db` (koşular ve aşama ölçümleri), `report.md` ve `report.json` (karşılaştırma), `idle_power.json` (boşta güç), `baseline_store/` (baseline'ın PDF deposu). `data/` git dışındadır.

## Sonuçlar (K1, N=50)

K1 değerlendirmesi 111 başvuru üzerinde yapıldı (bunların 11'i tekrar başvuruydu), Dengeli güç planı ve prizde çalışma koşulunda. Önceden belirlenen dört ölçütün sonuçları:

| Ölçüt | Kural | Baseline | Yeşil | Fark | Sonuç |
|---|---|---|---|---|---|
| Başvuru başına net enerji | yeşil < baseline | 0,01647 ± 0,00192 Wh | 0,01123 ± 0,00064 Wh | −%31,8 | GEÇTİ |
| LLM çağrısı | yeşil < baseline | 111 | 44 | −%60,4 | GEÇTİ |
| Retention sonrası depolama | yeşil < baseline | 3.777.982 B | 147.456 B | −%96,1 | GEÇTİ |
| precision@10 | yeşil ≥ baseline − 0,10 (eşik 0,40) | 0,500 | 0,600 | +0,10 | GEÇTİ |

Diğer sonuçlar: başvuru başına brüt enerji 0,0407 Wh'den 0,0272 Wh'ye (−%33,3), süre −%34,3, overlap@10 0,50.

Nihai N=300 sonuçları T-17'de yayımlanacaktır.

## Sınırlılıklar

- **CPU ve RAM ölçümü:** CPU enerjisi Windows EMI (RAPL) paket sayacından gelir; ekran, SSD ve Wi-Fi gibi bileşenleri kapsamaz. RAM enerjisi sabit bir 10 W tahminidir, ölçülmüş bir değer değildir. CPU enerjisi TDP tahmini değildir.
- **Gömülü karbon (M):** SCI hesabında M dışarıda bırakılmıştır: SCI = co2_g / başvuru sayısı.
- **Filtreler:** Benchmark'ta dil ve konum filtreleri kapalıdır. Sert filtre yalnızca `min_years` üzerinde çalışır.
- **Baseline modeli:** Baseline küçük bir 3B model (`qwen2.5:3b`) ile puanlar. Bu yüzden `overlap@k` yardımcı bir metriktir, ana kalite ölçütü `precision@k`'dır. Yeşil hattın kalite üstünlüğünün bir kısmı zayıf baseline'dan gelir.
- **Net enerji ve boşta güç:** Net enerji boşta güce duyarlıdır; K1 koşularında 27 ile 42 W arası ölçüldü. Net fark bu aralıkta %31,5 ile %32,6 arasında kaldı. Bu yüzden brüt ve net değerler birlikte raporlanır.
- **Kısa aşamalar:** Yaklaşık 100 ms'den kısa aşamaların (`hash`, `filter`, `score`, `rank`) enerjisi yaklaşıktır. GPU sayacı kabaca 100 ms adımlarla güncellenir. Karşılaştırmada toplam esas alınmalıdır.
- **Ölçüm maliyeti:** İzleyici ve NVML okuma maliyeti green hattı biraz cezalandırır. Örneğin `extract` çağrısı başına 290 ms (yeşil) ve 270 ms (baseline). Bu nedenle sonuçlar muhafazakâr yönde yorumlanmalıdır.
- **GPU enerjisi:** Aşama toplamları, koşu boyunca ölçülen GPU enerjisinin yaklaşık %3 ile %6'sını kaçırıyor (K1: %94,4 ile %97,1 yakalandı).
- **LLM süresi:** Yeşil hattın LLM aşaması koşudan koşuya değişiyor (toplam süre 114 ile 199 s). Nedeni doğrulanmadı; T-17'de araştırılacak.
- **`min_years` filtresi:** Veri setinde medyan deneyim 12 yıl olduğundan filtre nadiren eler (K1'de 4 başvuru). Kalite farkı büyük ölçüde sıralamadan gelir.
- **Veri seti:** Veri seti anonimleştirilmiştir; ad ve iletişim bilgileri şablon yer tutucularıyla değiştirilmiştir. İletişim bilgisi çıkarımı yalnızca sentetik testlerle doğrulanmıştır. Veri seti Türkçe içermez; Türkçe başlıklar da yalnızca sentetik testlerle sınanır.
- **Önbellek gizliliği:** Parse önbelleği `raw_hash` ile anahtarlanır. Aynı PDF'e sahip herkes, önceki onaylı profili ön izlemede görebilir. Bu, SPEC §6.2'deki hash tasarımının bilinen bir sonucudur.
- **Tek makine:** Tüm ölçümler tek bir dizüstünde (Intel Core i5-11400H, RTX 3050 Ti Laptop GPU, 4 GB VRAM) yapıldı. Sonuçlar başka donanımlara genellenemez.

## Proje yapısı

Dizin yapısı `docs/CLAUDE.md` içindeki "Layout" bölümünde açıklanmıştır. Tasarım kararları ve ölçüm kayıtları `docs/plans/` altındadır:

- `docs/plans/measurement-gate.md`: ölçüm kapısı sonuçları (GPU, tracker maliyeti, CPU yöntemi, boşta güç)
- `docs/plans/dataset-findings.md`: veri seti bulguları ve ayrıştırma çıkarımları
- `docs/plans/K1-evaluation.md`: K1 değerlendirmesi (N=50)
- `docs/plans/K1-report.md`: K1 benchmark raporu
