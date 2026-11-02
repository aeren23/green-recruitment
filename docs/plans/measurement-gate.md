# Ölçüm kapısı sonuçları (T-05)

Tarih: 2026-11-02. Kapı tanımı: `docs/plans/development-plan.md` §8.1. Tüm sayılar gerçek koşulardan gelir, mock kullanılmadı. Ölçüm betikleri repoda değil, oturumun geçici klasöründedir (`gate/`); yöntemler aşağıda yazılıdır.

## Özet

| Kontrol | Sonuç |
|---|---|
| 1. GPU enerjisi | GEÇTİ. CodeCarbon GPU enerjisi `nvidia-smi` integralinden yalnızca %1,4 ve %2,2 yüksek |
| 2. Tracker maliyeti | KOŞULLU GEÇTİ (kararlar D1 ve D5 ile kabul edildi, bkz. Kararlar). Aşama başına 4,7 ms (200 boş çağrı); ancak GPU boştayken NVML sayaç okuması ara sıra yaklaşık 0,55 s takılıyor. Yüzde 5 sınırı yalnızca LLM özetleri koşu süresine dahil edilirse sağlanıyor |
| 3. CPU yöntemi | GEÇTİ. Yöntem TDP tahmini değil: Windows Energy Meter Interface (gerçek RAPL sayacı). `measurement_notes` içine yazılıyor |
| 4. Boşta güç | Ölçüldü: 25,37 W (30 sn pencere) |

## Ortam

- Dizüstü: Intel Core i5-11400H, RTX 3050 Ti Laptop GPU (4 GB), 15,77 GB RAM, Windows 11 (10.0.26300), Python 3.13.14.
- Güç planı "Güç tasarrufu", OneDrive çalışıyor. Bu kapı için kabul edildi; sayılar bu koşula aittir ve nihai ölçümde plan "Dengeli" yapıldıktan sonra tekrarlanmalıdır (plan U-4).
- GPU, hiçbir yük yokken `nvidia-smi` ile 16 ila 21 W gösterdi (P3 durumu). Bu, ölçüm sırasında arka planda başka bir şeyin GPU'yu hafif kullandığını düşündürür; kaynağı araştırılmadı.
- CodeCarbon 3.3.2, torch `2.6.0+cu124`, `torch.cuda.is_available()` True.

## 1. GPU enerjisi

**Yöntem.** `CodeCarbonStageTracker` içinde `stage("embed")` penceresi boyunca 4096x4096 float32 matris çarpımları (CUDA, art arda 20 çarpım ve senkronizasyon) yaklaşık 35 sn çalıştırıldı. Ayrı iş parçacığı saniyede bir `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` örnekledi. Karşılaştırma için pencerenin başında ve sonunda `pynvml.nvmlDeviceGetTotalEnergyConsumption` sayacı da okundu. Bu üç değer iki bağımsız koşuda karşılaştırıldı.

| | Koşu 1 | Koşu 2 |
|---|---|---|
| Pencere | 37,59 s | 35,85 s |
| `nvidia-smi` örnek sayısı (pencere içi) | 38 | yaklaşık 36 |
| `nvidia-smi` ortalama güç | 29,24 W (17,69 ile 29,98) | 29,25 W |
| `nvidia-smi` integrali (yamuk kuralı) | 0,3040 Wh | 0,2876 Wh |
| NVML toplam enerji sayacı farkı | 0,3092 Wh | 0,2950 Wh |
| CodeCarbon GPU enerjisi | 0,3083 Wh | 0,2940 Wh |
| CodeCarbon / `nvidia-smi` integrali | 1,014 | 1,022 |
| CodeCarbon toplam (CPU + GPU + RAM) | 0,5995 Wh (CPU 0,1874, RAM 0,1039) | 0,5657 Wh (CPU 0,1727, RAM 0,0991) |

**NVML toplam enerji sayacı bu dizüstü GPU'sunda destekleniyor.** CodeCarbon GPU enerjisini güç örneklerinden türetmiyor, sayaçtan okuyor: cihaz nesnesinde `_has_counter=True` ve `_use_power_fallback=False` doğrulandı. Sayaç farkı (0,3092) ile CodeCarbon (0,3083) neredeyse aynı.

**Geçme koşulu** (sıfırdan büyük, aynı büyüklük mertebesi): sağlandı, fark %2 içinde. R-01 riski (dizüstü GPU enerjisi okunamaz) bu makinede gerçekleşmedi.

Dikkat: `nvidia-smi` integrali yaklaşık 1 Hz örneklemeyle hesaplandı, yük sabit olduğu için yeterli. Plan §3.1'deki "ortalama güç 0 W görünüyor" bulgusu bu sürümde tekrarlanmadı.

## 2. Tracker maliyeti

### 2.1 Aşama başına çağrı maliyeti

Kurulum bir kez ödeniyor: `codecarbon` içe aktarma ve donanım algılama toplam yaklaşık 6,8 sn. `CodeCarbonStageTracker` bunu kurucuda (bir atılan "warm_up" görevi dahil) ödüyor; ilk gerçek aşama 10 ms sürdü. Warm-up olmadan ilk aşama 1620 ms sürmüştü (ilk `start_task` içindeki yavaş yüklemeler).

200 boş `stage()` çağrısı (arka arda, GPU sayacı sıcak):

| | ms |
|---|---|
| Ortalama | 4,73 |
| Ortanca | 3,91 |
| %95 | 8,40 |
| En büyük | 14,47 |
| Toplam | 0,946 s |

Ham CodeCarbon API'sinde (aynı sırayla, 200 `start_task` + `stop_task`): `start_task` ortalama 2,12 ms (ortanca 1,71), `stop_task` ortalama 2,05 ms (ortanca 1,71). Yani maliyetin yarısı başlatmada, yarısı durdurmada.

Yedi aşamalı bir başvuru için bu yaklaşık 33 ms eder (7 x 4,73 ms).

### 2.2 Bulgu: NVML sayaç okuması GPU boştayken takılıyor

`nvmlDeviceGetTotalEnergyConsumption`, önceki okumadan sonra yaklaşık 0,3 sn'den fazla geçtiyse bir sonraki okumada 0,5 ila 0,6 sn bloklanıyor. `cProfile` ile doğrulandı: `stop_task` içindeki 514 ms'nin tamamı bu tek çağrıdır. Güç okuması (`nvmlDeviceGetPowerUsage`) bu durumda bile anında döner. Doğrudan NVML ölçümü (boşluk, ortanca okuma süresi):

| Önceki okumadan beri | CUDA bağlamı yok | CUDA bağlamı var |
|---|---|---|
| 0 ile 200 ms | 1 ms | 1 ms |
| 300 ms | 256 ms | 1,3 ms |
| 500 ms | 591 ms | 1,2 ms |
| 1 sn | 512 ms | 1,1 ms |
| 3 sn | 517 ms | 1,2 ms (3 okumadan biri 556 ms) |

Nedeni ölçülmedi; bulguya uyan açıklama, boştaki GPU'nun düşük güç durumundan uyandırılması. CUDA bağlamı tutan süreçte (uygulamada e5 modeli `cuda` üzerinde yüklü) takılma seyrek, ama sıfır değil.

`stage()` içinden gerçek ek süre (GPU boşta, aşama arası boşluklu, 12 tekrar):

| Çalışma / aşama arası boşluk | CUDA bağlamı var (ortanca / ortalama / en büyük) | CUDA bağlamı yok |
|---|---|---|
| 0 / 0 ms | 3,2 / 4,0 / 7,9 ms | 3,7 / 6,2 / 18,5 ms |
| 0 / 500 ms | 4,5 / 49,7 / 538 ms | 602 / 563 / 688 ms |
| 50 / 50 ms | 6,1 / 6,1 / 7,4 ms | 6,3 / 5,9 / 7,9 ms |
| 200 / 200 ms | 5,4 / 6,2 / 9,3 ms | 557 / 380 / 588 ms |
| 1000 / 500 ms | 6,9 / 96,8 / 555 ms | 1128 / 1085 / 1182 ms |
| 0 / 3000 ms | 5,5 / 141 / 556 ms | 543 / 500 / 558 ms |

### 2.3 Başvuru başına tahmini maliyet

**Yöntem.** Altı aşamalı simüle yeşil hat (`extract`, `hash`, `parse`, `filter`, `embed`, `score`), 40 gerçek PDF (`data/raw`), gerçek `extract_text`, `normalize`, `compute_raw_hash`. `parse` (10 ms), `filter` (1 ms) ve `score` (2 ms) uyku ile temsil edildi. `embed` CUDA bağlamıyla küçük bir GPU işi (6 matris çarpımı) ya da CPU modunda 150 ms uyku. Gerçek e5 modeli yerel önbellekte olmadığı için indirilmedi (indirme izin gerektirir); embed süresi bir varsayımdır. Aynı döngü önce `NoOpStageTracker`, sonra `CodeCarbonStageTracker` ile çalıştırıldı.

| | CUDA bağlamı var | CUDA bağlamı yok (embed CPU) |
|---|---|---|
| LLM'siz süre / başvuru (izleyicisiz) | 0,283 s | 0,380 s |
| İzleyici ek süresi / başvuru (ortalama) | 81 ms (ortanca 78) | 477 ms (ortanca 663) |
| LLM'siz yüzde | %28,7 | %125,6 |

Ölçülen ekstra süre, saf çağrı maliyetinin (33 ms) üzerinde; fark NVML takılmalarından geliyor.

**LLM özetli koşu süresi için varsayım (ölçülmedi, Ollama kurulu değil).** Plan §8.2: N=50 için yaklaşık 43 özet çağrısı, yaklaşık 100 başvuru, yani başvuru başına 0,43 çağrı. Çağrı başına 5 sn varsayılırsa (plan T-17 örneğiyle aynı) başvuru başına ek 2,15 sn, koşu süresi yaklaşık 0,283 + 2,15 = 2,43 sn.

| Senaryo | Maliyet | Yüzde | Sonuç |
|---|---|---|---|
| CUDA bağlamı var, LLM özetleri dahil, 5 sn/çağrı | 81 ms / 2,43 s | %3,3 | geçer |
| CUDA bağlamı var, LLM özetleri hariç | 81 ms / 0,283 s | %28,7 | kalır |
| CUDA bağlamı yok (embed CPU), LLM özetleri dahil | 477 ms / 2,53 s | %18,9 | kalır |
| Yalnızca çağrı maliyeti (33 ms), LLM'siz | 33 ms / 0,283 s | %11,7 | kalır |

Yüzde 5 sınırını sağlamak için başvuru başına koşu süresinin CUDA bağlamlı durumda en az 81 ms / 0,05 = 1,6 sn, bağlamsız durumda 9,5 sn olması gerekir. Bu, çağrı başına LLM süresinin yaklaşık 3,1 sn (bağlamlı) ya da 21 sn (bağlamsız) olması demek.

**Sonuç: KOŞULLU.** Geçme yalnızca (a) embed `cuda` üzerindeyken, (b) özet çağrıları koşu süresine dahilse, (c) çağrı başına en az yaklaşık 3 sn varsayımıyla sağlanıyor. Ayrıca baseline hattında 4 aşama var ve LLM çağrıları saniyeler sürdüğü için orada yüzde küçük, ama GPU'yu Ollama kullandığından süreçte CUDA bağlamı yok; `store_pdf` ve `extract` aşamaları (LLM'den önce, GPU boştayken) tek başına 0,55 sn'lik takılmalar yaşayabilir.

Brüt enerji üzerindeki etki: takılma sırasında makine boşta güce yakın çeker (yaklaşık 25 W x 0,55 s = 0,004 Wh) ve bu süre hem `duration_s` içinde hem brüt enerjide yer alır. Tek bir 50 ms'lik aşamanın gerçek enerjisi 0,0007 Wh'tir. Bu yüzden brüt değer takılma sıklığına duyarlıdır; `energy_wh_net = brüt - boşta_güç x süre / 3600` düzeltmesi takılma süresini kısmen dengelemelidir. Bu son cümle muhakemedir, ölçülmedi.

### 2.4 Pencereden kısa aşamalarda enerji dağılımı (R-02)

Kısa aşamalar sıfır ya da negatif değil, ama hataya açık. Boşta makine, 1 ms ve 50 ms uyku aşamaları (100'er tekrar):

| | 1 ms (duvar 5,7 ms) | 50 ms (duvar 55,9 ms) |
|---|---|---|
| Enerji ortalaması | 0,036 mWh | 0,657 mWh |
| Standart sapma | 0,046 mWh (%127) | 0,229 mWh (%35) |
| En küçük / en büyük | 0,0055 / 0,273 mWh | 0,311 / 1,010 mWh |
| Toplam sıfır | 0 / 100 | 0 / 100 |
| CPU bileşeni sıfır | 59 / 100 | 0 / 100 |
| GPU bileşeni sıfır | 96 / 100 | 45 / 100 |
| Ima edilen güç (enerji / duvar süresi) | 3,6 ile 100 W (ortalama 21,5) | 20 ile 67 W (ortalama 42) |

Karşılaştırma için 0,5 sn ve 2,5 sn'lik aşamalarda ima edilen güç 34 ± 9 W ve 28,2 ± 0,6 W. 2,5 sn'lik aşamalarda enerji standart sapması yaklaşık %2.

Bilinen sabit GPU yükü altında (arka plan iş parçacığı, `nvidia-smi` gerçeği yaklaşık 29,9 W) aşama başına GPU enerjisi:

| Aşama (uyku) | Duvar süresi | GPU ima edilen güç, toplamdan (gerçek 29,9 W) | GPU sıfır | Tek aşamada en büyük |
|---|---|---|---|---|
| 1 ms | 6,6 ms | 5,0 W (%17) | 145 / 150 | 301 W |
| 50 ms | 56,8 ms | 23,2 W (%78) | 46 / 100 | 52 W |
| 500 ms | 506 ms | 29,2 W (%98) | 0 / 20 | 35 W |
| 2,5 s | 2506 ms | 29,7 W (%99) | 0 / 6 | 29,8 W |

Yorum: GPU enerji sayacı kabaca 100 ms'lik adımlarla güncelleniyor gibi davranıyor; kısa pencerelerde ya 0 ya da bir yığın veriyor, toplamda ise gerçeğin altında kalıyor. Yaklaşık 100 ms'nin altındaki aşamalarda (`hash`, `filter`, `score`, `rank`) aşama başına enerji güvenilir değil. Raporda bu aşamalar "yaklaşık" işaretlenmeli ve toplam enerji esas alınmalı (R-02 önlemi). Önlem gerekçesi bu sayılarla doğrulandı.

### 2.5 Seçenekler (karar sizin)

Hiçbiri uygulanmadı.

1. **Kabul et.** `EMBED_DEVICE=cuda` ile e5 yüklü kalır (CUDA bağlamı tutulur), süre ve enerji toplam üzerinden raporlanır, raporda aşama başına değerler yaklaşık işaretlenir ve takılmanın sıklığı koşu çıktısına eklenir.
2. **Aşama sayısını azalt.** Çok kısa ve ucuz aşamalar (`hash`, `filter`, `score`, `rank`) tek bir aşamada toplanır ya da ayrı ölçülmez. Bu SPEC §7.4'teki aşama listesini ve CLAUDE.md kural 6'yı değiştirir, §8.1 notuna göre de artımlı top-k akışını etkileyebilir.
3. **GPU sayacını sıcak tutmak.** Arka planda sayaç okuyan bir iş parçacığı takılmayı büyük olasılıkla engeller, ama GPU'nun düşük güç durumuna inmesini engelleyerek ölçtüğünüz boşta GPU gücünü değiştirir. Önerilmez; denenmedi.
4. **İzleyiciyi koşu başına tek göreve indirmek** (aşama başına değil). Aşama kırılımı kaybolur; SPEC §5 `StageMeasurement` ve §7.4 ile çelişir.

## 3. CPU ve RAM yöntemi

CodeCarbon günlüğü ve kaynak kod okuması:

- **CPU: `CPU(Windows Emi)`.** Windows Energy Meter Interface, Windows 11'in sunduğu RAPL sayaçları (`core/windows_emi.py` başlığı). Cihaz kanalları: `RAPL_Package0_PKG`, `RAPL_Package0_DRAM`, `RAPL_Package0_PP0`, `RAPL_Package0_PP1`. Seçilen kanal yalnızca `RAPL_Package0_PKG` (DRAM `rapl_include_dram=False` olduğu için hariç). Enerji doğrudan sayaç farkından geliyor, güç varsayımı yok. Yedek olarak hesaplanan 85 W "generic TDP" bu modda kullanılmıyor.
- **GPU: NVML sayacı** (`pynvml`), bkz. kontrol 1.
- **RAM: sabit güç modeli** (`external/ram.py`). 15,77 GB için 2 DIMM tahmini x 5 W = 10 W; enerji = 10 W x süre, yüke bağlı değil. Bu bir tahmindir ve yükle değişmez.
- `measurement_notes` örneği (gerçek çıktı): `cpu_method=CPU(Windows Emi); cpu_model=11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz; gpu=1 x NVIDIA GeForce RTX 3050 Ti Laptop GPU; hardware=RAM(),GPU() (NVIDIA GeForce RTX 3050 Ti Laptop GPU),CPU(Windows Emi); codecarbon=3.3.2; tracking_mode=machine; country_iso=TUR` (koşu kayıtlarında sonuna `embed_device=...` eklenir).

**SPEC ile fark.** SPEC §11 ve plan §3.1, Windows'ta CPU enerjisinin TDP'den tahmin edileceğini varsayıyordu. Bu makinede gerçek sayaç (EMI) kullanılıyor. Rapora "CPU enerjisi tahmindir" yazılmamalı; doğrusu "CPU paket enerjisi Windows EMI (RAPL) sayacından, RAM enerjisi 10 W sabit tahmin". Paket sayacı ekran, SSD, Wi-Fi gibi bileşenleri kapsamaz.

## 4. Boşta güç

Komut: `python -m bench idle --seconds 30` (gerçek koşu).

| | |
|---|---|
| Boşta güç | 25,37 W |
| Enerji | 0,2196 Wh |
| Pencere | 31,17 sn (istenen 30; fazlası görev açma/kapama ve NVML takılması) |
| Zaman damgası | 2026-11-02T00:08:44 UTC |
| Dosya | `data/bench/idle_power.json` (git'e girmez) |

Bu 30 sn'lik deneme içindir. SPEC §7.5 nihai ölçüm için `--seconds 120` ister. Boşta güç, RAM'in 10 W sabit modelini ve GPU'nun gerçek boşta gücünü içerir; "Güç tasarrufu" planına aittir.

## CodeCarbon 3.3.2 API bulguları (kurulu kaynaktan)

Dosya: `.venv/Lib/site-packages/codecarbon/`.

- `emissions_tracker.py:797` `start_task(task_name=None)`, `:878` `stop_task(task_name=None) -> EmissionsData`. SPEC'teki adlar doğru.
- Birimler: `EmissionsData.energy_consumed`, `cpu_energy`, `gpu_energy`, `ram_energy` kWh (`core/units.py: Energy` "Measured in kWh"; `:1070` `_prepare_emissions_data` alanları `.kWh` ile doldurur). `emissions` kg CO2, `duration` saniye. Tracker kWh x 1000 ile Wh'e çeviriyor.
- Zamanlayıcı: `start_task` ana `_scheduler`'ı durdurur (`:829`), 1 sn'lik `_scheduler_monitor_power`'ı başlatır; `stop_task` onu durdurur (`:885`) ve `_measure_power_and_energy()` ile enerjiyi yalnızca durdurma anında hesaplar (`:894`). Yalnızca `start_task` / `stop_task` kullanımında `measure_power_secs` etkisizdir; 1 sn'lik izleme aralığı kodda sabittir (`:358`, `interval=1`). SPEC'teki `measure_power_secs=1` zararsız ama etkisiz.
- Aynı ada ikinci `start_task` çağrısında ad `uuid` ile yeniden adlandırılır (`:840`). `stop_task("ad")` ilk (eski) görevi bulur ve süresini yanlış hesaplar. Bu yüzden tracker `stop_task()` çağrısını adsız yapar (etkin görevi durdurur). `_tasks` sözlüğü her aşama çağrısı için bir giriş biriktirir, hiç temizlenmez; çok uzun koşularda bellekte küçük ama sınırsız bir birikim olur.
- Aktif görev varken ikinci `start_task` yalnızca uyarı verip döner (`:835` "A task is already under measure") ve izlemeyi sessizce bozar. Bu yüzden iç içe aşama `NestedStageError` ile reddediliyor.
- `allow_multiple_runs` v3'te varsayılan True (`_configure_multiple_runs`, `:165`). False olsaydı `%TEMP%\.codecarbon.lock` kilit dosyası oluşurdu ve ikinci örnek sessizce çıkardı. Kodda bu parametre geçilmiyor; varsayılan kilit dosyası yaratmaz.
- `OfflineEmissionsTracker.__init__` (`:1426`) `@suppress(Exception)` ile sarılı: kurulum hataları yutulur, yarım kurulmuş nesne döner. Tracker bu durumu ayrıca algılamıyor.
- `save_to_file=False` ile `emissions.csv` oluşmadı; depoda ve çalışma klasöründe artık dosya yok.
- `tracking_mode="machine"`: CPU ve GPU makine genelini ölçer; bu yüzden Ollama (ayrı süreç) GPU enerjisi görünür. CPU yolu `core/resource_tracker.py:_try_platform_cpu_backend` içinde Windows'ta önce `windows_emi.is_emi_available()` dener.
- Bu GPU'da `nvmlDeviceGetUtilizationRates` `NVMLError_Unknown` fırlatıyor. CodeCarbon bunu her saniye WARNING ve iz ile günlüğe yazıyor (`log_level="info"` ile görüldü); `log_level="error"` ile bastırılıyor. Etkisi: rapor edilen `gpu_utilization_percent` yanlışlıkla 0, enerji etkilenmiyor.
- Kurucu, düzey ayarlanmadan önce "offline tracker init" ve "Multiple instances ... allowed" satırlarını bir kez günlüğe yazar; zararsız, CV verisi içermez.
- Ortalama güç alanları (`cpu_power` vb.) tutarsız (EMI modunda 0,04 W ile 1,9 W arası gözlendi); yalnızca enerji alanları kullanılıyor.

## Kod tarafında verilen kararlar

- `duration_s`, `start_task` çağrısından önce başlayıp `stop_task` sonrası biten duvar saatidir: enerji penceresiyle aynı aralık, böylece boşta düzeltmesi (`boşta_güç x duration_s`) aynı süreyi kullanır. Tracker ek süresi bu süreye dahildir.
- Eşzamanlı istek yok varsayımıyla tek etkin aşama desteklenir. Web katmanında eşzamanlı istekler aynı izleyiciyi paylaşırsa `NestedStageError` alır; canlı demoda `GREEN_TRACKING` ya serileştirilmeli ya da aşama izleme yalnızca benchmark'ta açılmalı. Bu karar sonraki görevlere (T-12 ve sonrası) aittir.

## Kararlar

Kapı sonuçları gözden geçirildi (koordinatör kontrolü: CUDA bağlamıyla boşluktan sonraki aşama ortanca 4,7 ms, bağlamsız ortanca 263 ms, en büyük 530 ms; CPU yöntemi `CPU(Windows Emi)`). Kontrol 2 bu kararlarla **koşullu geçti** kabul edilir.

- **D1. Seçenek 1 kabul edildi.** Aşama başına izleme SPEC'te tanımlandığı gibi kalır. Koşu toplamları ve net enerji birincil sayılardır. Yaklaşık 100 ms'den kısa aşamalar raporda "yaklaşık" işaretlenir.
- **D2. Adillik.** CUDA bağlamı olmayan süreçte NVML takılması baseline'ın brüt enerjisini ve süresini şişirir ve karşılaştırmayı yeşil lehine yanlı yapar. Bu yüzden iki benchmark koşucusu ve boşta ölçümü, ölçen süreçte tüm koşu boyunca bir CUDA bağlamı tutar. Gereksinim: T-09, T-10 ve T-17.
- **D3. Web katmanı.** Web katmanı `GREEN_TRACKING` ne olursa olsun her zaman `NoOpStageTracker` kullanır: rapor sayıları benchmark'tan gelir (SPEC 1) ve tracker iş parçacığı güvenli değildir. Gereksinim: T-13.
- **D4. Kısa GPU aşamaları eksik sayılıyor** (50 ms'de %78), bu `embed` aşamasını olduğundan iyi gösterir. T-09 tüm koşu için çapraz kontrol ekler: koşu başında ve sonunda bir GPU enerji sayacı okunur ve aşamalar üzerinden toplanan GPU enerjisinin yanında raporlanır. Yeni bağımlılık yok; `nvidia-smi` ya da kurulu araçlarla yapılamıyorsa T-09'da bildirilir.
- **D5. Yeniden değerlendirme.** %5 maliyet kontrolü yalnızca varsayılan LLM süresiyle (5 sn/çağrı) geçiyor; K1'de gerçek LLM çağrılarıyla yeniden değerlendirilir. Kapı, nihai ölçümlerden önce "Dengeli" güç planında tekrarlanır.
