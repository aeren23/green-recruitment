# Geliştirme Planı: Green Hiring Pipeline

- **Durum:** Onaylandı
- **Tarih:** 2026-11-01
- **Teslim:** yaklaşık 2026-11-06 (5 gün; kesin tarih netleşince bu satır güncellenir)
- **Kaynaklar:** `docs/SPEC.md`, `docs/TASKS.md`, `docs/CLAUDE.md`, `docs/rules/coding-standarts.md`, `docs/rules/logging-standarts.md`

Esas kapsam SPEC ve TASKS'tır. Bu plan görev sırasını değiştirmez ve TASKS kabul kriterlerinin yerine geçmez; onlara uygulama notları ekler. SPEC'ten sapma gerektiren her nokta §4'te açıkça listelenmiştir.

## 1. Yaklaşım

1. **Sıra sabit:** T-00'dan T-18'e kadar TASKS sırası izlenir. Bonuslar (B-01, B-02, B-03) süre nedeniyle plan dışıdır.
2. **Önce iddia, sonra arayüz:** K1'e (T-11) kadar arayüz yazılmaz. Sayılar beklenmedik çıkarsa neden bulunmadan T-12'ye geçilmez.
3. **Kapsam dar tutulur:** SPEC'te olmayan özellik eklenmez. Soyutlamalar §5.2'deki dört arayüzle sınırlıdır. Yeni bağımlılık, tablo veya uç nokta sorulmadan eklenmez.
4. **Riskli varsayımlar erken sınanır:** Ölçümle ilgili belirsizlikler T-05'teki kapıda çözülür. Sonradan öğrenilirse K1 boşa gider.

## 2. Kararlar

| No | Konu | Karar | Kaynak |
|---|---|---|---|
| D-01 | Elenen adayın durum sayfası | `DecisionRecord`'a nullable `application_token` eklenir | Kullanıcı |
| D-02 | PDF'in diske düşmemesi | 1 MB sert limit (`MAX_UPLOAD_BYTES`), aşımda 413 | Kullanıcı |
| D-03 | Profilin yeniden onaylanması | Değişen profilin `ranked` başvuruları yeniden skorlanır | Kullanıcı |
| D-04 | Eksik uç noktalar | 5 uç nokta eklenir; retention tetikleme işveren çerezi ister | Kullanıcı |
| D-05 | Repo konumu | Proje OneDrive dışına taşınır (`C:\dev\green-recruitment`) | Kullanıcı |
| D-06 | Süre ve kapsam | Yaklaşık 5 gün; bonus yok, kapsam büyütülmez | Kullanıcı |
| D-07 | Kalite eşiği | Yeşil `precision@10` ≥ baseline `precision@10` − 0.10 | Kullanıcı |
| D-08 | Formatlayıcı ve linter | Eklenmez. `coding-standarts.md` §3.6 için bilinçli istisna; PEP 8 elle izlenir | Kullanıcı |
| D-09 | Python ortamı | Python 3.13 üzerinde `--system-site-packages` ile `.venv`; torch paylaşılır | Varsayılan |
| D-10 | Git akışı | `master` → `main`; görev commit'leri `develop` dalına, K1 ve MVP'de `main`'e birleştirme | Varsayılan |
| D-11 | Plan dili | Plan Türkçe; kod, commit ve `project_log.md` kayıtları İngilizce (CLAUDE.md) | Varsayılan |

## 3. Ortam bulguları ve ön koşullar

### 3.1 Doğrulanan bulgular (2026-11-01)

| Konu | Bulgu | Etkisi |
|---|---|---|
| Python ve torch | `torch 2.6.0+cu124` yalnızca Python 3.13'te kurulu, CUDA çalışıyor | Temiz `.venv` torch'u göremez; D-09 bu yüzden gerekli |
| GPU | RTX 3050 Ti Laptop, 4 GB VRAM, güç limiti 45 W | Ollama ve e5 aynı VRAM'i paylaşır; bkz. R-05 |
| GPU telemetrisi | Anlık güç okunuyor (23.06 W); ortalama güç 0 W görünüyor | CodeCarbon'un GPU enerjisini okuyabildiği T-05'te kanıtlanmalı; bkz. R-01 |
| Ollama | Kurulu değil, `localhost:11434` yanıt vermiyor | T-08 öncesi gerekli (U-2) |
| Eksik paketler | `sentence-transformers`, `codecarbon` kurulu değil | T-00'da kuru kurulumla uyum kontrol edilir |
| Starlette 0.38.6 | `formparsers.py:117`: 1 MB üstü yükleme geçici dosyaya taşıyor | CLAUDE.md kural 8'i varsayılan davranışta ihlal ediyor; D-02 |
| CodeCarbon (master) | `country_iso_code` yalnızca `OfflineEmissionsTracker`'da var | SPEC §7.5'teki çağrı hata verir; SC-05 |
| CodeCarbon (master) | `start_task` zamanlayıcıyı durdurur, enerji `stop_task` anında hesaplanır | Kısa aşamalarda dağılım güvenilir olmayabilir; bkz. R-02 |
| Türkçe başlıklar | NFKC + casefold ile denenen 8 başlıktan 6'sı eşleşmedi (`DENEYİM`, `HAKKIMDA` vb.) | Başlık eşleme için ayrı katlama gerekir; T-03 |
| Güç planı | Etkin plan "Güç tasarrufu" | Ölçüm öncesi "Dengeli" seçilir (U-4) |
| CPU ölçümü | Intel Power Gadget kurulu değil; kaynaklara göre Windows'ta CodeCarbon tahmine düşüyor | CPU enerjisinin tahmin olması bekleniyor, T-05 doğrular ve raporda yazılır (SPEC §11) |
| Git | Commit yok, uzak depo boş, yerel dal `master` | İlk commit öncesi dal yeniden adlandırılır |
| CLAUDE.md | `docs/` altında; oturum başında otomatik yüklenmez ve `docs/rules`'a atıf vermiyor | T-00'da kökte içe aktarma dosyası oluşturulur |

### 3.2 Kullanıcının yapacakları

| No | İş | Ne zaman |
|---|---|---|
| U-1 | Repoyu `C:\dev\green-recruitment` yoluna taşımak ve oturumu oradan açmak | T-00'dan önce |
| U-2 | Ollama kurmak, `ollama pull qwen2.5:3b` çalıştırmak | T-08'den önce |
| U-3 | Kaggle veri setini `data/raw/` altına koymak | T-01'den önce |
| U-4 | Ölçümde: "Dengeli" güç planı, prize takılı, uyku kapalı, OneDrive kapalı, diğer uygulamalar kapalı | T-10 ve sonrası her ölçümde |
| U-5 | T-03, T-09, T-10, T-13, T-14 planlarını onaylamak (TASKS bunu şart koşuyor) | İlgili görevde |

## 4. SPEC değişiklik önerileri

CLAUDE.md gereği sapmalar sessizce yapılmaz. Bu değişiklikler T-00'dan önce SPEC ve CLAUDE.md'ye işlenir. "Onaylandı" olanlar soru turlarında kabul edildi, diğerleri bu planın onayıyla kesinleşir.

| No | Yer | Değişiklik | Gerekçe | Durum |
|---|---|---|---|---|
| SC-01 | §5 DecisionRecord, §9.1 | `application_token` (nullable) eklenir. Durum sayfası önce Application'a, sonra DecisionRecord'a bakar; ikisi de yoksa "kayıt silindi" mesajı gösterir | R2 ve R3 Application'ı siliyor, sayfa 404 veriyordu | Uygulandı |
| SC-02 | §9.1, §10 | `MAX_UPLOAD_BYTES=1048576`; aşımda 413 ve Türkçe hata | Starlette 1 MB üstünü diske taşıyor | Uygulandı |
| SC-03 | §6.1 | Yeniden onayda `text_hash` eşleşen bölümlerin kimliği korunur. Değişiklik varsa profilin `ranked` başvuruları yeniden skorlanır, top-k kontrol edilir. Sert filtre yalnızca mevcut ilanda çalışır | `matrix_json` kopuk `section_id` taşıyordu | Uygulandı |
| SC-04 | §9.1 | `GET/POST /employer/login`, `POST /employer/applications/{id}/shortlist`, `POST /employer/jobs/{id}/close`, `POST /employer/jobs/{id}/finalize`, `POST /green/retention` eklenir | Bu işlemlerin yolu tanımlı değildi; `/green` üzerinden yetkisiz silme açığı vardı | Uygulandı |
| SC-05 | §7.5 | `EmissionsTracker` yerine `OfflineEmissionsTracker` | `country_iso_code` diğerinde kabul edilmiyor; ağ üzerinden konum sorgusu da önlenir | Uygulandı |
| SC-06 | §5, §8.2 | `Application.status` değerleri: `draft`, `ranked`, `shortlisted`. `filtered_out` ve `rejected` kaldırılır. `DUPLICATE_APPLICATION` ve `DRAFT_EXPIRED` kayda yazılmayan yanıt kodları olur | Hiçbir akış bu durumlara yazmıyor; R1 kayıt yazmıyor; tekrar denemede kayıt yazmak engel süresini uzatırdı | Uygulandı |
| SC-07 | §6.1 | Aynı ilan için `draft` başvuru varsa `DUPLICATE_APPLICATION` yerine mevcut taslağa yönlendirilir | Yarım kalan taslak adayı 24 saat engellerdi, demoyu da bozabilirdi | Uygulandı |
| SC-08 | §6.3 | `location` ayrıştırılmaz, önizlemede aday girer. Diller küçük bir TR→EN sözlüğüyle eşlenir (`ingilizce` → `english`) | Konum için kural yok; "İngilizce" ile `english` eşleşmiyordu | Uygulandı |
| SC-09 | §5, §7.5 | `energy_wh_gross` ve `duration_s` aşama toplamlarıdır (`model_load` hariç) | Görev modunda enerji yalnızca aşama içinde ölçülüyor; boşta düzeltmesi aynı süreyle yapılmalı | Uygulandı |
| SC-10 | §7.4, §7.7, §10 | `QUALITY_K=10` eklenir; baseline'da `num_predict=120`, aynı `num_ctx` ve `temperature` kullanılır; kalite metriğinde tekrar başvurular CV kimliğine göre tekilleştirilir; aday sayısı k'dan azsa payda aday sayısıdır | Değerler tanımsızdı; iki hattın koşulları eşit olmalı | Uygulandı |
| SC-11 | §7.8, CLAUDE.md | `bench compare` çalıştırmaları aynı `scenario_json` üzerinden gruplar. "last two runs" ifadesi düzeltilir | §7.5 tekrar, §7.3 iki senaryo istiyor; "son iki koşu" yetmiyor | Uygulandı |
| SC-12 | CLAUDE.md Layout | `app/pipeline.py` eklenir (§6.1 akışı); `bench/runner.py` eklenir | Akış hem `bench/green.py` hem web tarafından kullanılıyor ve T-09, T-13'ten önce geliyor | Uygulandı |
| SC-13 | §5 JobPosting | `closes_at` nullable ve yalnızca bilgi amaçlıdır; kapatma elle yapılır | Otomatik kapanma kuralı tanımlı değil | Uygulandı |
| SC-14 | §5 `GreenCounter` | `llm_calls_avoided` = onaylanan başvuru sayısı − LLM çağrısı | Tanımsızdı | Uygulandı |

**Koddan önce gelen dosyalar (görev sırası değişmez):**

| Dosya | TASKS'taki yeri | Yazılacağı görev | Neden |
|---|---|---|---|
| `app/green/counters.py` | Belirtilmemiş | T-06 | T-06 kabul kriteri sayaca bakıyor |
| `app/privacy/reasons.py` | Belirtilmemiş | T-07 | Filtreler neden kodu döndürüyor |
| `app/seed.py` (ilan verisi) | T-12 | T-09 | Benchmark aynı üç ilanı kullanıyor |
| `app/privacy/retention.py` (R2, R3, refcount) | T-14 | T-09 | T-09 metni bunu öngörüyor; R1, R4, R5 T-14'te gelir |

## 5. Mimari ve kurallar

### 5.1 Bağımlılık yönü

```
app/web/*  ──┐
             ├──► app/pipeline.py ──► ingest · scoring · llm · privacy · green
bench/*    ──┘                           │
                                         └─► models · db
```

- İş mantığı web katmanında bulunmaz. Route'lar yalnızca HTTP'yi `pipeline` çağrılarına çevirir, bu sayede benchmark ve demo aynı kodu çalıştırır.
- `app/pipeline.py` iki ana işlem sunar: `submit_upload` ve `confirm_draft`.
- Bağımlılıklar `PipelineContext` ile enjekte edilir: session, ayarlar, embedder, LLM, tracker ve `now`.
- `scoring/matrix.py` saf fonksiyon kalır; veritabanı ve model bilgisi taşımaz.

### 5.2 Soyutlamalar

Kapsamı büyütmemek için yalnızca değiştirilebilirliği gerçekten gereken noktalar soyutlanır:

| Arayüz | Yer | Gerçek uygulama | Test uygulaması |
|---|---|---|---|
| `EmbeddingProvider` | `scoring/embed.py` | sentence-transformers | Sahte vektörler |
| `LlmClient` | `llm/ollama.py` | Ollama (httpx) | Sahte yanıtlar |
| `StageTracker` | `green/tracker.py` | CodeCarbon | No-op |
| `BenchmarkRunner` (ABC) | `bench/runner.py` | `GreenRunner`, `BaselineRunner` | Yok |

Ağır kütüphaneler bu uygulamaların içinde geç yüklenir; testler GPU ve Ollama olmadan çalışır. DTO'lar: `MatrixInput`, `MatrixResult`, `ScenarioConfig`, `UploadInput`.

### 5.3 Kuralların uygulanması

| Kural | Uygulama | Denetim |
|---|---|---|
| SOLID, DI | §5.1 ve §5.2 | Commit öncesi öz kontrol |
| Fonksiyon boyu ve argüman sayısı | 25 satır, en fazla 3 argüman, gerekirse DTO | Commit öncesi öz kontrol |
| Adlandırma | SPEC'teki `R`, `S`, `w`, `α` kodda `requirement_embeddings` gibi açık adlarla yazılır | Commit öncesi öz kontrol |
| Sihirli sayı yok | `config.py` ayarları ve modül sabitleri (`MIN_EXTRACTED_CHARS`, `MAX_SECTION_WORDS`, `PAGE_SIZE`) | Commit öncesi öz kontrol |
| Hata yönetimi | Tipli hatalar (`UnreadablePdfError`, `TextTooShortError`); boş `except` yok | Test ve öz kontrol |
| Docstring | Tüm public sınıf ve fonksiyonlarda | Commit öncesi öz kontrol |
| Onaylı bağımlılık listesi | `pyproject.toml` yalnızca CLAUDE.md listesini içerir; `torch` listelenmez ve sürümü değişmez | T-00 kuru kurulum, öz kontrol |
| Arayüz sadeliği | Framework ve build yok, tek `app.css`, sistem fontu, görsel ve webfont yok, inline JS yalnızca zorunluysa | T-12 ve T-16 sayfa ağırlığı |
| Silme yalnızca retention'da | `session.delete` ve `DELETE` başka yerde bulunmaz | `tests/test_architecture_rules.py` |
| `retention`'da saat enjekte edilir | `datetime.now()` çağrılmaz | `tests/test_architecture_rules.py` |
| Model bir kez yüklenir | `SentenceTransformer(` yalnızca `scoring/embed.py` içinde; lifespan'de çağrılır | `tests/test_architecture_rules.py` |
| Önbellek atlanamaz | Parse `raw_hash`, embedding `text_hash` üzerinden tek yoldan, `app/pipeline.py` içinde yürür | T-06 ve T-13 sayaç testleri |
| Ölçümler `stage()` içinde | Adlar `tracker.py`'daki sabit listeden gelir (SPEC §7.4); bilinmeyen ad hata verir | T-05 testi |
| LLM yalnızca top-k özetinde | Yeşil hatta tek çağrı noktası `pipeline`'dır; `ingest` ve `scoring` modülleri `llm`'i içe aktarmaz | `tests/test_architecture_rules.py` |
| PDF diske yazılmaz | Bellek içi işleme, D-02 | T-13 testi |
| `raw_hash` anahtarlıdır | HMAC-SHA256 ve `HASH_SECRET`; düz SHA-256 kullanılmaz | T-02 testi |
| KVKK logları | Yalnızca id ve `raw_hash` ilk 8 karakteri; `short_hash()` yardımcısı | T-13 testi, §7 notları |
| İletişim verisi gizliliği | Kısa liste dışında işveren yanıtında yok; `profile_json` ve bölüm metinlerinde bulunmaz | T-03, T-13 ve T-15 testleri |
| Sayfalama ve sayfa ağırlığı | Sayfada 20 kayıt, 50 KB bütçe | T-15 ve T-16 |
| Zorunlu birim testleri | `normalize`, `sections`, `matrix`, `filters`, `retention`; LLM ve embedding sahte, GPU ve Ollama gerekmez | T-02, T-03, T-07, T-14 |
| Proje günlüğü | Her görev, önemli karar, ciddi hata ve çözümünde `docs/project_log.md` sonuna İngilizce kayıt | §9 |

`tests/test_architecture_rules.py` kısa bir kaynak taraması testidir, yeni bağımlılık gerektirmez ve T-00'da eklenir. Tüm görevler standartlardaki "Agent Execution Protocol" öz kontrolünden geçer.

## 6. Takvim

Tarihler teslimin 6 Kasım olduğu varsayımıyla verilmiştir.

| Gün | Görevler | Çıkış koşulu |
|---|---|---|
| G0 (1 Kas) | Plan onayı, U-1, U-2, U-3 | Repo taşındı, veri seti ve Ollama hazır |
| G1 (2 Kas) | SPEC değişiklikleri, T-00–T-04 | Ingest ve veri modeli tamam, parse raporu ≥ %80 |
| G2 (3 Kas) | T-05–T-08 | Ölçüm kapısı geçildi (§8.1) |
| G3 (4 Kas) | T-09–T-11 | **K1:** sayılar gösterildi, §8.2 değerlendirildi |
| G4 (5 Kas) | T-12–T-16, gece T-17 ölçümü | **MVP:** SPEC §3 demosu baştan sona çalışıyor |
| G5 (6 Kas) | T-17 raporu, T-18, demo provası | Teslim |

**Sıkışma olursa kesme sırası:**

1. Önce N düşürülür (SPEC §11 izin veriyor, raporda belirtilir).
2. Sonra panel görselliği sadeleştirilir, sayaçlar ve tablo kalır.
3. K1 ölçütleri, KVKK kuralları, iki senaryo, SPEC §3 demosu ve testler kesilmez.

## 7. Görev notları

Kabul kriterleri `docs/TASKS.md`'dedir. Aşağıdakiler ek uygulama notlarıdır.

### Faz 0: İskelet

- **T-00 Repo iskeleti**
  - `.venv` D-09'a göre kurulur:
    ```bash
    python -m venv --system-site-packages .venv
    ```
  - `.venv` içinde torch sürümü ve CUDA kontrol edilir (beklenen: `2.6.0+cu124`, `True`). Kuru kurulum torch'a dokunursa veya `codecarbon` ya da `sentence-transformers` Python 3.13 ile uyuşmazsa durulur ve sorulur.
  - Sürümler ölçüm tekrarlanabilirliği için `pyproject.toml`'da sabitlenir. Kök `CLAUDE.md` yalnızca `@docs/CLAUDE.md` ve iki kural dosyasını içe aktarır.
  - `.gitignore`: `data/`, `.env`, `.venv/`, `__pycache__/`. `.env.example` SC-02 ve SC-10 ayarlarını da içerir.
  - `create_app()` fabrikası testlerde sahte embedder verilmesini sağlar. `tests/test_architecture_rules.py` eklenir.
  - `git branch -m master main`, ilk commit, ardından `develop`.
- **T-01 Veri seti okuyucu** (U-3'e bağlı)
  - Yapı SPEC §7.1'den farklıysa durulur. Aynı görevde tarih biçimleri ve iletişim bilgisi varlığı örneklerle not edilir, çünkü bunlar çevrimiçi kaynaklardan doğrulanamadı ve T-03'ü doğrudan etkiliyor.

### Faz 1: Ingest

- **T-02 PDF, normalizasyon, hash**
  - Çıkarılan metin satır yapısını korur. `normalize` yalnızca hash ve `text_hash` içindir, bölümleme ham metinde çalışır.
  - Sır parametre olarak verilir. Test PDF'leri `tests/` içindeki küçük bir yardımcıyla ham baytlardan üretilir, yeni bağımlılık gerekmez.
- **T-03 Bölüm ayırıcı** (önce plan modu)
  - Başlık eşleme için `fold_heading()` yazılır: NFKC, `İ`/`I`/`ı` birleştirme, U+0307 temizleme, casefold. SPEC §6.2 değişmez, hash etkilenmez.
  - Yıl aralıkları tire dışında `to`, en dash ve ay/yıl biçimlerini de kapsar. Biçimler T-01 bulgusuna göre genişletilir.
  - `parse-report` için ek hedef: CV'lerin en az %60'ında `years_experience` bulunmalı. Aksi halde benchmark'taki `MIN_YEARS` filtresi etkisiz kalır ve K1 yanıltıcı olur.
  - Veri setinde iletişim bilgisi olmayabilir; `contact.extract` sentetik örneklerle test edilir.
- **T-04 Veri modeli**
  - SC-01 ve SC-06 uygulanır. `Section.text_hash` indekslenir.
  - Application → profil yönünde cascade yoktur. Böylece refcount hatası sessiz veri kaybı yerine FK hatası üretir.

### Faz 2: Ölçüm ve skor çekirdeği

- **T-05 Tracker** (ölçüm kapısı, §8.1)
  - SC-05 uygulanır. Kurulu sürümde API adları, yöntem ve `measurement_notes` doğrulanır.
- **T-06 Embedding servisi**
  - `text_hash` araması tüm profiller üzerinde yapılır. Model önceden indirilir; ilk indirme ölçüme girmez.
- **T-07 Sert filtre ve matris**
  - Filtreler SPEC §6.4 gereği SQL'dedir, JSON listeler için `json_each` kullanılır. Dil filtresi "herhangi biri" mantığıyla çalışır.
- **T-08 LLM istemcisi** (U-2'ye bağlı)
  - Her iki hat aynı temel seçenekleri kullanır (SC-10). Baseline JSON çıktısı için Ollama'nın `format: json` seçeneği kullanılır.
  - Özetler arayüzde gösterildiği için prompt Türkçe çıktı ister. `ollama ps` ile modelin %100 GPU'da çalıştığı doğrulanır.

### Faz 3: Benchmark

- **T-09 Yeşil hat** (önce plan modu)
  - `app/pipeline.py`, `app/seed.py` ilan verisi, `bench/runner.py`, `bench/scenario.py` ve `bench/green.py` yazılır. Retention bölümü §4'teki tabloya göre yazılır.
  - CLI seçenekleri: `--jobs-per-cv`, `--resubmit-rate`, `--seed`, `--repeats`. Her koşu yeni veritabanıyla başlar.
- **T-10 Baseline** (önce plan modu)
  - Çağrı başına süre ölçülür ve T-17'deki N bundan hesaplanır. U-4 koşulları burada başlar.
- **T-11 Kalite ve rapor**
  - SC-10 ve SC-11 uygulanır. Rapor gruplanmış ortalama ± sapma üretir.
  - **K1:** sayılar kullanıcıya gösterilir ve §8.2'ye göre değerlendirilir.

### Faz 4: Demo uygulaması

- **T-12 Seed ve işveren sayfaları**
  - §7.2 yalnızca `min_years` veriyor. Demo ilanlarına dil ve konum değerleri eklenir; benchmark'ta bu filtreler kapalı kalır.
  - `/employer/login` eklenir (SC-04). Çerez `HttpOnly` ve `SameSite=Lax`; token `secrets.compare_digest` ile karşılaştırılır.
- **T-13 Aday akışı** (önce plan modu)
  - Gövde akışı sayılarak D-02 uygulanır. Test `MAX_UPLOAD_BYTES ≤ MultiPartParser.max_file_size` koşulunu da doğrular, böylece Starlette güncellemesi kuralı sessizce bozamaz.
  - Limit üstü yüklemenin reddedildiği ve geçici dosya oluşmadığı test edilir, yalnızca küçük PDF ile yetinilmez.
  - Yeniden onayda bölüm metni tekrar temizlenir; aday e-postayı bölüme yazarsa `profile_json`'a girmez.
  - `uvicorn.access` günlüğünde token maskelenir, çünkü `/drafts/{token}` iletişim verisine erişim sağlar.
  - LLM özeti web'de FastAPI `BackgroundTasks` ile çalışır, onay yanıtı Ollama'yı beklemez.
  - Demo için sentetik bir CV PDF'i üretilir; adı, e-postası ve telefonu kurgusaldır.
- **T-14 Retention ve KVKK** (önce plan modu)
  - R1, R4, R5, başlangıç tetikleyicisi ve SC-04 kapatma ile finalize uç noktaları eklenir. `+N gün` yalnızca `now` değerini kaydırır.
- **T-15 Sıralı liste ve matris**
  - Tek aday varsa veya tüm skorlar eşitse min-max sıfıra bölünmez, 100 gösterilir. Kısa liste uç noktası burada eklenir.
- **T-16 Yeşil panel**
  - `pageweight` saf ASGI olarak GZip'in dışına konur, böylece sıkıştırılmış bayt ölçülür. Veriler bellektedir; yeni tablo gerekmez.
  - `bench.db` `mode=ro` ile açılır, dosya yoksa "henüz ölçüm yok" gösterilir. Retention POST'u çerez ister.

### Faz 5: Final ölçüm ve teslim

- **T-17 Final benchmark**
  - Kod dondurulur ve commit özeti `measurement_notes`'a yazılır. Boşta ölçümü Ollama ve e5 belleğe yüklüyken yapılır, böylece sabit VRAM tüketimi işe atfedilmez.
  - N şu formülle seçilir, baseline bütçesi bir gecede 6 saattir:
    - `N = min(300, 2250 / çağrı_başına_saniye)`
    - Varsayım: iki senaryo ve üç tekrarda N başına yaklaşık 9.6 LLM çağrısı. Örnek: 5 sn/çağrıda N=300, 15 sn/çağrıda N=150.
- **T-18 README**
  - TASKS'taki sınırlamalara ek olarak şunlar yazılır: Windows'ta CPU enerjisinin tahmin olması, kısa aşama dağılımının yaklaşık olması, aynı PDF'e sahip kişinin onaylı profil verisini önizlemede görebilmesi.

## 8. Kapılar

### 8.1 Ölçüm kapısı (T-05 sonu)

T-09'a geçmeden önce üçü de sağlanmalı. Biri başarısızsa durulur, bulgular ve seçenekler kullanıcıya sunulur.

| Kontrol | Yöntem | Geçme koşulu |
|---|---|---|
| GPU enerjisi | Gerçek GPU yükü altında CodeCarbon sonucu ile `nvidia-smi power.draw` örneklerinin karşılaştırılması | GPU enerjisi sıfırdan büyük ve aynı büyüklük mertebesinde |
| Tracker maliyeti | 200 boş `stage()` çağrısının süresi | Tahmini maliyet koşu süresinin %5'inden az |
| CPU yöntemi | CodeCarbon günlüğü ve `measurement_notes` | Yöntem kayıtlı; gerçekten tahminse raporda belirtildi |

Maliyet yüksek çıkarsa aşamaları topluca ölçmek gerekebilir, ancak bu §6.1'deki artımlı top-k akışını değiştirir. Bu nedenle sormadan uygulanmaz.

### 8.2 K1 değerlendirmesi (T-11 sonu, N=50)

Ölçütler sayılar görülmeden önce sabitlenmiştir.

| Ölçüt | Koşul |
|---|---|
| Başvuru başına net enerji | Yeşil < baseline |
| LLM çağrısı | Yeşil < baseline |
| Retention sonrası depolama | Yeşil < baseline |
| Kalite | Yeşil `precision@10` ≥ baseline `precision@10` − 0.10 (D-07) |

Herhangi biri sağlanmazsa neden bulunmadan T-12'ye geçilmez.

**Yorum notu:** N=50'de LLM farkı final ölçüme göre daha küçük görünebilir. Top-k artımlı doluyor ve ilk başvurularda sık değişiyor. Rastgele sırada kabaca `k·(1 + ln(n/k))` özet beklenir: N=50 için toplam yaklaşık 43 çağrıya karşılık baseline'da yaklaşık 110 çağrı. Bu sadece beklentidir, raporda ölçülen değer kullanılır.

## 9. Görev bitirme akışı

Her görev için:

1. `docs/TASKS.md`'deki ilk açık görev alınır; büyük görevse önce plan onaylatılır.
2. Kod yazılır, her dosya yazımından önce standartlardaki üç öz kontrol sorusu uygulanır.
3. `pytest -q` ve görevin kabul kriterleri çalıştırılır.
4. `docs/TASKS.md`'de kutu işaretlenir.
5. `docs/project_log.md` sonuna şablona uygun İngilizce kayıt eklenir. Dosyanın üzerine yazılmaz, CV verisi veya kişisel veri yazılmaz.
6. `develop` dalına commit atılır; mesajda kabul kriterlerinin nasıl doğrulandığı yazılır.

Push ve `main`'e birleştirme yalnızca kullanıcı istediğinde yapılır. Doğal birleştirme noktaları K1 ve MVP'dir.

## 10. Riskler

| No | Risk | Etki | Önlem |
|---|---|---|---|
| R-01 | CodeCarbon bu dizüstü GPU'sunda enerji okuyamaz | LLM enerjisi görünmez, iddia ölçülemez | §8.1 kapısı; çözüm gerekirse kullanıcıya sorulur |
| R-02 | Kısa aşamaların enerji dağılımı güvenilmez, tracker maliyeti yeşil hattı daha çok etkiler | Aşama tablosu yanıltır | §8.1 kapısı; toplam enerji esas alınır, kısa aşamalar raporda yaklaşık işaretlenir |
| R-03 | CPU enerjisi tahminidir | Mutlak değerler kesin değildir | `measurement_notes` ve raporda açıkça belirtilir |
| R-04 | `years_experience` düşük bulunur | `MIN_YEARS` filtresi etkisiz kalır, K1 yanıltır | T-01 bulgusu ve T-03 %60 hedefi |
| R-05 | 4 GB VRAM'e LLM ve e5 sığmaz, LLM kısmen CPU'ya kayar | Süre ve enerji profili değişir | `ollama ps` kontrolü; durum `measurement_notes`'a yazılır |
| R-06 | Baseline süresi gece penceresini aşar | Final ölçüm yetişmez | T-10'daki süre ölçümü ve T-17 formülü |
| R-07 | G4 yoğun (beş arayüz görevi) | MVP gecikir | §6 kesme sırası; T-17 için G5 sabahı tampon |
| R-08 | Paket uyumsuzluğu torch'u değiştirmeyi gerektirir | CLAUDE.md kural 2 ihlali | T-00 kuru kurulum; torch değişikliği yerine durup sorulur |
| R-09 | Mutlak fark küçük çıkar | Sunum etkisi zayıflar | SPEC §11'deki yıllık 1 milyon başvuru projeksiyonu, tahmin olarak etiketlenir |

## 11. İsteğe bağlı öneriler (kapsam dışı)

Bunlar SPEC'e dahil değildir ve varsayılan olarak yapılmaz:

- **Emniyet koşusu (kod gerektirmez):** K1 geçilirse G3 gecesi aynı komutlarla N=150 koşulur. G4 uzarsa elde güncel sonuç olur. Pipeline kodu sonradan değişirse bu sayılar kullanılmaz.
- **Demo provası (kod gerektirmez):** Sunumdan önce 3-4 veri seti PDF'i yüklenir. Sıralı liste tek adayla zayıf görünür ve göreli skor anlamsız olur.
- **Bonuslar:** B-01, B-02 ve B-03 ancak T-18 bittikten sonra zaman kalırsa ele alınır. Bu plan için zaman ayrılmamıştır.
