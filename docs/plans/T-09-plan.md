# T-09 Planı: Yeşil Hat Koşucusu

- **Durum:** Onaylandı (T9-1 ile T9-4 kararları dahil)
- **Girdiler:** SPEC §6.1, §7.3–§7.6, §8.3; `docs/plans/measurement-gate.md` kararları D1–D5; T-03 ve T-06 bulguları

## 1. Kapsam

TASKS T-09: `bench/scenario.py`, `bench/green.py` ve bunların dayandığı ortak modüller. Kabul kriteri, `bench run --pipeline green --n 50` komutunun çalışması ve `BenchmarkRun` ile `StageMeasurement` kayıtlarının yazılmasıdır. Retention bu görevde asgari düzeyde yazılır, T-14'te tamamlanır.

Ollama kurulu değilse koşu yine tamamlanır; SPEC §6.6 gereği özet alanı boş kalır. Ancak bu durumda koşu `measurement_notes` içinde `llm_unavailable` olarak işaretlenir ve K1 için geçersiz sayılır.

## 2. Modüller

| Dosya | İçerik |
|---|---|
| `app/pipeline.py` | SPEC §6.1 akışı: `submit_upload` ve `confirm_draft`. Bağımlılıklar `PipelineContext` ile gelir: session, settings, embedder, LLM, tracker, `now`, `LlmUsage`. Web (T-13) aynı fonksiyonları kullanır |
| `app/privacy/retention.py` | Asgari sürüm: R2 (sert filtreyle eleme), R3 (ilan kapanınca `NOT_SELECTED`), profil refcount silmesi. Saat `now` parametresiyle gelir, `application_token` kopyalanır. R1, R4, R5 T-14'te eklenir |
| `app/seed.py` | SPEC §7.2'deki üç ilanın verisi ve `seed_postings`. Benchmark modunda dil ve konum filtreleri kapalıdır; demo değerleri T-12'de eklenir |
| `bench/scenario.py` | `ScenarioConfig` ve deterministik başvuru listesi |
| `bench/runner.py` | `BenchmarkRunner` ABC: hazırlık, koşu, kapanış, sonuçların yazılması |
| `bench/green.py` | `GreenRunner` |
| `bench/__main__.py` | `run` alt komutu: `--pipeline`, `--n`, `--jobs-per-cv`, `--resubmit-rate`, `--seed`, `--repeats` |

## 3. Akış

Aşamalar SPEC §7.4'teki adlarla `stage()` içinde çalışır.

1. **`extract`:** PDF baytları bellekte okunur.
2. **`hash`:** Normalizasyon ve HMAC.
3. **Tekrar kontrolü:** Aynı ilan ve `raw_hash` için başvuru veya süresi dolmamış karar kaydı varsa `DUPLICATE_APPLICATION` döner; mevcut başvuru taslaksa ona yönlendirilir.
4. **`parse`:** Yalnızca profil önbellekte yoksa çalışır. İletişim bilgisi ayrı tabloya yazılır.
5. **Onay:** Benchmark'ta otomatik ve düzeltmesizdir (SPEC §7.3). Yeniden onayda SC-03 kuralı uygulanır.
6. **`filter`:** Başarısızsa R2 uygulanır.
7. **`embed`:** Yalnızca önbellekte olmayan bölümler embed edilir.
8. **`score`:** Matris skoru hesaplanır; `matrix_json` kanıt ve beyan işaretiyle yazılır.
9. **`llm_summary`:** İlanın top-k listesine giren yeni adaylar için, yalnızca matriste seçilen bölümlerle özet üretilir.
10. **`retention`:** Koşu sonunda tüm ilanlar kapatılır, R3 uygulanır, ardından `VACUUM` çalışır.

Her başvuru kendi transaction'ında commit edilir, çünkü web akışı da istek başına commit eder.

## 4. Ölçüm

- **Ölçülen değerler:** Brüt enerji ve süre aşama toplamlarıdır, `model_load` hariç (SC-09). Net enerji için `data/bench/idle_power.json` kullanılır; dosya yoksa koşu açık bir hatayla durur. CO₂, CodeCarbon'un her görev için verdiği emisyon değerinden toplanır; tracker'a küçük bir ekleme gerekir.
- **D2, adil ölçüm:** Koşucu, ölçüm süresince süreçte bir CUDA bağlamı tutar. Baseline da aynısını yapar.
- **D4, GPU çapraz kontrolü:** Koşunun başında ve sonunda GPU enerji sayacı okunur, aşamalardan toplanan GPU enerjisiyle karşılaştırılıp `measurement_notes`'a yazılır. Yeni bağımlılık eklenmez. Sayaç `nvidia-smi` ile okunamazsa durulur ve sorulur.
- **Depolama (SPEC §7.6):** `storage_bytes_active` ilanlar kapanmadan önce, `storage_bytes_after_retention` ise retention ve `VACUUM` sonrasında ölçülür.
- **`measurement_notes`:** Tracker notlarına ek olarak git commit özeti, güç planı, D4 sonucu ve Ollama durumu yazılır.

## 5. Onay istenen kararlar

| No | Konu | Öneri | Gerekçe |
|---|---|---|---|
| T9-1 | Sıralama kaydı | Her koşuda, retention'dan önce ilan başına nihai sıralama veri seti kimlikleriyle `data/bench/rankings/<run_id>.json` dosyasına yazılır. Yeni tablo eklenmez | Retention sıralanmış başvuruları siliyor; aksi halde T-11'deki `precision@k` hesaplanamaz. SPEC bu noktada boş |
| T9-2 | Tekrar başvuru sayısı | `resubmit_rate` toplam başvuruların payı olarak yorumlanır: tekrar sayısı = taban × r / (1 − r). Tekrar, orijinalinden sonra deterministik bir sıraya yerleştirilir | SPEC "this share of applications" diyor; bu, tanıma uyan yorum |
| T9-3 | `llm_calls_avoided` | Koşu sonunda hesaplanır: onaylanan başvurular − LLM çağrıları (SC-14) | Başvuru başına artırılıp azaltılan bir sayaç anlamsız ara değerler üretir |
| T9-4 | Ollama yokken | Koşu tamamlanır, `llm_unavailable` olarak işaretlenir, `bench compare` bu koşuları K1 dışında tutar | Ölçüm olmadan iddia kanıtlanamaz; ama kod ve akış yine test edilebilir |

## 6. Testler

GPU ve Ollama gerektirmez: sahte embedder, sahte LLM ve no-op tracker kullanılır.

- **Pipeline:** Yeni profil, önbellek isabeti (parse atlanır ve sayaç artar), tekrar başvurunun engellenmesi, taslağa yönlendirme, filtreyle eleme (R2 ve karar kaydı), top-k değişiminde yalnızca yeni girenler için özet, Ollama yokken özetin boş kalması.
- **Retention:** R2, R3 ve iki ilana başvurmuş bir profilin bir ilandan elenince silinmemesi.
- **Senaryo:** Aynı tohum aynı listeyi verir; `jobs_per_cv` ve `resubmit_rate` oranları doğru.
- **Koşucu:** Küçük sentetik veri ağacıyla uçtan uca çalışma, iki depolama noktası, sıralama dosyası, `BenchmarkRun` ve `StageMeasurement` kayıtları.
- **Mimari kurallar:** Silme yalnızca retention'da yapılır; LLM yalnızca `llm_summary` aşamasında çağrılır.

## 7. Uygulama sırası

| Adım | Görev | Alt ajan | Neden |
|---|---|---|---|
| A | `retention.py` (asgari), `seed.py`, `pipeline.py` ve testleri | 1 | Akışın kendisi |
| B | `scenario.py`, `runner.py`, `green.py`, CLI ve testleri | 1, A'dan sonra | A'ya bağlı |
| C | Gerçek koşu: `bench idle`, ardından `bench run --pipeline green --n 50` | Kendim değerlendiririm | Ölçüm |

C adımı Ollama kurulu değilse `llm_unavailable` olarak çalışır. Asıl K1 ölçümü için Ollama kurulumu (U-2), "Dengeli" güç planı ve OneDrive'ın kapatılması (U-4) gerekir.
