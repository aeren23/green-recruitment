# T-10 Planı: Baseline Koşucusu

- **Durum:** Onaylandı (T10-1 ile T10-5 kararları dahil)
- **Girdiler:** SPEC §7.4–§7.6; T-09 koşucusu (`bench/runner.py`); ölçüm kararları D1–D6

## 1. Kapsam

`bench/baseline.py`, SPEC §7.4'teki geleneksel hattı uygular. Kabul kriterleri:

- `bench run --pipeline baseline --n 50` çalışır.
- Toplam süre konsola yazılır.
- N=300 için süre tahmini gösterilir.

## 2. Adalet ilkesi

Karşılaştırmanın geçerli olması için iki hat yalnızca hattın kendisinde farklılaşmalıdır. Bu yüzden baseline, T-09'daki ortak koşucuyu (`BenchmarkRunner`) kullanır:

- Aynı senaryo ve tohum: aynı CV'ler, aynı ilanlar, aynı tekrar başvurular, aynı sıra.
- Aynı ölçüm altyapısı: tracker, CUDA bağlamı (D2), GPU güç akışı kontrolü (D4), boşta güç, `measurement_notes`.
- Aynı LLM ayarları: `temperature=0`, `num_ctx=4096`, `num_predict=120` (SC-10).

## 3. Akış (her başvuru için, önbellek yok)

| Aşama | İş |
|---|---|
| `store_pdf` | PDF, `data/bench/baseline_store/<run_id>/` altına kopyalanır |
| `extract` | Metin her seferinde yeniden çıkarılır |
| `llm_score` | Tam metnin ilk `MAX_CV_CHARS` karakteri ve gereksinim listesi LLM'e gider; çıktı `{"score": 0-100, "reason": ...}`. Bozuk yanıt bir kez yeniden denenir, yine bozuksa skor boş kalır (T-08) |
| `rank` | Koşu sonunda her ilan, LLM skoruna göre sıralanır. Skoru boş olanlar en sona konur |

- Tekrar başvurular da işlenir, hiçbir şey silinmez, sert filtre yoktur.
- `model_load` aşaması yalnızca LLM ısınmasını kapsar; baseline embedding modeli kullanmaz.
- Metinler ve LLM çıktıları aynı klasörde bir JSON dosyasına yazılır. Bu SPEC §7.6'nın tarif ettiği geleneksel depolamadır. Dosya `data/` altında kalır ve git'e girmez.

## 4. Depolama ve sıralama

- `storage_bytes_active`: Mağaza klasörünün toplam boyutu, PDF'ler ve JSON dahil.
- `storage_bytes_after_retention`: `storage_bytes_active` ile aynı değer, çünkü baseline hiçbir şey silmez.
- Sıralama dosyası T9-1 ile aynı biçimde `data/bench/rankings/<run_id>.json` olarak yazılır. Tekrar başvurular listede kalır; T-11 kalite hesabında tekilleştirir (SC-10).

## 5. Onay istenen kararlar

| No | Konu | Öneri | Gerekçe |
|---|---|---|---|
| T10-1 | Ollama ulaşılamazsa | Koşu hatayla durur ve geçersiz sayılır | LLM olmadan baseline anlamsız; T-08'deki kararla tutarlı |
| T10-2 | Eski mağaza klasörleri | Her koşu kendi klasörünü kullanır; önceki koşuların klasörleri yeni koşunun başında silinir | Disk birikmez, her koşunun depolaması ayrı ölçülür |
| T10-3 | `rank` aşaması | İlan başına bir kez, koşu sonunda | Sıralama bir kez yapılan bir iştir; başvuru başına sıralamak baseline'ı yapay olarak pahalılaştırırdı |
| T10-4 | N=300 tahmini | Başvuru başına ortalama aşama süresi × N=300 senaryosunun başvuru sayısı, iki senaryo ve üç tekrar için ayrı ayrı yazılır | T-17 takvimi bu tahmine dayanıyor |
| T10-5 | Baseline'a giden metin | İletişim bilgisi temizlenmeden tam metin gider | SPEC "CV'nin tam metni" diyor ve geleneksel hat böyle çalışır. Model yerel çalıştığı için veri makineden çıkmaz |

## 6. Testler

GPU ve Ollama gerektirmez.

- Uçtan uca: sahte LLM ve sahte tracker ile küçük sentetik veri ağacında koşu.
- Ön bellek kullanılmadığı: her başvuruda `extract` ve `llm_score` çalışır, tekrar başvurular dahil.
- Skoru boş kalanlar sıralamada sonda.
- Depolamanın iki değeri eşit.
- Sıralama dosyası yalnızca kimlik ve skor içerir.
- Mağaza klasörü temizliği (T10-2).
- Ollama yokken koşu açık bir hatayla durur (T10-1).
- N=300 tahmininin aritmetiği.
- Yeşil koşucuyla aynı senaryonun aynı başvuru listesini ürettiği.

## 7. Gerçek koşu ve K1'e geçiş

1. `bench run --pipeline baseline --n 50`. Ollama ile, şu anki koşullarda işlevsel bir koşu olarak yapılır.
2. **K1 için sizden gerekenler:** güç planı "Dengeli", bilgisayar prize takılı, OneDrive kapalı, diğer uygulamalar kapalı.
3. Ardından T-11'de, aynı koşullarda boşta güç (120 s, birkaç tekrar) ve iki hattın N=50 koşuları alınır.
