# Green Hiring Pipeline: Benchmark Raporu

Bu rapor `python -m bench compare` ile üretildi. Değerler `data/bench/bench.db` içindeki koşulardan gelir; tablolar ortalama ± örnek standart sapma verir (tek koşu: `± n/a`). Fark yüzdesi = (green − baseline) / baseline × 100.

Kalite kesim noktası k = 10.

## Senaryo grubu 1

- scenario_json: `{"jobs_per_cv":2,"models":{"embedding":"intfloat/multilingual-e5-small","llm":"qwen2.5:3b"},"n":50,"resubmit_rate":0.1,"seed":42,"top_k":5}`
- green koşuları: [1, 3, 5], baseline koşuları: [2, 4, 6]

### Ana tablo

| metric | baseline | green | difference % |
|---|---|---|---|
| `applications` (count) | 111.0 ± 0.0 | 111.0 ± 0.0 | +0.0% |
| `energy_wh_gross` (Wh) | 4.5188 ± 0.2818 | 3.0132 ± 0.5484 | -33.3% |
| `energy_wh_net` (Wh) | 1.8280 ± 0.2135 | 1.2464 ± 0.0712 | -31.8% |
| `co2_g` (g) | 2.0994 ± 0.1309 | 1.3999 ± 0.2548 | -33.3% |
| `duration_s` (s) | 248.06 ± 7.18 | 162.88 ± 44.04 | -34.3% |
| `energy_wh_gross_per_application` (Wh) | 0.04071 ± 0.00254 | 0.02715 ± 0.00494 | -33.3% |
| `energy_wh_net_per_application` (Wh) | 0.01647 ± 0.00192 | 0.01123 ± 0.00064 | -31.8% |
| `sci_g_per_application` (g, M excluded) | 0.01891 ± 0.00118 | 0.01261 ± 0.00230 | -33.3% |
| `storage_bytes_active` (B) | 3777982 ± 217 | 2311509 ± 2365 | -38.8% |
| `storage_bytes_after_retention` (B) | 3777982 ± 217 | 147456 ± 0 | -96.1% |
| `llm_calls` (count) | 111.0 ± 0.0 | 44.0 ± 0.0 | -60.4% |
| `llm_prompt_tokens` (count) | 137651.0 ± 0.0 | 21651.0 ± 0.0 | -84.3% |
| `llm_completion_tokens` (count) | 6220.3 ± 34.1 | 4512.3 ± 152.9 | -27.5% |
| `parse_cache_hits` (count) | 0.0 ± 0.0 | 48.0 ± 0.0 | n/a |
| `embed_cache_hits` (count) | 0.0 ± 0.0 | 490.0 ± 0.0 | n/a |
| `precision_at_k` (share) | 0.500 ± 0.000 | 0.600 ± 0.000 | +20.0% |
| `overlap_at_k` (share) | 0.500 ± 0.000 | 0.500 ± 0.000 | +0.0% |

### Aşama bazlı enerji dağılımı

#### baseline

| stage | energy (mWh) | duration (s) | count | ms / call | note |
|---|---|---|---|---|---|
| extract | 416.176 | 29.97 | 111.0 | 270.0 |  |
| llm_score | 4096.998 | 217.52 | 111.0 | 1959.7 |  |
| model_load | 28.462 | 2.06 | 1.0 | 2056.6 | toplama dahil değil |
| rank | 0.013 | 0.01 | 3.0 | 3.6 | yaklaşık (< 100 ms/çağrı) |
| store_pdf | 5.633 | 0.56 | 111.0 | 5.0 | yaklaşık (< 100 ms/çağrı) |

#### green

| stage | energy (mWh) | duration (s) | count | ms / call | note |
|---|---|---|---|---|---|
| embed | 84.255 | 5.46 | 96.0 | 56.9 | yaklaşık (< 100 ms/çağrı) |
| extract | 451.604 | 32.21 | 111.0 | 290.2 |  |
| filter | 4.392 | 0.60 | 100.0 | 6.0 | yaklaşık (< 100 ms/çağrı) |
| hash | 6.340 | 0.92 | 111.0 | 8.3 | yaklaşık (< 100 ms/çağrı) |
| llm_summary | 2445.605 | 121.74 | 44.0 | 2766.9 |  |
| model_load | 420.395 | 34.56 | 1.0 | 34558.8 | toplama dahil değil |
| parse | 13.114 | 1.16 | 52.0 | 22.3 | yaklaşık (< 100 ms/çağrı) |
| retention | 4.516 | 0.34 | 5.0 | 68.4 | yaklaşık (< 100 ms/çağrı) |
| score | 3.379 | 0.45 | 96.0 | 4.7 | yaklaşık (< 100 ms/çağrı) |

### K1 değerlendirmesi

| metric | rule | baseline | green | threshold | result |
|---|---|---|---|---|---|
| `energy_wh_net_per_application` | net energy per application: green < baseline | 0.01647 | 0.01123 | n/a | PASS |
| `llm_calls` | LLM calls: green < baseline | 111.0 | 44.0 | n/a | PASS |
| `storage_bytes_after_retention` | storage after retention: green < baseline | 3777982 | 147456 | n/a | PASS |
| `precision_at_k` | quality (D-07): green precision@k >= baseline precision@k - 0.10 | 0.500 | 0.600 | 0.400 | PASS |

K1 sonucu (dört ölçütün hepsi): **PASS**

### Yılda 1 milyon başvuru projeksiyonu (tahmin)

Başvuru sayısı: 1,000,000 / yıl.

| yearly total | baseline | green | difference % |
|---|---|---|---|
| net energy (kWh) | 16.47 | 11.23 | -31.8% |
| gross energy (kWh) | 40.71 | 27.15 | -33.3% |
| sci (kg CO₂e, M excluded) | 18.91 | 12.61 | -33.3% |

Bu, bu dizüstünün donanımında ve bu senaryoda başvuru başına ortalamaların doğrusal ekstrapolasyonudur. Tasarruf, ilan başına başvuru sayısına bağlıdır (LLM yalnızca ilk k adaylar için çağrılır); bu yüzden bir tahmindir, ölçüm değildir.

### Koşullar ve uyarılar

Tüm koşular nihai ölçüm koşullarına uyuyor (Dengeli plan, OneDrive kapalı, git temiz).

| run | pipeline | power plan | OneDrive running | git | git dirty | idle_power_w | GPU stage-sum / stream (D4) | unstaged share | extract ms/call |
|---|---|---|---|---|---|---|---|---|---|
| 1 | green | Dengeli | false | 9cad32c | false | 39.05 | 0.944 | 0.019 | 259.9 |
| 2 | baseline | Dengeli | false | 9cad32c | false | 39.05 | 0.994 | 0.000 | 263.4 |
| 3 | green | Dengeli | false | 9cad32c | false | 39.05 | 0.969 | 0.011 | 292.0 |
| 4 | baseline | Dengeli | false | 9cad32c | false | 39.05 | 0.993 | 0.000 | 268.5 |
| 5 | green | Dengeli | false | 9cad32c | false | 39.05 | 0.971 | 0.010 | 318.7 |
| 6 | baseline | Dengeli | false | 9cad32c | false | 39.05 | 0.992 | 0.000 | 278.0 |

- CPU ölçüm yöntemi (notlardan): CPU(Windows Emi)
- extract ms/call: baseline 270.0, green 290.2, fark (green − baseline) 20.2
- `extract` çağrı başına maliyeti green'de baseline'dan yüksek: izleyici ve NVML ek yükü (D6) green'i cezalandırır, yani bu aşamada karşılaştırma green aleyhine temkinlidir.

## Senaryo grubu 2

- scenario_json: `{"jobs_per_cv":1,"models":{"embedding":"intfloat/multilingual-e5-small","llm":"qwen2.5:3b"},"n":300,"resubmit_rate":0.0,"seed":42,"top_k":5}`
- green koşuları: [7, 9, 11], baseline koşuları: [8, 10, 12]

### Ana tablo

| metric | baseline | green | difference % |
|---|---|---|---|
| `applications` (count) | 300.0 ± 0.0 | 300.0 ± 0.0 | +0.0% |
| `energy_wh_gross` (Wh) | 25.0751 ± 10.7478 | 8.3396 ± 2.3783 | -66.7% |
| `energy_wh_net` (Wh) | 6.6912 ± 0.3970 | 2.1603 ± 0.1301 | -67.7% |
| `co2_g` (g) | 11.6495 ± 4.9933 | 3.8745 ± 1.1049 | -66.7% |
| `duration_s` (s) | 1610.38 ± 911.94 | 541.29 ± 219.18 | -66.4% |
| `energy_wh_gross_per_application` (Wh) | 0.08358 ± 0.03583 | 0.02780 ± 0.00793 | -66.7% |
| `energy_wh_net_per_application` (Wh) | 0.02230 ± 0.00132 | 0.00720 ± 0.00043 | -67.7% |
| `sci_g_per_application` (g, M excluded) | 0.03883 ± 0.01664 | 0.01291 ± 0.00368 | -66.7% |
| `storage_bytes_active` (B) | 9657837 ± 0 | 12169216 ± 0 | +26.0% |
| `storage_bytes_after_retention` (B) | 9657837 ± 0 | 212992 ± 0 | -97.8% |
| `llm_calls` (count) | 300.0 ± 0.0 | 60.0 ± 0.0 | -80.0% |
| `llm_prompt_tokens` (count) | 358587.0 ± 0.0 | 32041.0 ± 0.0 | -91.1% |
| `llm_completion_tokens` (count) | 16787.0 ± 0.0 | 6214.3 ± 13.3 | -63.0% |
| `parse_cache_hits` (count) | 0.0 ± 0.0 | 0.0 ± 0.0 | n/a |
| `embed_cache_hits` (count) | 0.0 ± 0.0 | 21.0 ± 0.0 | n/a |
| `precision_at_k` (share) | 0.867 ± 0.000 | 0.900 ± 0.000 | +3.8% |
| `overlap_at_k` (share) | 0.467 ± 0.000 | 0.467 ± 0.000 | +0.0% |

### Aşama bazlı enerji dağılımı

#### baseline

| stage | energy (mWh) | duration (s) | count | ms / call | note |
|---|---|---|---|---|---|
| extract | 2247.512 | 179.11 | 300.0 | 597.0 |  |
| llm_score | 22802.972 | 1427.85 | 300.0 | 4759.5 |  |
| model_load | 29.529 | 2.06 | 1.0 | 2057.1 | toplama dahil değil |
| rank | 0.041 | 0.02 | 3.0 | 5.3 | yaklaşık (< 100 ms/çağrı) |
| store_pdf | 24.551 | 3.41 | 300.0 | 11.4 | yaklaşık (< 100 ms/çağrı) |

#### green

| stage | energy (mWh) | duration (s) | count | ms / call | note |
|---|---|---|---|---|---|
| embed | 536.659 | 38.81 | 291.0 | 133.4 |  |
| extract | 1696.536 | 128.34 | 300.0 | 427.8 |  |
| filter | 17.169 | 2.32 | 300.0 | 7.7 | yaklaşık (< 100 ms/çağrı) |
| hash | 23.705 | 3.33 | 300.0 | 11.1 | yaklaşık (< 100 ms/çağrı) |
| llm_summary | 5939.945 | 356.10 | 60.0 | 5935.0 |  |
| model_load | 593.630 | 50.64 | 1.0 | 50635.6 | toplama dahil değil |
| parse | 93.411 | 9.03 | 300.0 | 30.1 | yaklaşık (< 100 ms/çağrı) |
| retention | 18.113 | 1.47 | 10.0 | 146.7 |  |
| score | 14.083 | 1.90 | 291.0 | 6.5 | yaklaşık (< 100 ms/çağrı) |

### K1 değerlendirmesi

| metric | rule | baseline | green | threshold | result |
|---|---|---|---|---|---|
| `energy_wh_net_per_application` | net energy per application: green < baseline | 0.02230 | 0.00720 | n/a | PASS |
| `llm_calls` | LLM calls: green < baseline | 300.0 | 60.0 | n/a | PASS |
| `storage_bytes_after_retention` | storage after retention: green < baseline | 9657837 | 212992 | n/a | PASS |
| `precision_at_k` | quality (D-07): green precision@k >= baseline precision@k - 0.10 | 0.867 | 0.900 | 0.767 | PASS |

K1 sonucu (dört ölçütün hepsi): **PASS**

### Yılda 1 milyon başvuru projeksiyonu (tahmin)

Başvuru sayısı: 1,000,000 / yıl.

| yearly total | baseline | green | difference % |
|---|---|---|---|
| net energy (kWh) | 22.30 | 7.20 | -67.7% |
| gross energy (kWh) | 83.58 | 27.80 | -66.7% |
| sci (kg CO₂e, M excluded) | 38.83 | 12.91 | -66.7% |

Bu, bu dizüstünün donanımında ve bu senaryoda başvuru başına ortalamaların doğrusal ekstrapolasyonudur. Tasarruf, ilan başına başvuru sayısına bağlıdır (LLM yalnızca ilk k adaylar için çağrılır); bu yüzden bir tahmindir, ölçüm değildir.

### Koşullar ve uyarılar

Tüm koşular nihai ölçüm koşullarına uyuyor (Dengeli plan, OneDrive kapalı, git temiz).

| run | pipeline | power plan | OneDrive running | git | git dirty | idle_power_w | GPU stage-sum / stream (D4) | unstaged share | extract ms/call |
|---|---|---|---|---|---|---|---|---|---|
| 7 | green | Dengeli | false | 6647744 | false | 41.10 | 0.940 | 0.027 | 430.0 |
| 8 | baseline | Dengeli | false | 6647744 | false | 41.10 | 0.993 | 0.000 | 645.6 |
| 9 | green | Dengeli | false | 6647744 | false | 41.10 | 0.956 | 0.018 | 595.1 |
| 10 | baseline | Dengeli | false | 6647744 | false | 41.10 | 0.992 | 0.001 | 867.7 |
| 11 | green | Dengeli | false | 6647744 | false | 41.10 | 0.930 | 0.029 | 258.3 |
| 12 | baseline | Dengeli | false | 6647744 | false | 41.10 | 0.987 | 0.001 | 277.9 |

- CPU ölçüm yöntemi (notlardan): CPU(Windows Emi)
- extract ms/call: baseline 597.0, green 427.8, fark (green − baseline) -169.2
- `extract` çağrı başına maliyeti green'de baseline'dan yüksek değil: bu aşamada green aleyhine bir sapma görülmüyor.

## Dışarıda bırakılan koşular

Yok.

## Kalite notları

Dışlanan ilan veya eşleşmemiş koşu yok.

## Sabit notlar

- CPU enerjisi bir donanım sayacından gelir (Windows Energy Meter Interface, RAPL sayacı); TDP tahmini değildir. Paket sayacı ekran, SSD ve Wi-Fi gibi bileşenleri kapsamaz.
- RAM enerjisi sabit güç modelidir (yüke bağlı değildir); ölçülmüş bir değer değildir.
- SCI = co2_g / applications. Gömülü karbon (M) ölçüm dışıdır ve SCI'ya dahil edilmemiştir. R, koşuya gönderilen tüm başvurulardır (engellenen tekrarlar dahil).
- Benchmark'ta dil ve konum filtreleri kapalıdır; sert filtre yalnızca `min_years` üzerinde çalışır (SPEC 7.2). CV'lerin ayrıştırılmasında sabit referans tarihi kullanılır (SPEC 7.3).
- Baseline küçük bir 3B modelle puanlanır; bu yüzden `overlap_at_k` yardımcıdır, ana kalite ölçütü `precision_at_k` dir (SPEC 7.7).
- `model_load` aşaması toplamlara dahil değildir, ayrı satırda gösterilir.
- 100 ms'den kısa aşamaların (çağrı başına ortalama) aşama bazlı enerjisi yaklaşıktır (D1); toplam enerji esas alınmalıdır.
