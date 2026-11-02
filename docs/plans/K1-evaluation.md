# K1 Değerlendirmesi (T-11 sonu, N=50)

- **Tarih:** 2026-11-02
- **Kod:** `9cad32c`, git temiz
- **Ortam:** Dengeli güç planı, prizde, OneDrive istemcisi kapalı, arka plan yükü %1'in altında
- **Senaryo:** N=50, seed 42, `jobs_per_cv=2`, `resubmit_rate=0.1`; 111 başvuru (11'i tekrar)
- **Koşu düzeni:** Yeşil ve baseline dönüşümlü, üçer koşu (run 1–6). Boşta güç dört kez ölçüldü: koşulardan önce üç kez, sonra bir kez.
- **Kaynak:** `data/bench/report.md` (git dışında; değerler aşağıya aktarıldı)

## 1. Önceden belirlenen ölçütler (plan §8.2)

| Ölçüt | Baseline | Yeşil | Fark | Sonuç |
|---|---|---|---|---|
| Başvuru başına net enerji | 0,01647 ± 0,00192 Wh | 0,01123 ± 0,00064 Wh | −%31,8 | GEÇTİ |
| LLM çağrısı | 111 | 44 | −%60,4 | GEÇTİ |
| Retention sonrası depolama | 3.777.982 B | 147.456 B | −%96,1 | GEÇTİ |
| `precision@10` (eşik: baseline − 0,10 = 0,40) | 0,500 | 0,600 | +0,10 | GEÇTİ |

**K1: GEÇTİ.**

Diğer sonuçlar:

- Başvuru başına brüt enerji: 0,0407'ye karşı 0,0272 Wh (−%33,3).
- CO₂: −%33,3.
- Süre: −%34,3.
- İstem tokenı: −%84,3.
- `overlap@10`: 0,50.

## 2. Sağlamlık kontrolleri

### 2.1 Boşta güç oynaklığı

Koşulardan önceki ölçümler 40,4, 42,2 ve 34,5 W çıktı; ortalamaları (39,05 W) kullanıldı. Koşulardan sonraki ölçüm 27,1 W çıktı. Net enerji boşta güce doğrudan bağlı olduğu için hesap ölçülen her değerle tekrarlandı:

| Boşta güç | Yeşil net/başvuru | Baseline net/başvuru | Fark | En kötü yeşil < en iyi baseline |
|---|---|---|---|---|
| 27,15 W | 0,01608 | 0,02386 | −%32,6 | evet |
| 34,54 W | 0,01307 | 0,01927 | −%32,2 | evet |
| 39,05 W | 0,01123 | 0,01647 | −%31,8 | evet |
| 42,17 W | 0,00996 | 0,01453 | −%31,5 | evet |

**Sonuç:** Mutlak net değerler boşta güce duyarlı, ama göreli fark ve yön değişmiyor. Raporda brüt ve net birlikte verilmeli; net için bu aralık belirtilmeli.

### 2.2 GPU'nun eksik ölçülen kısmı (D4)

Aşama toplamı, tüm koşu boyunca ölçülen GPU akışının yeşil hatta %94,4–97,1'ini, baseline'da %99,2–99,4'ünü yakaladı. Aradaki eksik GPU enerjisi brüte eklendiğinde:

- Yeşil: 3,013'ten 3,082 Wh'a çıkıyor.
- Baseline: 4,519'dan 4,537 Wh'a çıkıyor.
- Fark −%33,3'ten −%32,1'e iniyor.

**Sonuç:** Yeşil hat lehine küçük bir ölçüm eksikliği var, ama iddiayı etkilemiyor. Raporda belirtilmeli.

### 2.3 Yeşil koşular arası değişkenlik

Yeşil koşuların süreleri 113,7, 176,2 ve 198,7 s. Değişkenliğin tamamı `llm_summary` aşamasından geliyor: aynı 44 çağrı 77, 135 ve 154 s sürdü. Diğer aşamalar koşudan koşuya kararlı. Baseline'ın LLM çağrıları kararlı (241–256 s).

**Olası nedenler (doğrulanmadı):** GPU ısınması, ya da embedding modeli ile LLM birlikte yüklüyken 4 GB VRAM'in dolması ve Windows'un ekran kartı belleğini sistem belleğine taşıması.

**Sonuç:** Yavaşlama yeşil hattın enerjisini artırıyor, yani karşılaştırmayı yeşil hattın aleyhine etkiliyor. T-17'de, koşu sırasında GPU'nun paylaşılan bellek kullanımı izlenerek araştırılmalı.

### 2.4 Diğer gözlemler

- **PDF okuma süresi (D6):** Başvuru başına yeşil hatta 290 ms, baseline'da 270 ms. Fark yeşil hattın aleyhine.
- **Ölçülmeyen süre:** Yeşil hatta %1–2, baseline'da sıfıra yakın.
- **Kalite metriği:** Baseline'ın 3B modeli adayları zayıf ayırt ediyor (T-10 gözlemi). Yeşil hattın kalitede önde çıkması kısmen buna bağlı ve raporda belirtilmeli.
- **Sert filtrenin etkisi:** Veri setindeki CV'lerin medyan deneyimi 12 yıl olduğu için `min_years` filtresi yalnızca 4 başvuruyu eledi. Kalite farkı büyük ölçüde sıralamadan geliyor.

## 3. Karar

K1 geçti ve sonuç yukarıdaki üç kontrolde yön değiştirmedi. Arayüz aşamasına (T-12) geçilebilir. T-17 için açık kalan işler:

- Boşta güç oynaklığının kaynağı.
- Yeşil hattaki LLM gecikmesinin kaynağı.
- İki senaryo ile N=300.
