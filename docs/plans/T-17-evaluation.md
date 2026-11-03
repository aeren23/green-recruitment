# T-17 Nihai Değerlendirme (Senaryo A, N=300)

- **Tarih:** 2026-11-03
- **Kod:** `6647744`, git temiz
- **Ortam:** Dengeli güç planı, prizde, OneDrive istemcisi kapalı, diğer uygulamalar kapalı
- **Senaryo A:** N=300, `jobs_per_cv=1`, `resubmit_rate=0`; 300 başvuru
- **Koşu düzeni:** Yeşil run 7, 9, 11; baseline run 8, 10, 12; dönüşümlü.
- **Boşta güç:** Koşulardan önce 120 s'lik üç ölçüm: 44,30, 39,12 ve 39,87 W; ortalaması (41,10 W) kullanıldı. Koşulardan sonraki ölçüm: 43,15 W.
- **Kaynak:** `docs/plans/T-17-report.md` (`python -m bench compare` çıktısı)

## 1. Kapsam ve sapma

Varsayılan senaryo N=300'de **koşulmadı**. 2026-11-03'te kullanıcı, ısıl kısıtlama ve zaman nedeniyle bunu bırakmaya karar verdi (önce tekrar sayısı 1'e indirildi, sonra tamamen atlandı). Varsayılan senaryo için rapor K1'e dayanır (N=50, 3 tekrar, run 1–6, başvuru başına net enerji −%31,8). Bu, TASKS T-17'deki "her biri 3 tekrar" ifadesinden bir sapmadır; görev "ya da zamanın elverdiği kadar" diyerek buna izin veriyor, ama açıkça belirtilmelidir.

## 2. Sonuçlar (Senaryo A, ortalama ± standart sapma)

| Ölçüt | Baseline | Yeşil | Fark | Sonuç |
|---|---|---|---|---|
| Başvuru başına net enerji | 0,02230 ± 0,00132 Wh | 0,00720 ± 0,00043 Wh | −%67,7 | GEÇTİ |
| LLM çağrısı | 300 | 60 | −%80,0 | GEÇTİ |
| Retention sonrası depolama | 9.657.837 B | 212.992 B | −%97,8 | GEÇTİ |
| `precision@10` (eşik: baseline − 0,10 = 0,767) | 0,900 | 0,867 | −0,033 | GEÇTİ |

**K1: dört ölçütün hepsinde GEÇTİ.**

Diğer sonuçlar:

- Başvuru başına brüt enerji: 0,08358'e karşı 0,02780 Wh (−%66,7).
- İstem tokenı: 358.587'ye karşı 32.041 (−%91,1).
- `overlap@10`: 0,467.

**Dürüst bulgu:** Retention öncesi (aktif) depolamada yeşil hat baseline'dan **büyük**: 12.169.216 B'a karşı 9.657.837 B (+%26). Nedeni, bölüm embedding'lerinin ve SQLite sayfalarının saklanması. Depolama kazancı yalnızca retention sonrası ortaya çıkıyor.

## 3. Sağlamlık kontrolleri

### 3.1 Boşta güç

Net enerji, ölçülen her boşta güç değeriyle yeniden hesaplandı:

| Boşta güç | Net enerji farkı | En kötü yeşil < en iyi baseline |
|---|---|---|
| 39,12 W | −%67,6 | evet |
| 41,10 W | −%67,7 | evet |
| 43,15 W | −%67,9 | evet |
| 44,30 W | −%68,1 | evet |

Çift bazında fark (7–8, 9–10, 11–12): yaklaşık −%70, −%69 ve −%63.

**Sonuç:** Boşta güç aralığı sonucu değiştirmiyor.

### 3.2 GPU'nun eksik ölçülen kısmı (D4)

Aşama toplamı / GPU akışı oranı yeşilde 0,930–0,956, baseline'da 0,987–0,993. Eksik GPU enerjisi brüte eklenince koşu başına yeşil 8,34'ten 8,60 Wh'a, baseline 25,08'den 25,20 Wh'a çıkıyor. Brüt fark −%66,7'den −%65,9'a iniyor.

**Sonuç:** İddiayı değiştirmiyor.

### 3.3 Isıl kısıtlama

Dizüstü yatağın üzerindeydi. GPU izleyicisi (nvidia-smi, 5 s aralıkla) run 7–10 sırasında medyan yaklaşık 89 °C ve çoğunlukla 210 MHz SM saati gösterdi. Baseline `llm_score` çağrı başına yaklaşık 6,4 s sürdü (K1'de yaklaşık 2 s); yeşil `llm_summary` 394 ve 484 s oldu. Kullanıcı yaklaşık 03:50'de bir fan açtı; run 11–12 daha serin çalıştı (run 11'de medyan saat yaklaşık 900 MHz, `llm_summary` 190 s; baseline run 12 559 s sürdü, run 8 ve 10'da 2087 ve 2185 s).

Bu yüzden süreler ve brüt enerji tekrarlar arasında çok oynadı (baseline brüt 25,08 ± 10,75 Wh). Net enerji ise kararlı kaldı (yaklaşık ±%6), çünkü kısıtlama altındaki ek süre çoğunlukla boşta güce yakın harcanıyor. Fark yönü hem sıcak çiftlerde hem serin çiftte korundu; serin çift en küçük farkı verdi (−%63).

**Sonuç:** K1'de açık kalan yeşil LLM gecikmesi sorusu cevaplandı: K1'deki değişkenlik büyük olasılıkla VRAM değil, ısıl kısıtlamaydı. Tam olarak kanıtlanmış değildir.

## 4. K1'e göre neden daha büyük tasarruf (−%68 ve −%32)

Senaryo A'da ilan başına yaklaşık 100 başvuru var; LLM yalnızca ilk k aday için çağrılıyor (ilan başına 20, toplam 60 çağrı). K1'de ilan başına yaklaşık 37 başvuru vardı (111 çağrının 44'ü). Tasarruf, ilan başına başvuru sayısıyla artıyor. Varsayılan senaryodaki önbellek etkileri (parse ve embedding önbellek isabetleri) yalnızca K1'de görülüyor.

## 5. Yıllık 1 milyon başvuru projeksiyonu (tahmin)

`T-17-report.md` içindeki projeksiyon bölümüne bakın. Senaryo A için: net enerji yaklaşık 22,30 kWh'ye karşı 7,20 kWh (−%67,7); brüt 83,58'e karşı 27,80 kWh (−%66,7); SCI (M hariç) 38,83'e karşı 12,91 kg CO₂e. Varsayılan senaryo (K1): net 16,47'ye karşı 11,23 kWh. Bunlar bu dizüstü ve bu senaryo için doğrusal ekstrapolasyondur; tahmindir, ölçüm değildir.

## 6. Sınırlılıklar

- Veri seti anonimleştirilmiş.
- Baseline küçük bir 3B model kullanıyor.
- Tek dizüstünde ölçüldü.
- Isıl koşullar sabit değildi (run 11–12 daha serindi).
- Varsayılan senaryo N=300'de koşulmadı.

## 7. Karar

T-17 tamamlandı.
