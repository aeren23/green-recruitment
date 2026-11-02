# T-13 Planı: Aday Akışı

- **Durum:** Onaylandı (T13-1 ile T13-5 kararları dahil)
- **Girdiler:** SPEC §3 (adım 2–4), §6.1, §8.1, §9; SC-01, SC-02, SC-03, SC-07; ölçüm kararı D3; T-12'deki şablon iskeleti

## 1. Kapsam ve kabul kriterleri

Akış: yükleme → taslak → önizleme ve düzeltme → onay → durum sayfası. Akışın kendisi T-09'daki `app/pipeline/` paketindedir. Web katmanı yalnızca HTTP isteklerini bu fonksiyonlara çevirir ve sonucu Türkçe sayfalarla gösterir; iş mantığı içermez.

TASKS T-13 kabul kriterleri:

- SPEC §3'ün 2., 3. ve 4. adımları uçtan uca çalışır.
- PDF'in diske yazılmadığı bir testle kanıtlanır.

## 2. Sayfalar

| Yol | İçerik |
|---|---|
| `GET /` | Açık ilanlar, 20'li sayfalama |
| `GET /jobs/{id}` | İlan detayı, sert filtreler, KVKK aydınlatma metni, CV yükleme formu. Kapalı ilanda form gösterilmez |
| `POST /jobs/{id}/apply` | Yükleme. Başarılıysa `/drafts/{token}` adresine yönlenir; hata Türkçe olarak aynı sayfada gösterilir |
| `GET /drafts/{token}` | Önizleme ve düzeltme formu |
| `POST /drafts/{token}` | Onay. Başarılıysa `/applications/{token}` adresine yönlenir |
| `GET /applications/{token}` | Durum sayfası |

## 3. Önizleme sayfası (SPEC §3 adım 2–3)

**Mesajlar:**

- "Dosyanız saklanmadı" her yüklemeden sonra gösterilir.
- Önbellekten gelindiyse "Önceki başvurunuzdan alındı, yeniden işlenmedi" yazar.
- CV'de başlık bulunamadıysa bir uyarı gösterilir.

**Düzenlenebilir alanlar:**

- Bölümler: her biri için tür seçimi ve metin alanı. Metni boşaltılan bölüm silinir. Formun sonunda iki boş satır bulunur, böylece JavaScript olmadan yeni bölüm eklenebilir.
- Deneyim yılı, diller, konum. Boş bırakılan konum `None` olarak kaydedilir, boş metin olarak değil (T-07 notu).
- İletişim bilgisi: ad, e-posta, telefon. Yalnızca aday görür; işveren ancak kısa listede görür.

**"Hakkınızda ne saklanacak" özeti:** Hangi alanların saklanacağı ve ne kadar süre tutulacağı gösterilir (SPEC §8.1, aydınlatma).

## 4. Durum sayfası (SC-01)

Arama sırası: önce başvuru, yoksa `application_token` ile karar kaydı, o da yoksa "Bu başvurunun kaydı saklama süresi dolduğu için silindi" mesajı.

| Durum | Türkçe metin |
|---|---|
| `draft` | Onayınızı bekliyor (önizlemeye bağlantı) |
| `ranked` | Değerlendirmede |
| `shortlisted` | Kısa listeye alındınız |
| Karar kaydı: `LANG_MISSING`, `LOCATION_MISMATCH`, `MIN_YEARS` | "Elendiniz", ardından gerekçe ve itiraz hakkına dair kısa not (KVKK m.11) |
| Karar kaydı: `NOT_SELECTED` | Bu ilan için seçilmediniz |

## 5. Güvenlik ve KVKK

- **1 MB sınırı (SC-02):** Başvuru yükleme isteğinin gövdesi baytlar sayılarak ölçülür, sınır aşılırsa 413 ve Türkçe hata döner. Bir test, `MAX_UPLOAD_BYTES` değerinin Starlette'in disk eşiğinden büyük olmadığını doğrular; Starlette güncellenirse kural sessizce bozulamaz.
- **PDF diske yazılmaz:** Yüklenen baytlar bellekte okunup pipeline'a verilir. Test iki yönlüdür: sınır altındaki PDF'de ve sınırı aşan PDF'de, geçici dizinde hiçbir dosya oluşmamalı.
- **Token maskeleme:** `uvicorn.access` günlüğünde `/drafts/…` ve `/applications/…` yollarındaki token maskelenir, çünkü token iletişim bilgisine erişim sağlıyor.
- **Tracker:** Web katmanı her zaman `NoOpStageTracker` kullanır (D3).
- **Düzeltmelerde iletişim temizliği:** Aday bir bölüme e-posta veya telefon yazarsa pipeline bunu temizler. Web katmanı ayrıca bir şey yapmaz, pipeline'ı çağırır.

## 6. Onay istenen kararlar

| No | Konu | Öneri | Gerekçe |
|---|---|---|---|
| T13-1 | LLM özetinin zamanı | Onay isteği LLM olmadan tamamlanır. Top-k özetleri, FastAPI'nin yerleşik `BackgroundTasks` özelliğiyle ve kendi oturumuyla yanıttan sonra üretilir. Bunun için pipeline'a küçük bir genel fonksiyon eklenir (ilan için top-k özetlerini tazele) | Aday 2–5 saniye Ollama'yı beklemez. Mantık yine pipeline'da kalır |
| T13-2 | Süresi dolmuş taslak | `GET /drafts/{token}` adresinde "Taslağınızın süresi doldu, lütfen yeniden yükleyin" mesajı ve ilan sayfasına bağlantı gösterilir; taslak R1 kuralıyla silinir. Bu bir yanıt kodudur, kayıt yazılmaz (SC-06) | Önceki oturumdaki karar |
| T13-3 | Tekrar başvuru | İlan sayfasında Türkçe hata gösterilir. Taslak varsa doğrudan taslağa yönlenir (SC-07) | SPEC §3 adım 4 |
| T13-4 | Demo CV | Kurgusal kişiyle sentetik bir İngilizce CV PDF'i (ad, e-posta, telefon, 4–5 yıl deneyim, Türkçe ve İngilizce dil bilgisi) testlerdeki PDF üreticisiyle bir kez üretilir ve `docs/demo/demo-cv.pdf` olarak repoya eklenir | SPEC §3 demosu için gerçek bir PDF gerekiyor. Veri setinde iletişim bilgisi yok. Üretici yalnızca ASCII desteklediği için içerik İngilizce |
| T13-5 | Aday tarafında oturum | Aday için çerez yoktur. URL'deki tahmin edilemez token, erişim yetkisi işlevi görür | SPEC'te aday girişi yok; en sade ve veri minimizasyonuna uygun yol |

## 7. Testler

- **Uçtan uca, SPEC §3 adımları (sahte embedder ve sahte LLM):**
  - Adım 2: yükleme → önizleme (mesajlar ve alanlar) → düzeltme → onay → "Değerlendirmede".
  - Adım 3: aynı CV ikinci ilana yüklenir → önbellek mesajı görünür, parse atlanır.
  - Adım 4: aynı CV aynı ilana tekrar yüklenir → engellenir.
- **Durum sayfası:** Eleme gerekçesi, "seçilmedi", silinmiş kayıt ve süresi dolmuş taslak durumları.
- **PDF diske yazılmaz:** Küçük PDF ve sınırı aşan PDF için geçici dizin izlenir, 413 yanıtı doğrulanır, sınır testi yapılır.
- **Gizlilik:** Günlüklerde token maskelenir. Hiçbir yanıtta ve günlükte başka bir adayın verisi bulunmaz. Düzeltmede bölüme yazılan e-posta temizlenir.
- **Arka plan özeti:** Yanıt LLM'i beklemeden döner; arka plan işi çalıştıktan sonra top-k özetleri yazılmış olur. Ollama yokken akış bozulmaz.
- **Arayüz bütçesi:** Her sayfa 50 KB altında; dış kaynak ve JavaScript yok; otomatik kaçış.

## 8. Gerçek kontrol

Gerçek embedding modeli ve gerçek Ollama ile uygulama başlatılır ve SPEC §3'ün 2–4. adımları `docs/demo/demo-cv.pdf` ile betik üzerinden çalıştırılır. Ayrıca sayfa boyutları ve özetin arka planda gelme süresi ölçülür.
