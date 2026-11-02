# T-14 Planı: Retention ve KVKK

- **Durum:** Onaylandı
- **Girdiler:** SPEC §8 ve SC-01, SC-04, SC-06. T-09'da yazılan asgari retention (R2, R3, tek taslak için R1, refcount silmesi).

## 1. Kapsam

`app/privacy/retention.py` tamamlanır:

- `run(session, now)` R1–R5 kurallarını uygular.
- Sonuç bir rapor nesnesidir: her kural için kaç kayıt silindiği.
- Saat her zaman `now` parametresiyle gelir; kurallar tekrar çalıştırılınca aynı sonucu verir (SPEC §8.3).

TASKS T-14 kabul kriterleri:

- Her kural için bir test.
- İki ilana başvurmuş bir profil, birinden elenince silinmez.
- Karar kaydının süresi içinde aynı ilana tekrar başvuru engellenir.

## 2. Kurallar

| Kural | Tetikleyici | Eylem | Durum |
|---|---|---|---|
| R1 | Süresi dolmuş taslak (`created_at + DRAFT_TTL_HOURS < now`) | Başvuru silinir, kayıt yazılmaz, refcount | Toplu sürüm eklenecek |
| R2 | Sert filtreyle eleme | Gerekçeli karar kaydı, başvuru silinir, refcount | Var (T-09) |
| R3 | İlan kapatılır | `ranked` başvurular için `NOT_SELECTED`, silme, refcount | Var (T-09) |
| R4 | İlan `finalized`, ya da `closed_at + SHORTLIST_RETENTION_DAYS < now` | `shortlisted` başvurular silinir, kayıt yazılmaz, refcount | Eklenecek |
| R5 | `DecisionRecord.expires_at < now` | Kayıt silinir | Eklenecek |

Silinen her satır `GreenCounter.records_deleted` sayacına eklenir; birim T-09'daki gibi satır sayısıdır. `VACUUM` bu fonksiyonda çalışmaz; SPEC'e göre yalnızca benchmark'ta ve panel butonunda çalışır (T-16).

## 3. Tetikleyiciler ve uç noktalar

- **Uygulama başlangıcı:** Lifespan içinde bir kez çalışır. Günlüğe yalnızca sayılar yazılır.
- **`POST /employer/jobs/{id}/close` (SC-04):** Önce R3, ardından `run(now)`.
- **`POST /employer/jobs/{id}/finalize` (SC-04):** Yalnızca kapalı ilan için çalışır: durumu `finalized` yapar, `finalized_at` yazar, ardından `run(now)`. Bu sırada R4 tetiklenir.
- **Butonlar:** "Kapat" ve "Sonuçlandır" T-12'deki işveren ilan listesinde gösterilir; T-15'teki detay sayfası da aynı uç noktaları kullanacak. Her ikisi de işveren girişi gerektirir.
- **Panel butonu ve "+N gün" alanı:** T-16'da aynı `run` fonksiyonunu kaydırılmış `now` ile çağıracak. T-14 yalnızca fonksiyonu hazırlar.

## 4. Onay istenen kararlar

| No | Konu | Öneri | Gerekçe |
|---|---|---|---|
| T14-1 | Kapatılan ilanın taslakları | İlan kapatılınca o ilana ait taslaklar hemen silinir: kayıt yazılmaz, refcount uygulanır. Bu SPEC'e küçük bir ek olur (**SC-22**, R3'e ek) | Kapalı ilanda taslak artık onaylanamıyor (T-13). Süresinin dolmasını beklemek, gereksiz yere kişisel veri tutmak demek |
| T14-2 | Sonuçlandırma sırası | `open` durumundaki ilan doğrudan sonuçlandırılamaz; önce kapatılmalı | R3'ün `NOT_SELECTED` kayıtlarının yazılması garanti olur; aksi halde sıralanmış adaylar gerekçesiz silinebilir |
| T14-3 | Başlangıçta çalışma | Başlangıç çalışmasının hatası uygulamayı durdurmaz; günlüğe yalnızca hata sınıfı yazılır | Uygulama bir veri temizliği hatası yüzünden açılamaz olmamalı; hata görünür kalır |
| T14-4 | Kısa listedeki adayların iletişim bilgisi | R4 kısa listedeki başvuruyu silince profil refcount ile gider; profil gidince iletişim bilgisi de cascade ile silinir | SPEC'in "kısa liste süresi" amacı tam olarak bu |
| T14-5 | Durum sayfası | R4 ile silinen kısa liste başvurusunun durum sayfası "kayıt silindi" mesajını gösterir. R5 ile silinen karar kaydı için de aynısı geçerlidir | SC-01'deki arama sırasıyla tutarlı |

## 5. Testler

Saat testlerde enjekte edilir.

- **R1:** Süresi dolan taslak silinir; dolmayan kalır. Sınır testi: tam TTL anı.
- **R2 ve R3:** Mevcut testler korunur.
- **SC-22:** İlan kapanınca taslaklar silinir.
- **R4:** Hem `finalized` yoluyla hem süre yoluyla çalışır; açık ilan ve süresi dolmamış kapalı ilan etkilenmez.
- **R5:** Süresi dolan kayıt silinir; dolmayan kalır. Silindikten sonra aynı ilana yeniden başvuru açılır.
- **Tekrar çalıştırma:** `run` iki kez çalıştırılınca ikincisi hiçbir şey silmez.
- **Refcount:** İki ilana başvurmuş profil birinden elenince kalır, ikincisinden de elenince silinir; bölümler ve iletişim bilgisi de gider.
- **Tekrar başvuru:** Karar kaydı süresi içinde aynı ilana tekrar başvuru engellenir.
- **Uç noktalar:** Girişsiz istek reddedilir; açık ilanı sonuçlandırma reddedilir (T14-2); kapat ve sonuçlandır doğru kuralları tetikler.
- **Başlangıç:** Uygulama açılırken `run` çalışır; hatası uygulamayı düşürmez (T14-3).
- **Mimari kural:** `retention.py` içinde `datetime.now` çağrısı yok (zaten var).
- **Sayaç:** `records_deleted` beklenen satır sayısı kadar artar.

## 6. Gerçek kontrol

Demo veritabanında akış uçtan uca çalıştırılır: iki demo CV başvurusu → ilanı kapat → sonuçlandır → `now`'ı 31 gün ileri sar → R5. Her adımda tablo sayıları raporlanır.
