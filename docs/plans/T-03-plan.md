# T-03 Planı: Bölüm Ayırıcı ve İletişim Çıkarımı

- **Durum:** Onay bekliyor (TASKS: T-03 için kod öncesi plan onayı zorunlu)
- **Girdiler:** `docs/plans/dataset-findings.md`, SPEC §6.3, TASKS T-03
- **Yöntem notu:** Bulgular 200 PDF'lik örnekten geliyor. Ana oranları farklı tohumla seçilen 300 PDF'te ayrıca doğruladım (tire kuralı %15, `to` kalıbı %95, boşluk satırı payı %19).

## 1. Sorun

SPEC §6.3 olduğu gibi uygulanırsa kabul hedefi tutar, ama üretilen veri benchmark'ı yanıltır:

| Konu | SPEC'e göre davranış | Veride gerçekleşen |
|---|---|---|
| Yıl aralıkları | Tireli `2015 - 2019` | Belgelerin yalnızca %13,5'i yakalanır, `years_experience` çoğunlukla boş kalır, `MIN_YEARS` filtresi çalışmaz |
| `Current` bitişi | Bugünün tarihi varsayılır | Belgelerin %55'inde var; CV'ler çoğunlukla 2015–2017 dönemi, deneyim yaklaşık 10 yıl şişer |
| Satır başı başlık eşleme | `Experience in ...` başlık sanılır | Belgelerin %10'unda düz cümle |
| Boş satırla girdi bölme | pypdf'in boşluk satırları ayraç olur | Satırların %19'u, deneyimler küçük parçalara bölünür ve embedding sayısı artar |
| Eksik başlıklar | `Accomplishments` vb. önceki bölüme karışır | Belgelerin %35'i; kişisel alanlar da embedding'e girebilir |

## 2. Onay istenen SPEC değişiklikleri

| No | Yer | Değişiklik |
|---|---|---|
| SC-15 | §6.3 başlık tablosu | `summary`: professional summary, executive profile, career overview, career focus, professional profile, accomplishments, additional information. `skills`: highlights, core qualifications |
| SC-16 | §6.3 | Yeni "yok sayılan" başlıklar: interests, affiliations, professional affiliations, personal information. Metinleri saklanmaz ve embed edilmez |
| SC-17 | §6.3 | Başlık yalnızca tam satır eşleşmesiyle kabul edilir (kırpma sonrası, sondaki `:` hariç) |
| SC-18 | §6.3 | Tarih aralığı ayraçları `-`, en dash, `to`, `through`; biçimler `YYYY`, `MM/YYYY`, `Month YYYY`; bitiş sözcükleri Present, Current, Now, Günümüz; eşleşme satırlar arasında çalışır, ay bilgisi varsa ay çözünürlüğü kullanılır |
| SC-19 | §6.3, §7.3, §10 | Ayrıştırıcı `reference_date` alır. Web `now` verir, benchmark sabit `BENCH_CV_REFERENCE_DATE=2018-01-01` kullanır |
| SC-20 | §6.3 | `experience` ve `project` girdileri tarih aralıklarına göre bölünür. Yalnızca boşluk içeren satırlar ayraç değildir; gerçek boş satır yalnızca bölümde tarih yoksa kullanılır |

**Gerekçeler:**

- **SC-15, Accomplishments ve Additional Information → `summary`:** Bunlar kanıt değil beyandır, bu yüzden deneyim katsayısı (1.0) yerine özet katsayısı (0.7) alır. Yeni bölüm türü eklemek şemayı büyütürdü.
- **SC-16:** Veri minimizasyonu için. "Personal Information" gibi başlıklar SPEC §8.1'in toplamadığı alanları içerebilir.
- **SC-19, 2018-01-01:** Benchmark tekrar üretilebilir olur. Örneğimde `Current` içeren CV'lerin büyük çoğunluğunda en büyük yıl 2018 veya öncesi. Bu değer raporda sınırlılık olarak yazılır.

## 3. Tasarım

- **`app/ingest/sections.py`:** `parse_cv(text, reference_date) -> ParsedCv`. Çıktı `profile_json` şemasına uyar, `parser_version` "1".
- **Girdi:** `extract_text` çıktısı. `normalize` kullanılmaz; ayrıştırma satır yapısına bağlıdır.
- **Akış:** Satır kırpma → gürültü satırı temizleme (U+200B, harf veya rakam içermeyen ve mojibake artığı kısa satırlar) → başlık bulma → bölüm atama → girdi bölme → 200 sözcükte parçalama → yıl hesabı.
- **`fold_heading()`:** NFKC, `İ`/`I`/`ı` birleştirme, U+0307 temizleme, casefold. `DENEYİM` ve `HAKKIMDA` böylece eşleşir; SPEC §6.2 ve hash değişmez.
- **İlk başlıktan önceki metin:** `summary` olur. Hiç başlık yoksa SPEC'teki tek `summary` ve uyarı davranışı korunur.
- **`years_experience`:** Yalnızca `experience` girdilerindeki aralıklardan hesaplanır; eğitim tarihleri sayılmaz. Örtüşen aralıklar birleştirilir, `reference_date` sonrası kırpılır. Bitişi başlangıçtan önce olan aralık yok sayılır.
- **`languages`:** Embed edilmez, küçük TR→EN sözlüğüyle eşlenir (SC-08). `location` boş bırakılır.
- **`app/ingest/contact.py`:** E-posta ve telefon regex ile, ad ilk satır sezgisiyle bulunur (2–4 sözcük, rakam yok). Bulunan değerler bölüm metinlerinden çıkarılır.
- **Bilinen sonuç:** Veri setinde ilk satır iş unvanı olduğundan ad sezgisi unvanı bulur ve metinden çıkarır. Bunu kabul ediyorum: kaçan bir ad KVKK ihlali olurken, kaybolan bir unvan yalnızca küçük bir kalite kaybıdır. Yeşil hattın aleyhine bir etkidir, raporda belirtilir.
- **Doğrulanmayan alan:** Veri setinde iletişim bilgisi ve Türkçe başlık neredeyse yok. Bu yollar yalnızca sentetik örneklerle test edilebilir.

## 4. `parse-report --n 50` çıktısı ve hedefler

| Ölçüt | Hedef | Kaynak |
|---|---|---|
| 2+ bölüm türü bulunan CV | ≥ %80 | TASKS |
| `years_experience` bulunan CV | ≥ %90 | Plan hedefi %60 idi; ölçülen potansiyel %99 olduğu için yükseltildi |
| Başlıksız CV oranı | Raporlanır | TASKS |
| CV başına medyan `Section` | 5–35 | SPEC §4'teki 10–30 varsayımına yakın; parçalanmayı yakalar |
| Bölüm türü dağılımı | Raporlanır | TASKS |

`years_experience` için ek doğruluk kontrolü: rastgele 30 CV'de hesaplanan değer, belgedeki tarihlerden bağımsız yöntemle hesaplanan değerle karşılaştırılır; yalnızca sayısal fark raporlanır, CV metni alıntılanmaz.

## 5. Testler

TASKS'taki birim testlerine ek olarak:

- Büyük harfli Türkçe başlıklar.
- `Experience in ...` satırının başlık sayılmaması.
- Satırlara bölünmüş `to` aralığı, `MM/YYYY`, `Month YYYY`, `Current` ve `Günümüz`.
- `reference_date` etkisi, eğitim tarihlerinin sayılmaması.
- Boşluk satırlarının girdi bölmemesi, tarihsiz bölümde gerçek boş satırla bölme.
- Yok sayılan başlık metninin çıktıda bulunmaması.
- E-posta, telefon ve adın bölüm metninden ve `profile_json`'dan çıkarılması.
- Sızıntı kontrolü: ayrıştırıcı hata ve günlüklerinde CV metni bulunmaması.

## 6. Kapsam dışı

Mojibake onarımı yapılmaz, gürültü satırları yalnızca atılır. `Company Name`, `City`, `State` yer tutucuları veri setine özgü olduğundan koda kural olarak girmez. Hash hesabı değişmez.
