# Veri Seti Bulguları (T-01 doğrulaması, T-03 girdisi)

Bu dosya Kaggle `snehaanbhawal/resume-dataset` üzerinde yapılan ölçümlerin özetidir. KVKK gereği yalnızca sayılar, yüzdeler ve genel biçim kalıpları yer alır; veri setinden hiçbir özgeçmiş metni, ad, e-posta veya telefon alıntılanmamıştır. Ölçüm betikleri repoda değildir (geçici dizinde çalıştırıldı), sonuçlar aşağıdadır.

## 1. Yöntem

- İndirme: resmi herkese açık URL, anonim. Arşiv 65.529.677 bayt, 2485 giriş (1 CSV + 2484 PDF), açılmış boyut 118.289.355 bayt (yaklaşık 113 MiB).
- Arşiv denetimi: geçerli zip, CRC hatası yok, mutlak yol yok, `..` yok, sembolik bağ yok, yalnızca `.csv` ve `.pdf`.
- PDF ölçümleri: 2484 kaydın `(id, category)` sırasıyla dizilmiş listesinden `random.Random(20260101).sample(..., 200)` ile seçilen 200 PDF üzerinde `app.ingest.pdf.extract_text` çalıştırıldı. Yüzdeler 200 belgeye göredir (bir belge = %0,5). Örneklem küçüktür; %1 altı farklara anlam yüklenmemelidir.
- Başlık analizinde, KVKK nedeniyle yalnızca belgelerin en az %5'inde geçen satırlar listelenmiştir.

## 2. Yapı (SPEC §7.1 ile karşılaştırma)

Yapı SPEC ile birebir uyumludur: `Resume/Resume.csv` (sütunlar `ID`, `Resume_str`, `Resume_html`, `Category`) ve `data/data/<CATEGORY>/<ID>.pdf`. 24 kategori, 2484 kayıt, 2484 PDF, 0 eksik PDF. SPEC'te değişiklik gerekmez.

Kategori adları SPEC §7.2 ve `bench/dataset.py` ile aynıdır: `INFORMATION-TECHNOLOGY` (120), `ACCOUNTANT` (118), `FINANCE` (118), `HR` (110). Katmanlar: IT 120, ACCOUNTANT+FINANCE 236, HR 110, distractor 2018 kayıt. Her katman N=300 için gereken 75'i karşılar. Havuz büyüklüklerinden hesaplanan tavan (denenmedi): HR 110 kayıtla sınırlı olduğundan N en fazla 440 olabilir; IT için 480.

Tekrarlanabilirlik: aynı seed ayrı süreçlerde aynı örneği verdi (n=50 ve n=300, seed 42; sıralı kimlik listesinin özeti eşit). Farklı seed (43) farklı örnek verdi. N=300 örneği 75/75/75/75 dağılımlıdır.

## 3. PDF metin çıkarımı (`extract_text`, 200 belge)

| ölçüt | değer |
|---|---|
| Başarılı | 200 / 200 |
| `UnreadablePdfError` | 0 |
| `TextTooShortError` | 0 |
| Çıkarılan metin uzunluğu (karakter) | medyan 5515,5; en az 1085; en çok 29496 |
| Satır sayısı, boşluk dışı satırlar | medyan 124 (en az 47, en çok 538) |
| Satır sayısı, tüm satırlar | medyan 155 |
| Satır başına karakter (boşluk dışı) | medyan yaklaşık 44 |
| `\n\n` içeren belge | %0 |
| Yalnızca boşluk (` `) içeren satır | 6547 / 33381 satır (%19,6); belgelerin %100'ünde var |
| Sonunda boşluk olan satır | %34,6 |
| Başında boşluk olan satır | %19,9 |

Önemli: `pypdf` her metin parçasını ayrı satıra koyar ve parçalar arasına yalnızca bir boşluktan oluşan satırlar ekler. Gerçek boş satır (`""`) neredeyse yoktur (tüm örneklemde 1). Bölümleyici "boş satır" ayracını `line.strip() == ""` olarak anlamalı, satırları baş ve sondan kırpmalıdır. `"\n\n"` ile bölmek hiçbir şey bulmaz.

### 3.1 PDF çıktısı ile `Resume_str` karşılaştırması

- İlk 20 belirteç (a-z0-9) 200 belgenin 200'ünde konumsal olarak aynıdır; belirteç kümesi benzerliği (Jaccard) 1,0. PDF ve `Resume_str` aynı içeriktir.
- Uzunluk oranı (PDF / `Resume_str`) medyan 0,96.
- Yapı farklıdır: `Resume_str` düz metindir. Belgelerin %60'ı tek satırdır, %40'ında satır sonu vardır (medyan 0), `Skills` kelimesinden önce satır sonu olan belge %0. PDF çıktısında başlıklar kendi satırındadır (bölüm 4). Bölüm ayırma PDF çıktısı üzerinde yapılmalıdır, `Resume_str` bölümlemeye uygun değildir.
- Karakter kodlaması: PDF çıktısında "mojibake" (UTF-8 baytlarının yanlış çözülmesi) artığı vardır, `Resume_str` içinde yoktur. Belgelerin %78,5'inde en az biri görülür (`Â` %39, madde işareti artığı olan `ï¼` %52,5, `â€“` %9,5, `â€¢` %1,5, `â€œ` %2,0, `Ã` %3,5). Sıfır genişlikli boşluk (U+200B) belgelerin %54'ünde. Bu artıklar çoğunlukla tek başına satır olarak görünür ve madde işareti yerine geçer. Gerçek U+2013/U+2014 karakteri %0. Hash'i etkilemez (SPEC §6.2 değişmez), ama bölümleyici bu satırları gürültü olarak yok saymalıdır. En dash'in mojibake biçimi (`â€“`) tarih aralığı ayracı olarak kullanılmıyor (bölüm 5).
- ASCII dışı karakter içeren belge oranı %79; en sık 12 kod noktası yukarıdaki artıklar, kıvrık tırnak, euro ve telif işaretidir, aralarında Türkçe harf yoktur.

## 4. Başlıklar

Başlıklar PDF çıktısında kendi satırındadır ve hepsi `Title Case` yazılmıştır (`Skills`, `Education`, `Experience`, `Summary`, `Highlights` için belge başına ilk görünüm: neredeyse %100 Title Case; `Accomplishments` bir belgede büyük harf). İki nokta ile biten başlık yok denecek kadar azdır (`Skills:` %0,5). Belgelerin %95'inde TAMAMI BÜYÜK HARF kısa bir satır var; bu, ilk satırdaki iş unvanıdır (bölüm 6).

### 4.1 En sık görülen kısa satırlar (üst liste, belge payı)

Ölçüt: en çok 40 karakter, en çok 4 sözcük, rakam yok, noktalama ile bitmiyor; katlanmış küçük harf; belgelerin en az %5'inde. Bu eşikte 38 satır çıktı (40 değil).

Gerçek başlıklar:

| başlık | belge payı |
|---|---|
| skills | %94,0 |
| education | %78,5 |
| experience | %69,5 |
| summary | %47,5 |
| accomplishments | %35,0 |
| highlights | %32,5 |
| professional summary | %22,0 |
| additional information | %20,0 |
| education and training | %18,5 |
| professional experience | %17,0 |
| core qualifications | %13,5 |
| work history | %11,5 |
| certifications | %10,5 |
| interests | %10,5 |
| languages | %7,5 |
| professional affiliations | %7,0 |
| skill highlights | %6,5 |
| affiliations | %6,5 |
| work experience | %5,5 |

Başlık olmayan, ama sık görülen satırlar (bölümlemede gürültü):

| satır türü | belge payı |
|---|---|
| `Company Name` yer tutucusu | %100,0 |
| `City` ve `State` yer tutucuları (ayrı satırlar) | %98,0 ve %96,5 |
| `to` (tarih aralığı ayracı, tek başına satır) | %95,5 |
| `Current` | %55,0 |
| madde işareti mojibake satırı (`ï¼` + U+200B) | %52,5 |
| tek karakterli `-` satırı | %19,5 |
| derece adları (`bachelor of science` %24,0, `bachelor of arts` %13,5, `high school diploma` %12,5, `master of science` %6,5, `associate of arts` %5,5, `business administration` %5,5) | yukarıdaki oranlar |
| `usa` %13,0, `united states` %9,5, `us` %5,5, `gpa` %12,0 | yukarıdaki oranlar |
| `â€“` mojibake satırı %8,0, tek `â` satırı %7,5, `â€‹` satırı %6,5 | yukarıdaki oranlar |

### 4.2 SPEC §6.3 başlıkları: bulunan ve eksik olanlar

Tam satır eşleşmesine göre belge payı (satır, boşluk ve iki nokta kırpıldıktan sonra başlığa eşit):

| SPEC başlığı | belge payı | | SPEC başlığı | belge payı |
|---|---|---|---|---|
| skills | %94,0 | | experience | %69,5 |
| education | %78,5 | | professional experience | %17,0 |
| summary | %47,5 | | work history | %11,5 |
| education and training | %18,5 | | work experience | %5,5 |
| certifications | %10,5 | | languages | %7,5 |
| skill highlights | %6,5 | | qualifications | %4,0 |
| technical skills | %2,5 | | licenses | %1,5 |
| profile | %1,5 | | objective | %1,5 |
| core competencies | %0,5 | | employment | %0,5 |
| projects | %0,5 | | certificates | %0,5 |
| personal projects | %0 | | about | %0 |

SPEC'te olmayan ama sık olan başlıklar (kapsanmıyor): `Accomplishments` %35,0; `Highlights` %32,5; `Professional Summary` %22,0; `Additional Information` %20,0; `Core Qualifications` %13,5; `Interests` %10,5; `Affiliations` %6,5 (ve `Professional Affiliations` %7,0); `Executive Profile` %4,5; `Career Overview` %3,5; `Career Focus` %2,5; `Personal Information` %2,5; `Professional Profile` %2,0.

"Satır başında eşleşme" kuralının risk ölçümleri (başlıkla başlayan ve satırın devamında metin olan satırlar): `experience` + metin %10,0 (ikinci sözcük `in` ve `with`; yani "Experience in ..." gibi düz cümle başlangıçları), `education` + metin %21,0 (ikinci sözcük `and`; çoğu `Education and Training`), `employment` + metin %3,0, `awards` + metin %6,0, `projects` + metin %3,5. Başlık eşlemesi yalnızca satırın tamamı başlığa eşitse (veya iki nokta ile bitiyorsa) kabul edilmeli; yalnızca satır başı eşleşmesi düz cümleleri başlık sayar. `Education and Training` zaten SPEC listesinde tam ifade olarak var.

Bölüm türü kapsamı (SPEC listesi, tam satır eşleşmesiyle): belge payı experience %100, skills %97,0, education %96,5, summary %50,5, certification %12,5, languages %7,5, project %0,5. Belge başına bulunan tür sayısı: 2 tür 6 belge, 3 tür 76, 4 tür 102, 5 tür 15, 6 tür 1. En az 2 tür bulunan belge oranı %100 (T-03 kabul eşiği %80); hiç başlık bulunmayan belge %0. Satır başı eşleşmesiyle de aynı (%100).

SPEC dışı ek başlıkların eklenmesi bulunan tür sayısını artırmaz ama metin kapsamını artırır: `Accomplishments` ve `Highlights` belgelerin üçte birinde var ve içerikleri deneyim ile yetenek metnidir. Bu başlıklar tanınmazsa, sonraki bir başlığa kadar olan metnin hangi bölüme yazılacağı splitter tasarımına bağlıdır (ölçülmedi). Eklenip eklenmeyeceği (ve hangi türe eşleneceği) T-03 planında karar verilecek konudur; SPEC değişikliği gerektirir, burada yapılmadı.

## 5. Tarih biçimleri (200 PDF, belge payı)

Örtüşmeyen eşleşmelerle, başlangıç | ayraç | bitiş. "Aralık satır içi" sütunu, aralığın tek satırda bulunduğu belgeleri gösterir; "satır bölünmüş" biçimlerde tarihler ve ayraç ayrı satırlardadır.

| biçim | belge payı (satırlar arası dahil) | tek satırda |
|---|---|---|
| `MM/YYYY to MM/YYYY` | %47,5 | %0,0 |
| `Month YYYY to Month YYYY` | %47,5 | %0,5 |
| `MM/YYYY to Current` | %28,0 | %0,0 |
| `Month YYYY to Current` | %26,5 | %0,5 |
| `YYYY - YYYY` (tire) | %11,0 | %11,0 |
| `MM/YYYY - MM/YYYY` (tire) | %4,0 | %0,5 |
| `YYYY - Current` (tire) | %3,5 | %3,5 |
| `Month YYYY - Month YYYY` (tire) | %3,0 | %3,0 |
| `MM/YYYY - Current` (tire) | %1,5 | %0,5 |
| `YYYY to YYYY` | %1,0 | %1,0 |
| `YYYY through YYYY` | %1,0 | %1,0 |
| `YYYY to Current` | %0,5 | %0,5 |
| `Month YYYY - YYYY` (tire) | %0,5 | %0,5 |
| `Month YYYY - Current` (tire) | %0,5 | %0,5 |

Açık aralık bitiş sözcüğü (yıldan ve ayraçtan hemen sonra): `Current` %55,0, `Present` %5,5, `Now` %0, `Günümüz` %0 (`Till Date`, `To Date`, `Today`, `Ongoing` de %0). Tablodaki `Current` sütunları bu sözcüklerin hepsini kapsar. En dash (U+2013) ve em dash (U+2014) ayracı olarak 0 belgede, mojibake en dash (`â€“`) de ayraç olarak 0 belgede görüldü (tek başına satır olarak 123 kez geçiyor, ama komşu satırlarda tarih yok).

Yapısal bulgu: iki ana kalıbın (`MM/YYYY` ve `Month YYYY`) PDF'te yarı yarıya iki ayrı şablonu var ve ikisi de `to` ayracı kullanıyor. Bu ayraç ve iki tarih PDF çıktısında ayrı satırlara bölünür; tipik görünüm: tarih satırı, yalnızca boşluk içeren satır, yalnızca `to` içeren satır, bitiş tarihi veya `Current` satırı (ayrıca aralığın hemen ardında `Company Name` yer tutucu satırı). `to` ayracı yalnızca belgelerin %3,0'ında bir tarihle aynı satırdadır. Tam aralığı tek satırda barındıran belge oranı %18,0. Yani aralık eşleştirmesi satırlar arasını (boşluk-yalnız satırlar dahil) kapsamalıdır; satır satır çalışan bir regex neredeyse hiçbir `to` aralığını yakalamaz.

Aralık kapsamı:

| kural | yakalanan belge payı |
|---|---|
| SPEC §6.3 lafzı (`YYYY - YYYY`, `YYYY - Present/Current/Now`, ASCII tire) | %13,5 |
| Herhangi bir tire ayraçlı aralık (her başlangıç ve bitiş biçimi) | %19,0 |
| `to` ayraçlı aralık (her biçim, satırlar arası dahil) | %95,5 |
| Tüm ayraçlar ve biçimler (tire, `to`, `through`; satırlar arası dahil) | %99,0 |
| Yalnızca `to` ayraçlı aralığı olan belgeler | %79,5 |

Sonuç: yalnızca tire kuralı belgelerin yaklaşık %13,5'inde (en fazla %19'unda) aralık bulur; bu T-03'ün `years_experience` bulunma hedefinin (%60) çok altındadır. `to`, ay/yıl (`MM/YYYY`, `Month YYYY`) ve `Current` desteklenir ve eşleştirme satırlar arasını geçerse oran yaklaşık %99'a çıkar. `MM/YYYY` ve `Month YYYY` aralıkları ay çözünürlüğü taşır; SPEC §6.3 yalnızca yıl aralıklarını tarif ettiği için ay bilgisinin kullanılıp kullanılmayacağı T-03 planında karara bağlanmalıdır.

Ek ölçüm: belgelerin %100'ünde en az bir 4 haneli yıl, %99,5'inde tek başına `Month YYYY` veya `MM/YYYY` tarihi var.

## 6. İletişim verisi ve anonimlik

| desen | PDF çıktısı | `Resume_str` |
|---|---|---|
| e-posta | %0,5 (1 belge) | %0,5 |
| telefon (`ddd-ddd-dddd` benzeri) | %1,5 (3 belge) | %1,5 |
| `City, ST` (gerçek şehir + iki harfli eyalet) | %33,0 | %33,0 |
| yer tutucu `City` ve `State` sözcükleri | %96,5 | |
| `Company Name` yer tutucusu | %100,0 | |
| `http`, `www` veya `linkedin` | %5,0 | |

Veri seti anonimleştirilmiş görünüyor: ad ve iletişim bilgisi şablon yer tutucularla değiştirilmiş (`Company Name`, `City`, `State`), e-posta ve telefon neredeyse hiç yok. İlk boş olmayan satır ad değildir: belgelerin %99,5'inde tamamı büyük harf, medyan 2 sözcük (iş unvanı biçimi), %1'inde rakam var. Bu nedenle `contact.extract` içindeki "ilk satır ad" sezgisi bu veri setinde ad yerine iş unvanını yakalar (SPEC §6.3). Ad sezgisi sentetik örneklerle test edilmeli; benchmark'ta bu alanın değeri yok sayılmalı. Kalan e-posta ve telefon örnekleri tek tek incelenmedi (KVKK); yalnızca sayıldı.

## 7. Dil

`Languages` başlığı tam satır olarak %7,5 belgede, satır başında %9,0, düz sözcük olarak (`language` dahil) %19,0 belgede geçer. Türkçe başlık sözcükleri (`deneyim`, `eğitim`, `yetenek`, `hakkımda`, `özet`) belgelerin %0'ında. Dil filtresi benchmark'ta zaten kapalıdır (SPEC §7.2); bu bulgu o kararı doğrular.

## 8. T-03 için özet çıkarımlar

1. Bölümleme PDF çıktısı üzerinde yapılmalı. `Resume_str` düz metindir, başlıklar satır başında değildir.
2. Satırlar kırpılmalı; yalnızca boşluk içeren satırlar boş satır sayılmalı. Mojibake ve U+200B artıkları yok sayılmalı veya temizlenmeli.
3. Başlıklar tam satır eşleşmesiyle aranmalı. Yalnızca satır başı eşleşmesi %10'a varan belgede "Experience in ..." gibi cümleleri başlık sayar.
4. SPEC başlık listesi ≥2 tür hedefini %100 ile karşılıyor. Ancak `Accomplishments` (%35), `Highlights` (%32,5), `Professional Summary` (%22), `Additional Information` (%20) ve `Core Qualifications` (%13,5) tanınmayan sık başlıklardır.
5. `years_experience` için tire kuralı yetmez (%13,5). `to` ayracı, `MM/YYYY` ve `Month YYYY` biçimleri, `Current` bitişi ve satırlar arası eşleşme gerekir (%99). `Current` (%55) ve `Present` (%5,5) bitiş sözcükleri görülür; `Now` ve `Günümüz` görülmedi.
6. Deneyim girdilerini bölmek için tarih aralığı uygun bir ayraçtır. Deneyim başlığından sonraki ilk beş satırın en sık görülen kalıbı: unvan benzeri satır, başlangıç tarihi, `to`, bitiş tarihi veya `Current`, `Company Name` (altı en sık kalıp birlikte örneklemin çoğunu kapsar; kalıplar arasında unvan ve tarih sırası değişir).
7. `contact.extract` bu veri setinde neredeyse hiçbir şey bulamaz (e-posta %0,5, telefon %1,5) ve ad sezgisi iş unvanı bulur; birim testleri sentetik örneklerle yazılmalı.
8. Veri seti Türkçe içermiyor; Türkçe başlıklar yalnızca sentetik testlerle sınanabilir.
