# Demo CV'leri

Tamamı kurgusaldır (example.com e-postaları, sahte telefonlar). Üretmek için:
`.venv/Scripts/python docs/demo/make_demo_cv.py`. "Present" bitişleri ayrıştırma
tarihine kırpılır, bu yüzden yıllar zamanla artar (ölçüm: 2026-11-03).

Hepsi "Backend Developer" ilanına başvurur (demo filtreleri: İngilizce zorunlu,
Denizli, uzaktan çalışma serbest; en az 2 yıl). Skorlar gerçek gömme modeliyle,
geçici bir veritabanında ölçülmüştür.

| Dosya | Persona | Neyi gösterir | Beklenen sonuç |
|---|---|---|---|
| `demo-cv.pdf` | Ada, 5 yıl backend geliştirici | Anahtar kelimeler ve kanıt birlikte; referans aday | Sıra 1, skor 0.855 (4.8 yıl) |
| `demo-cv-paraphrase.pdf` | Deniz, Spring Boot / PostgreSQL / Kubernetes | "REST", "API", "SQL", "backend", "database" kelimeleri geçmez; anlamsal eşleşme kelime aramasının ötesine geçer | Sıra 2, skor 0.840 (5.8 yıl), Ada'nın hemen arkasında |
| `demo-cv-frontend.pdf` | Elif, React / CSS arayüz geliştirici | Kısmi uyum, orta sıra | Sıra 3, skor 0.818 (4.8 yıl) |
| `demo-cv-keywords.pdf` | Mert, pazarlama / satış; Skills bölümünde REST API, SQL, Python, Java, Docker, Linux | Beyan (Skills) kanıt (Experience) olmadan yetmez; anahtar kelime doldurma ödüllendirilmez | Sıra 4 (sıralananların en düşüğü), skor 0.780 (7.8 yıl) |
| `demo-cv-junior.pdf` | Can, yaklaşık 1,8 yıl deneyim | Sert filtre MIN_YEARS ve karar kaydı | Elenir (`MIN_YEARS`), sıralamada görünmez |

Not: gömme modeli benzerlikleri dar bir aralığa sıkıştırır (0.78 - 0.85), bu
yüzden sıralama doğru ama skor farkları küçüktür. Keywords CV'sinde en iyi
eşleşme Skills yerine pazarlama deneyimi bölümünden gelir; Skills bölümü beyan (Beyan)
olarak sayıldığı ve deneyim kanıtı olmadığı için sırayı yükseltmez.
