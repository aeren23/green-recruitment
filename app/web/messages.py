"""Turkish UI strings that the Python code produces (templates hold their own text)."""

from app.models import JobStatus, SectionType
from app.privacy.reasons import StoredReasonCode

LOGIN_FAILED = "Erişim anahtarı yanlış. Lütfen tekrar deneyin."

STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.OPEN: "Açık",
    JobStatus.CLOSED: "Kapalı",
    JobStatus.FINALIZED: "Sonuçlandı",
}

WEIGHT_LABELS = {1: "1 (düşük)", 2: "2 (orta)", 3: "3 (yüksek)"}

TITLE_REQUIRED = "İlan başlığı boş bırakılamaz."
TITLE_TOO_LONG = "İlan başlığı en fazla {limit} karakter olabilir."
DESCRIPTION_TOO_LONG = "Açıklama en fazla {limit} karakter olabilir."
LANGUAGES_TOO_LONG = "Dil listesi en fazla {limit} karakter olabilir."
LOCATION_TOO_LONG = "Konum en fazla {limit} karakter olabilir."
MIN_YEARS_INVALID = (
    "Asgari deneyim 0 ile {limit} arasında bir sayı olmalıdır (örneğin 2 veya 2,5)."
)
REQUIREMENTS_REQUIRED = "En az bir gereksinim girmelisiniz."
REQUIREMENT_TOO_LONG = "Gereksinim {row}: metin en fazla {limit} karakter olabilir."
REQUIREMENT_WEIGHT_INVALID = "Gereksinim {row}: ağırlık {low} ile {high} arasında olmalıdır."

# Candidate flow (T-13).
JOB_NOT_FOUND = "Aradığınız ilan bulunamadı."
UPLOAD_UNREADABLE = (
    "PDF dosyası okunamadı. Dosya bozuk, boş ya da şifreli olabilir; "
    "lütfen şifresiz bir PDF yükleyin."
)
UPLOAD_TOO_SHORT = (
    "PDF'den yeterli metin okunamadı. Taranmış (görüntü halindeki) PDF'ler "
    "desteklenmiyor; lütfen metin içeren bir PDF yükleyin."
)
POSTING_CLOSED = "Bu ilan artık başvuru almıyor."
DUPLICATE_APPLICATION = (
    "Bu özgeçmişle bu ilana zaten başvurdunuz. Aynı ilana aynı özgeçmişle "
    "ikinci kez başvurulamaz."
)
UPLOAD_TOO_LARGE = "Dosya çok büyük. Başvuru isteği en fazla {limit} olabilir."
DRAFT_EXPIRED = (
    "Taslağınızın süresi doldu ve verileriniz silindi. "
    "Başvurmak için özgeçmişinizi yeniden yükleyebilirsiniz."
)
YEARS_INVALID = "Deneyim yılı 0 ile {limit} arasında bir sayı olmalıdır (örneğin 4 veya 4,5)."
SECTION_TYPE_INVALID = "Bölüm {row}: bölüm türü geçersiz."
SECTION_TOO_LONG = "Bölüm {row}: metin en fazla {limit} karakter olabilir."
TOO_MANY_SECTIONS = "En fazla {limit} bölüm gönderebilirsiniz."
FIELD_TOO_LONG = "{field} en fazla {limit} karakter olabilir."
FIELD_LANGUAGES = "Diller"
FIELD_LOCATION = "Konum"
FIELD_CONTACT_NAME = "Ad"
FIELD_CONTACT_EMAIL = "E-posta"
FIELD_CONTACT_PHONE = "Telefon"

SECTION_TYPE_LABELS = {
    SectionType.SUMMARY: "Özet",
    SectionType.EXPERIENCE: "Deneyim",
    SectionType.PROJECT: "Projeler",
    SectionType.EDUCATION: "Eğitim",
    SectionType.CERTIFICATION: "Sertifikalar",
    SectionType.SKILLS: "Beceriler",
}

LANGUAGE_LABELS = {
    "english": "İngilizce",
    "turkish": "Türkçe",
    "german": "Almanca",
    "french": "Fransızca",
    "spanish": "İspanyolca",
    "arabic": "Arapça",
    "russian": "Rusça",
    "italian": "İtalyanca",
    "chinese": "Çince",
    "japanese": "Japonca",
}

ELIMINATION_REASONS = {
    StoredReasonCode.LANG_MISSING: (
        "İlanın aradığı dillerin hiçbiri özgeçmişinizde yer almıyor."
    ),
    StoredReasonCode.LOCATION_MISMATCH: (
        "İlan uzaktan çalışmaya uygun değil ve özgeçmişinizdeki konum "
        "ilanın konumuyla uyuşmuyor."
    ),
    StoredReasonCode.MIN_YEARS: (
        "Özgeçmişinizdeki deneyim süresi, ilanın istediği asgari süreden kısa."
    ),
}

BYTES_PER_KILOBYTE = 1024
BYTES_PER_MEGABYTE = 1024 * 1024


def format_size(byte_count: int) -> str:
    """A byte count as Turkish UI text: whole megabytes when exact, else kilobytes."""
    if byte_count >= BYTES_PER_MEGABYTE and byte_count % BYTES_PER_MEGABYTE == 0:
        return f"{byte_count // BYTES_PER_MEGABYTE} MB"
    return f"{max(byte_count // BYTES_PER_KILOBYTE, 1)} KB"
UPLOAD_TOO_LARGE_HEADING = "Dosya çok büyük"
BACK_TO_POSTINGS = "İlanlara dön"
NOT_FOUND_HEADING = "Sayfa bulunamadı"
UPLOAD_AGAIN = "Özgeçmişimi yeniden yükle"
DRAFT_EXPIRED_HEADING = "Taslağın süresi doldu"
POSTING_CLOSED_HEADING = "İlan kapandı"
SECTION_REQUIRED = "En az bir bölümün metni dolu olmalıdır."
FORM_INCOMPLETE = (
    "Form eksik ya da bozuk gönderildi; hiçbir değişiklik kaydedilmedi. "
    "Lütfen bilgileri kontrol edip tekrar gönderin."
)

# Posting lifecycle (T-14). Notices travel as a whitelisted query flag, never as text.
NOTICE_CLOSED = "closed"
NOTICE_FINALIZED = "finalized"
POSTING_NOTICES = {
    NOTICE_CLOSED: "İlan kapatıldı. Sıralanan adaylar için karar kaydı yazıldı.",
    NOTICE_FINALIZED: "İlan sonuçlandırıldı. Kısa listedeki başvurular silindi.",
}
FINALIZE_REQUIRES_CLOSED = "İlan sonuçlandırılamadı: önce ilanı kapatmalısınız."

# Ranked list, matrix and shortlisting (T-15).
APPLICATION_NOT_FOUND = "Aradığınız başvuru bulunamadı."
POSTING_NOT_FOUND = "Aradığınız ilan bulunamadı."
SHORTLIST_REQUIRES_RANKED = (
    "Bu başvuru kısa listeye alınamaz: yalnızca sıralanmış başvurular alınabilir."
)
SHORTLIST_REQUIRES_OPEN_POSTING = "Bu başvuru kısa listeye alınamadı: ilan artık açık değil."
SHORTLIST_FAILED_HEADING = "Kısa listeye alınamadı"
BACK_TO_EMPLOYER_POSTINGS = "İlanlara dön"
CANDIDATE_LABEL = "Aday #{application_id}"
FILTER_LANGUAGES = "Diller"
FILTER_LOCATION = "Konum"
FILTER_REMOTE = "Uzaktan çalışma"
FILTER_MIN_YEARS = "Asgari deneyim"
FILTER_REMOTE_YES = "Uygun"
FILTER_REMOTE_NO = "Uygun değil"
FILTER_MIN_YEARS_VALUE = "{years} yıl"
MATCH_EVIDENCE = "Kanıt"
MATCH_CLAIM = "Beyan"
MATCH_OTHER = "Diğer"
