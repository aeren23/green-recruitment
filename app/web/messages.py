"""Turkish UI strings that the Python code produces (templates hold their own text)."""

from app.models import JobStatus

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
