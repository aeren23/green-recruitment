"""Renders a `Report` as Markdown (Turkish headings, English metric identifiers).

Only the numbers of the `Report` are used, so `report.md` and `report.json`
cannot disagree. The fixed sentences at the end state the method limits that
hold for every benchmark run (CPU counter, RAM model, M, filters, baseline).
"""

import json
from collections.abc import Sequence

from bench.quality_update import QualityUpdate
from bench.report_metrics import METRICS_BY_KEY
from bench.report_model import (
    CriterionResult,
    ExtractComparison,
    GroupConditions,
    GroupReport,
    K1Evaluation,
    MetricRow,
    Report,
    RunCondition,
    StageRow,
    Summary,
)
from bench.report_stages import APPROXIMATE_BELOW_MS
from bench.scenario import JSON_COMPACT_SEPARATORS

NOT_AVAILABLE = "n/a"
MILLI = 1000.0
PERCENT = 100.0
PASS_LABEL = "PASS"
FAIL_LABEL = "FAIL"
NOT_EVALUABLE_LABEL = "N/A (a number is missing)"
STAGE_ENERGY_DECIMALS = 3
STAGE_DURATION_DECIMALS = 2
STAGE_COUNT_DECIMALS = 1
MS_DECIMALS = 1
SHARE_DECIMALS = 3
IDLE_DECIMALS = 2
NO_QUALITY_NOTES = "Dışlanan ilan veya eşleşmemiş koşu yok."
QUALITY_NOTES_HEADING = "## Kalite notları"

TITLE = "# Green Hiring Pipeline: Benchmark Raporu"
INTRO = (
    "Bu rapor `python -m bench compare` ile üretildi. Değerler `data/bench/bench.db` "
    "içindeki koşulardan gelir; tablolar ortalama ± örnek standart sapma verir "
    "(tek koşu: `± n/a`). Fark yüzdesi = (green − baseline) / baseline × 100."
)
WARNING_NOT_FINAL = (
    "**WARNING: bu grup geçerli bir K1 ölçümü DEĞİLDİR.** Aşağıdaki koşullardan en az "
    "biri nihai ölçüm koşullarına uymuyor (güç planı 'Dengeli' olmalı, OneDrive "
    "çalışmamalı, git temiz olmalı). Sayılar yalnızca işlevsel koşu değerleridir; K1 "
    "kararında kullanılmamalıdır."
)
CONDITIONS_OK = "Tüm koşular nihai ölçüm koşullarına uyuyor (Dengeli plan, OneDrive kapalı, git temiz)."
D6_PENALIZES_GREEN = (
    "`extract` çağrı başına maliyeti green'de baseline'dan yüksek: izleyici ve NVML "
    "ek yükü (D6) green'i cezalandırır, yani bu aşamada karşılaştırma green aleyhine "
    "temkinlidir."
)
D6_NO_PENALTY = (
    "`extract` çağrı başına maliyeti green'de baseline'dan yüksek değil: bu aşamada "
    "green aleyhine bir sapma görülmüyor."
)
D6_NO_DATA = "`extract` çağrı başına maliyeti hesaplanamadı (aşama verisi eksik)."
FIXED_NOTES = (
    "CPU enerjisi bir donanım sayacından gelir (Windows Energy Meter Interface, RAPL "
    "sayacı); TDP tahmini değildir. Paket sayacı ekran, SSD ve Wi-Fi gibi bileşenleri "
    "kapsamaz.",
    "RAM enerjisi sabit güç modelidir (yüke bağlı değildir); ölçülmüş bir değer değildir.",
    "SCI = co2_g / applications. Gömülü karbon (M) ölçüm dışıdır ve SCI'ya dahil "
    "edilmemiştir. R, koşuya gönderilen tüm başvurulardır (engellenen tekrarlar dahil).",
    "Benchmark'ta dil ve konum filtreleri kapalıdır; sert filtre yalnızca `min_years` "
    "üzerinde çalışır (SPEC 7.2). CV'lerin ayrıştırılmasında sabit referans tarihi "
    "kullanılır (SPEC 7.3).",
    "Baseline küçük bir 3B modelle puanlanır; bu yüzden `overlap_at_k` yardımcıdır, "
    "ana kalite ölçütü `precision_at_k` dir (SPEC 7.7).",
    "`model_load` aşaması toplamlara dahil değildir, ayrı satırda gösterilir.",
    "100 ms'den kısa aşamaların (çağrı başına ortalama) aşama bazlı enerjisi "
    "yaklaşıktır (D1); toplam enerji esas alınmalıdır.",
)


def format_number(value: float | None, decimals: int) -> str:
    """A number with fixed decimals, or `n/a`."""
    return NOT_AVAILABLE if value is None else f"{value:.{decimals}f}"


def format_summary(summary: Summary, decimals: int) -> str:
    """`mean ± sd`; `± n/a` for a single run, `n/a` without values."""
    if summary.mean is None:
        return NOT_AVAILABLE
    spread = NOT_AVAILABLE if summary.sd is None else f"{summary.sd:.{decimals}f}"
    return f"{summary.mean:.{decimals}f} ± {spread}"


def format_difference(value: float | None) -> str:
    """A signed percentage, or `n/a`."""
    return NOT_AVAILABLE if value is None else f"{value:+.1f}%"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def scenario_label(scenario: dict[str, object]) -> str:
    """The scenario as compact JSON."""
    return json.dumps(scenario, sort_keys=True, separators=JSON_COMPACT_SEPARATORS)


def metric_cells(row: MetricRow) -> list[str]:
    """The four cells of a main-table row."""
    spec = METRICS_BY_KEY[row.key]
    return [
        f"`{row.key}` ({spec.unit})",
        format_summary(row.baseline, spec.decimals),
        format_summary(row.green, spec.decimals),
        format_difference(row.difference_pct),
    ]


def render_main_table(rows: Sequence[MetricRow]) -> list[str]:
    """The main table (metric, baseline, green, difference %)."""
    header = ["metric", "baseline", "green", "difference %"]
    return table(header, [metric_cells(row) for row in rows])


def stage_cells(row: StageRow) -> list[str]:
    """The cells of one stage-table row."""
    note = ""
    if not row.counted_in_totals:
        note = "toplama dahil değil"
    elif row.approximate:
        note = f"yaklaşık (< {APPROXIMATE_BELOW_MS:.0f} ms/çağrı)"
    return [
        row.stage,
        format_number(row.energy_wh * MILLI, STAGE_ENERGY_DECIMALS),
        format_number(row.duration_s, STAGE_DURATION_DECIMALS),
        format_number(row.count, STAGE_COUNT_DECIMALS),
        format_number(row.ms_per_call, MS_DECIMALS),
        note,
    ]


def render_stage_table(title: str, rows: Sequence[StageRow]) -> list[str]:
    """One pipeline's per-stage breakdown."""
    header = ["stage", "energy (mWh)", "duration (s)", "count", "ms / call", "note"]
    return [f"#### {title}", "", *table(header, [stage_cells(row) for row in rows]), ""]


def criterion_value(criterion: CriterionResult, value: float | None) -> str:
    """A criterion number with the precision of the metric it comes from."""
    return format_number(value, METRICS_BY_KEY[criterion.key].decimals)


def verdict_label(passed: bool | None) -> str:
    """PASS, FAIL or the missing-number label."""
    if passed is None:
        return NOT_EVALUABLE_LABEL
    return PASS_LABEL if passed else FAIL_LABEL


def criterion_cells(criterion: CriterionResult) -> list[str]:
    """The cells of one K1 row, with the exact numbers used."""
    return [
        f"`{criterion.key}`",
        criterion.rule,
        criterion_value(criterion, criterion.baseline_value),
        criterion_value(criterion, criterion.green_value),
        criterion_value(criterion, criterion.threshold),
        verdict_label(criterion.passed),
    ]


def render_k1(k1: K1Evaluation) -> list[str]:
    """The K1 section (criteria of plan 8.2)."""
    header = ["metric", "rule", "baseline", "green", "threshold", "result"]
    rows = [criterion_cells(criterion) for criterion in k1.criteria]
    overall = verdict_label(k1.all_passed)
    return [
        "### K1 değerlendirmesi",
        "",
        *table(header, rows),
        "",
        f"K1 sonucu (dört ölçütün hepsi): **{overall}**",
        "",
    ]


CONDITION_HEADER = (
    "run",
    "pipeline",
    "power plan",
    "OneDrive running",
    "git",
    "git dirty",
    "idle_power_w",
    "GPU stage-sum / stream (D4)",
    "unstaged share",
    "extract ms/call",
)


def condition_cells(run: RunCondition) -> list[str]:
    """The cells of one run-condition row."""
    return [
        str(run.run_id),
        run.pipeline,
        run.power_plan,
        run.onedrive_running,
        run.git_revision,
        run.git_dirty,
        format_number(run.idle_power_w, IDLE_DECIMALS),
        format_number(run.gpu_stage_sum_over_stream, SHARE_DECIMALS),
        format_number(run.unstaged_share, SHARE_DECIMALS),
        format_number(run.extract_ms_per_call, MS_DECIMALS),
    ]


def render_run_conditions(conditions: GroupConditions) -> list[str]:
    """One row per run: plan, OneDrive, git, idle power, D4 ratio, unstaged share."""
    return table(CONDITION_HEADER, [condition_cells(run) for run in conditions.runs])


def render_warnings(conditions: GroupConditions) -> list[str]:
    """The non-final-conditions WARNING with one line per reason, or the OK line."""
    if conditions.valid_for_k1:
        return [CONDITIONS_OK, ""]
    lines = [WARNING_NOT_FINAL, ""]
    for run in conditions.runs:
        lines.extend(f"- run {run.run_id} ({run.pipeline}): {text}" for text in run.warnings)
    return [*lines, ""]


def render_extract(extract: ExtractComparison) -> list[str]:
    """The D6 extract per-call comparison."""
    if extract.penalizes_green is None:
        return [D6_NO_DATA, ""]
    sentence = D6_PENALIZES_GREEN if extract.penalizes_green else D6_NO_PENALTY
    return [
        f"- extract ms/call: baseline {format_number(extract.baseline_ms, MS_DECIMALS)}, "
        f"green {format_number(extract.green_ms, MS_DECIMALS)}, "
        f"fark (green − baseline) {format_number(extract.difference_ms, MS_DECIMALS)}",
        f"- {sentence}",
        "",
    ]


def render_conditions(conditions: GroupConditions) -> list[str]:
    """The conditions and caveats section of one group."""
    methods = ", ".join(conditions.cpu_methods) or NOT_AVAILABLE
    return [
        "### Koşullar ve uyarılar",
        "",
        *render_warnings(conditions),
        *render_run_conditions(conditions),
        "",
        f"- CPU ölçüm yöntemi (notlardan): {methods}",
        *render_extract(conditions.extract),
    ]


def render_group(index: int, group: GroupReport) -> list[str]:
    """One scenario group: tables, K1 and conditions."""
    return [
        f"## Senaryo grubu {index}",
        "",
        f"- scenario_json: `{scenario_label(group.scenario)}`",
        f"- green koşuları: {list(group.green_run_ids)}, baseline koşuları: "
        f"{list(group.baseline_run_ids)}",
        "",
        "### Ana tablo",
        "",
        *render_main_table(group.metrics),
        "",
        "### Aşama bazlı enerji dağılımı",
        "",
        *render_stage_table("baseline", group.stages.baseline),
        *render_stage_table("green", group.stages.green),
        *render_k1(group.k1),
        *render_conditions(group.conditions),
    ]


def render_exclusions(report: Report) -> list[str]:
    """Excluded runs and groups that lack one pipeline."""
    lines = ["## Dışarıda bırakılan koşular", ""]
    if not report.excluded_runs and not report.skipped_groups:
        return [*lines, "Yok.", ""]
    lines.extend(
        f"- run {run.run_id} ({run.pipeline}): {run.reason}" for run in report.excluded_runs
    )
    lines.extend(
        f"- scenario `{scenario_label(group.scenario)}`: karşılaştırılamaz "
        f"(green {list(group.green_run_ids)}, baseline {list(group.baseline_run_ids)})"
        for group in report.skipped_groups
    )
    return [*lines, ""]


def quality_note_lines(quality: QualityUpdate) -> list[str]:
    """One line per quality-update finding."""
    lines = [
        f"- run {run_id}: precision@k dışında bırakılan ilanlar: {', '.join(titles)}"
        for run_id, titles in sorted(quality.precision_excluded.items())
    ]
    lines.extend(
        f"- scenario `{item.scenario}`: green {item.green_runs}, baseline "
        f"{item.baseline_runs} koşu; overlap@k yalnızca mevcut çiftlerle hesaplandı"
        for item in quality.mismatches
    )
    lines.extend(f"- run {run_id}: rankings dosyası yok" for run_id in quality.runs_without_rankings)
    lines.extend(f"- run {run_id}: eşleşmemiş, overlap@k yok" for run_id in quality.unpaired_run_ids)
    return lines


def render_quality_notes(quality: QualityUpdate | None) -> list[str]:
    """What the quality update excluded or could not pair."""
    if quality is None:
        return []
    lines = quality_note_lines(quality) or [NO_QUALITY_NOTES]
    return [QUALITY_NOTES_HEADING, "", *lines, ""]


def render_markdown(report: Report) -> str:
    """The whole report as Markdown."""
    lines = [TITLE, "", INTRO, "", f"Kalite kesim noktası k = {report.k}.", ""]
    for index, group in enumerate(report.groups, start=1):
        lines.extend(render_group(index, group))
    if not report.groups:
        lines.extend(["Karşılaştırılabilir senaryo grubu yok.", ""])
    lines.extend(render_exclusions(report))
    lines.extend(render_quality_notes(report.quality))
    lines.extend(["## Sabit notlar", "", *(f"- {note}" for note in FIXED_NOTES), ""])
    return "\n".join(lines)
