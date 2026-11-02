# SPEC: Green Hiring Pipeline

## 1. Goal and claim

In digital hiring, application management and pre-screening produce unnecessary storage and compute load: PDFs are stored indefinitely, the same CV is reprocessed on every application, every CV is read by a large language model, and personal data is kept longer than needed.

The claim of this project: **A tiered, data-minimal pipeline, on the same data and the same machine, spends less energy, less storage and fewer LLM calls per application than a conventional pipeline, while keeping shortlist quality acceptable.**

This claim is measured by `bench/`. The demo application shows the same logic live. The numbers in the report come from the benchmark, not from the demo.

Hiring stages touched: 2 (job posting: requirement definition), 4 (applicant management), 6 (candidate selection), and the process as a whole (KVKK, retention).

## 2. Scope

**In scope**

- Employer: creating job postings (hard filters and weighted requirements), a ranked candidate list, the requirement matrix, shortlisting, closing a posting.
- Candidate: viewing open postings, uploading a CV, previewing, correcting and confirming the parsed information, viewing application status.
- Cache: reuse of parse results and section embeddings, blocking repeat applications to the same posting.
- KVKK: data minimization, a separate contact table, reason-coded decision records, reference-counted deletion, retention.
- Green panel: live counters, the latest benchmark comparison, page weight.
- Benchmark CLI: baseline and green pipelines, per-stage energy, storage, LLM calls and quality metrics.

**Out of scope**

- Real authentication. Employer pages live under `/employer`, and a simple cookie check against `EMPLOYER_TOKEN` in `.env` is enough.
- Sending email or SMS.
- OCR and scanned-PDF support. A PDF from which no text can be extracted is rejected.
- Multiple services, message queues, Docker orchestration.
- Recommending a candidate to postings they did not apply to (KVKK purpose limitation).
- A multilingual UI. The UI is in Turkish.

## 3. Demo scenario (5 minutes)

1. The employer opens the "Backend Developer" posting: hard filters and weighted requirements are visible.
2. A candidate uploads a CV. The preview screen shows the parsed sections, years of experience, languages and contact information. The candidate corrects and confirms. The screen shows "Dosyanız saklanmadı" (Your file was not stored).
3. The candidate uploads the same CV to the "Accountant" posting. The preview comes up pre-filled from the earlier confirmed data and shows "Önceki başvurunuzdan alındı, yeniden işlenmedi" (Taken from your previous application, not reprocessed).
4. The candidate uploads the same CV again to "Backend Developer" and is blocked (`DUPLICATE_APPLICATION`).
5. The employer sees the ranked list anonymously. They open a candidate's matrix: for each requirement, the best-matching section, its type (evidence/claim) and its score. When the employer shortlists the candidate, the contact information becomes visible.
6. The employer closes the posting. Retention runs and the panel counters update.
7. Green panel: baseline vs. green comparison table, live counters, page weight.

## 4. Architecture and stack

Single process: FastAPI + server-side rendering with Jinja2 + SQLite. The embedding model is loaded in-process. The LLM is called over HTTP on a locally running Ollama.

| Layer | Choice | Note |
|---|---|---|
| Web | FastAPI, Jinja2, plain HTML forms | no SPA, no build |
| DB | SQLite, SQLAlchemy 2.x | no separate DB server |
| PDF | pypdf | no OCR |
| Embedding | `intfloat/multilingual-e5-small`, sentence-transformers | cuda/cpu via `EMBED_DEVICE` |
| Vector | numpy, embeddings as `float32` BLOB | no ANN index needed (§6.5) |
| LLM | Ollama, `qwen2.5:3b` | local, measurable |
| Measurement | CodeCarbon | `tracking_mode="machine"` |

Why no FAISS or separate vector DB: an application's score comes from a similarity matrix between roughly 4 to 10 requirements and roughly 10 to 30 sections. That is a small matrix multiplication and adds no extra dependency or process.

SQLite pragmas on every connection: `foreign_keys=ON`, `secure_delete=ON`.

## 5. Data model

All time fields are UTC. `id` fields are integer PKs.

**JobPosting**
- `title`, `description`
- `status`: `open` | `closed` | `finalized`
- `closes_at` (nullable, informational only; closing is manual), `closed_at`, `finalized_at`, `created_at`
- Hard filters: `required_languages` (JSON list, may be empty), `location` (nullable), `remote_ok` (bool), `min_years` (nullable float)
- `relevant_categories` (JSON list): used only for the benchmark quality metric, not shown in the UI

**Requirement**
- `job_id`, `text` (English, since the CVs are English), `weight` (1 to 3), `embedding` (BLOB), `model_version`

**CandidateProfile**: the parse cache itself
- `raw_hash` (unique, HMAC-SHA256 hex)
- `profile_json`: the confirmed structure (§6.3). Contains no contact information.
- `years_experience` (nullable), `languages` (JSON list), `location` (nullable): also kept as columns for the hard filters
- `parser_version`, `created_at`, `confirmed_at`, `last_used_at`

**Section**
- `profile_id` (FK, cascade), `type`: `experience` | `project` | `education` | `certification` | `summary` | `skills`
- `ordinal`, `text`, `text_hash` (SHA-256 of `model_version + normalized text`), `embedding` (BLOB), `model_version`

**CandidateContact**
- `profile_id` (FK, unique, cascade), `name`, `email`, `phone`

**Application**
- `job_id`, `profile_id`, `raw_hash`
- `status`: `draft` | `ranked` | `shortlisted`
- `score` (nullable float, raw score), `matrix_json` (nullable), `summary` (nullable, LLM summary)
- `draft_token` (unique, used in URLs), `created_at`, `confirmed_at`, `decided_at`
- `unique(job_id, raw_hash)`

**DecisionRecord**: the only trace left after a profile is deleted
- `job_id`, `raw_hash`, `reason_code`, `decided_at`, `expires_at`
- `application_token` (nullable): copy of the deleted application's `draft_token`, so the status page still works after the Application is gone

**GreenCounter**: a single-row counter table for the demo panel
- `pdf_bytes_not_stored`, `parse_cache_hits`, `embed_cache_hits`, `embeddings_computed`, `llm_calls`, `llm_calls_avoided`, `records_deleted`
- `llm_calls_avoided` = confirmed applications − `llm_calls`

**BenchmarkRun**
- `pipeline` (`baseline` | `green`), `scenario_json` (N, jobs_per_cv, resubmit_rate, seed, top_k, models)
- `energy_wh_gross`, `energy_wh_net`, `co2_g`, `duration_s`, `idle_power_w`
- `energy_wh_gross` and `duration_s` are sums over the stages, excluding `model_load`
- `storage_bytes_active`, `storage_bytes_after_retention`
- `llm_calls`, `llm_prompt_tokens`, `llm_completion_tokens`, `parse_cache_hits`, `embed_cache_hits`
- `precision_at_k`, `overlap_at_k`, `measurement_notes` (CPU measurement method, device, CodeCarbon version)

**StageMeasurement**
- `run_id`, `stage`, `energy_wh`, `duration_s`, `count`

Benchmark tables are kept in `data/bench/bench.db`, not in the demo DB. The green panel opens this file read-only.

## 6. Green pipeline

### 6.1 Flow

```
upload (in memory) → extract_text → normalize → raw_hash
  → does (job_id, raw_hash) exist in Application, or in a non-expired DecisionRecord?
       Application with status draft: redirect to that draft, stop
       otherwise: DUPLICATE_APPLICATION, stop
  → does CandidateProfile(raw_hash) exist?
       yes: build a draft from profile_json, parse is skipped (parse_cache_hits++)
       no:  sections.parse + contact.extract → new profile (confirmed_at empty)
  → Application(status=draft, draft_token) → preview form
  → candidate confirms → profile_json, columns and contact are updated
       if a section's text_hash matches, its row and id are kept and its embedding is reused (embed_cache_hits++)
       the rest are embedded (embeddings_computed++)
       if any section changed, the profile's `ranked` applications are re-scored and their postings' top-k re-checked
  → hard filter (current posting only) → if failed: DecisionRecord + Application deleted + profile refcount (§8)
  → scoring (matrix) → status=ranked
  → if the top-k for the posting changed, an LLM summary only for the new top-k candidates
```

### 6.2 Normalization and hash

`normalize(text)`: Unicode NFKC → casefold → turn punctuation into spaces, except `+` and `#` (C++ and C# are preserved) → collapse all whitespace to a single space → strip.

`raw_hash = HMAC-SHA256(HASH_SECRET, normalize(extracted_text))`. The hash is taken from the raw text, not from the candidate's corrections. This way a re-exported version of the same PDF gives the same hash.

### 6.3 Section splitting (rule-based, no LLM)

A line is a heading only if the whole line, after trimming and without a trailing `:`, equals a heading below. The match is case-insensitive with Turkish-aware folding (`DENEYİM` matches `deneyim`), so a sentence such as "Experience in Python" is not a heading. The Turkish headings below are matched literally against Turkish CVs and must stay in Turkish:

| type | headings |
|---|---|
| summary | summary, profile, objective, about, professional summary, executive profile, career overview, career focus, professional profile, accomplishments, additional information, özet, hakkımda |
| skills | skills, technical skills, core competencies, qualifications, skill highlights, highlights, core qualifications, yetenekler, beceriler, yetkinlikler |
| experience | experience, work experience, work history, professional experience, employment, deneyim, iş deneyimi |
| project | projects, personal projects, projeler |
| education | education, education and training, eğitim |
| certification | certifications, certificates, licenses, sertifikalar |
| languages | languages, diller (not embedded, goes into the `languages` column) |
| ignored | interests, affiliations, professional affiliations, personal information (text is neither stored nor embedded) |

Rules:
- Text before the first heading becomes `summary`. Lines without a letter or digit, mojibake residue and zero-width characters are noise and are dropped, except a lone dash line, which may be the separator of a date range (pypdf puts it on its own line).
- `experience` and `project` sections are split into entries at the start of each date range. A short title-like line (at most 8 words, no terminal sentence punctuation) directly before the date range belongs to the new entry. Lines holding only whitespace are never separators; a real blank line separates entries only in a section that has no dates. Each entry becomes its own `Section` row.
- An entry longer than 200 words is split into 200-word chunks (the e5-small limit is 512 tokens).
- `skills` is kept as a single section.
- `years_experience`: date ranges in experience entries only (education dates are not counted) are extracted. Separators are `-`, en dash, `to` and `through`; formats are `YYYY`, `MM/YYYY` and `Month YYYY`; end words are `Present`, `Current`, `Now`, `Günümüz`. A range may span several lines. Months are used when present; a year-only start or end means January of that year, and an open end means the month of `reference_date`. Intervals are clipped to `reference_date`, a range whose end precedes its start is ignored, overlapping ranges are merged, and the total months are divided by 12 and rounded to one decimal. If no range is found, `None`.
- `reference_date`: the parser takes it as an argument. The web layer passes today's date; the benchmark passes the fixed `BENCH_CV_REFERENCE_DATE` (§10), so results are reproducible and a `Current` end in a dataset from about 2018 does not inflate experience.
- `location` is not parsed: the parser leaves it null and the candidate enters it in the preview.
- `languages` entries are mapped through a small TR→EN dictionary (for example `ingilizce` → `english`) so they match `required_languages`.
- If no heading is found, the whole text becomes a single `summary` section and a warning is shown to the candidate in the preview.
- `contact.extract`: email and phone are extracted with regex, and a first-line heuristic is used for the name. These values are removed from section texts (blind screening).

`profile_json` schema:

```json
{
  "parser_version": "1",
  "sections": [{"type": "experience", "ordinal": 0, "text": "..."}],
  "years_experience": 4.5,
  "languages": ["english"],
  "location": "denizli"
}
```

### 6.4 Hard filters

Main principle: **"Unknown" is not a reason for elimination.** If a field is empty, the candidate passes the filter and is ranked in the list with a "missing information" marker.

| Filter | Elimination condition | reason_code |
|---|---|---|
| Language | none of `required_languages` appears in the candidate's languages, and the candidate's language list is not empty | `LANG_MISSING` |
| Location | `remote_ok=false`, the posting has a location, the candidate has a location, and they differ | `LOCATION_MISMATCH` |
| Experience | `min_years` is set, `years_experience` is set, and `years_experience < min_years` | `MIN_YEARS` |

Filters run in SQL over the profile's columns. No ML is used.

### 6.5 Scoring

E5 rule: requirement texts are embedded with the `"query: "` prefix and section texts with the `"passage: "` prefix. Since `normalize_embeddings=True`, cosine similarity equals the inner product.

```
R: (m × d) requirement embeddings, w: (m,) weights
S: (n × d) section embeddings, α: (n,) section-type coefficients
sim = R @ S.T                      # (m × n)
adj = sim * α[None, :]
best_j = argmax(adj, axis=1)       # best section for each requirement
score = Σ w_i · adj[i, best_j[i]] / Σ w_i
```

Section-type coefficients (configurable via `.env`): experience 1.0, project 1.0, certification 0.8, education 0.8, summary 0.7, skills 0.6. The types counted as evidence are experience and project; the type counted as a claim is skills.

`matrix_json`: `[{"requirement_id", "section_id", "section_type", "sim", "adjusted", "evidence": true|false}]`

Calibration: e5 similarities cluster in a narrow band (mostly between 0.7 and 0.9). Ranking uses the raw score. In the UI, a value min-max scaled to 0–100 among the ranked candidates of the same posting is shown, and it is labeled as a relative score.

`scoring/matrix.py` takes only numpy and returns. It contains no DB or model knowledge.

### 6.6 LLM (top-k only)

- Model `qwen2.5:3b`, `temperature=0`, `num_ctx=4096`.
- `TOP_K=5` (per posting). The summary prompt contains the requirement list and **only the text of the sections selected in the matrix**, not the whole CV.
- Output is at most 2 sentences, `num_predict=120`.
- `prompt_eval_count` and `eval_count` from the Ollama response are added to the token counters.
- If Ollama is unreachable, the summary field stays empty and the pipeline keeps running.

## 7. Benchmark

### 7.1 Dataset

Kaggle Resume Dataset (`snehaanbhawal/resume-dataset`). Expected structure: `Resume/Resume.csv` (ID, Resume_str, Resume_html, Category) and `data/data/<CATEGORY>/<ID>.pdf`. The structure is verified in T-01. If it differs, the SPEC is updated. The dataset goes under `data/raw/` and does not enter the repo.

### 7.2 Postings and category mapping

| Posting | relevant_categories | min_years | Requirements (weight) |
|---|---|---|---|
| Backend Developer | INFORMATION-TECHNOLOGY | 2 | Building REST APIs and backend services (3); Relational databases and SQL (2); Python or Java programming (2); Linux, cloud or container infrastructure (1) |
| Accountant | ACCOUNTANT, FINANCE | 3 | Financial statements and general ledger accounting (3); Tax preparation and compliance (2); Accounts payable and receivable (2); Excel and accounting software (1) |
| HR Specialist | HR | 2 | Recruitment and talent acquisition (3); Employee relations and HR policies (2); Payroll and benefits administration (2); HRIS systems (1) |

In the benchmark, the language and location filters are off, because these fields are not reliable in the dataset. In the benchmark the hard filter runs on `min_years` only. This is stated in the report.

### 7.3 Scenario

- Sampling: N CVs, stratified with a fixed `seed`. One quarter IT, one quarter ACCOUNTANT+FINANCE, one quarter HR, one quarter distractors from other categories.
- Each CV applies to `jobs_per_cv` postings (default 2, chosen randomly among the 3 postings).
- `resubmit_rate` (default 0.1): this share of applications are repeat applications with the same CV to the same posting.
- In the green pipeline the preview step is auto-confirmed (the candidate makes no corrections).
- CV parsing uses the fixed `reference_date` `BENCH_CV_REFERENCE_DATE` (§10, default 2018-01-01) instead of today, because the dataset's CVs date from about 2018 or earlier and a `Current` end would otherwise inflate `years_experience` by roughly eight years. The report states this as a limitation.
- Scenario parameters are stored as `scenario_json`. Because the cache gain depends on these parameters, the report contains at least two scenarios: `jobs_per_cv=1, resubmit_rate=0` and the default scenario.

### 7.4 Pipelines and stage names

**Baseline** (conventional pipeline)
- `store_pdf`: for every application, the PDF is copied under `data/bench/baseline_store/`.
- `extract`: for every application, text is extracted from scratch, no cache.
- `llm_score`: for every application, the CV's full text (first `MAX_CV_CHARS=6000` characters) and the requirement list go to the LLM. Output is JSON: `{"score": 0-100, "reason": "..."}`. Options: `num_predict=120` and the same `num_ctx` and `temperature` as §6.6, so both pipelines run under equal conditions.
- `rank`: ranked by LLM score. Repeat applications are processed too. Nothing is deleted.

**Green pipeline**: the flow in §6, with the same modules
- Stages: `extract`, `hash`, `parse`, `filter`, `embed`, `score`, `llm_summary`, `retention`.

Model loading (`model_load`) is measured as a separate stage and is not included in total energy; it is shown on a separate line in the report. In both pipelines the LLM is loaded into memory with a warm-up call before the run.

### 7.5 Measurement protocol

- `OfflineEmissionsTracker(tracking_mode="machine", country_iso_code="TUR", measure_power_secs=1, save_to_file=False, log_level="error")` (`country_iso_code` is accepted only by the offline tracker, which also avoids a network geolocation lookup). The CodeCarbon task API (`start_task` / `stop_task`) is used per stage. The API names are verified against the installed version in T-05.
- `process` mode is not used. Because Ollama is a separate process, LLM energy is not measured in that mode.
- Before measuring, idle power (W) is measured with `bench idle --seconds 120`. `energy_wh_net = energy_wh_gross − idle_power_w × duration_s / 3600`. `energy_wh_gross` and `duration_s` are sums over the measured stages, excluding `model_load`, so the idle correction uses the same duration. The report gives gross and net values together.
- Laptop plugged in, other applications closed, same power plan. Each run is repeated 3 times; mean ± standard deviation is reported.
- The `measurement_notes` field records the CPU measurement method (RAPL / TDP estimate), the GPU name, the CodeCarbon version and the embedding device.
- SCI: `(E · I + M) / R`, where R = number of applications. M (embodied carbon) is outside the measurement and the report states this explicitly.

### 7.6 Storage

- Baseline: total size of `baseline_store/`, plus the JSON file of texts and LLM outputs.
- Green pipeline: the size of `bench_green.db` is measured at two points. `storage_bytes_active` after all applications are processed, then `storage_bytes_after_retention` after all postings are closed and retention and `VACUUM` have run.

### 7.7 Quality

- `precision@k` (k=`QUALITY_K`=10): for each posting, the share of the final ranking's top k candidates whose category is in `relevant_categories`. Averaged over postings.
- For quality metrics, repeat applications are deduplicated by CV id. If a posting has fewer than k candidates, the denominator is the candidate count.
- `overlap@k`: the intersection of the green pipeline's top-k list and the baseline's top-k list, divided by min(k, unique green candidates, unique baseline candidates), which equals k whenever both rankings have at least k unique candidates (SC-21: dividing by k would penalise green when the hard filter leaves fewer than k candidates). Because the baseline is scored by a small 3B model, this metric is auxiliary; the main quality metric is precision@k.
- Candidates eliminated by the hard filter do not appear in the green ranking. So precision@k includes the effect of the filter as well as of the ranking.

### 7.8 Output

The `bench compare` command groups runs by identical `scenario_json`, computes mean ± standard deviation per group, and writes baseline and green results per group to `data/bench/report.md` and `report.json`. Table columns: metric, baseline, green, difference %. A per-stage energy breakdown table is also produced.

## 8. KVKK and retention

### 8.1 Principles

- **Data minimization:** No photo, age, gender or marital status fields. The parser does not extract them.
- **Disclosure:** A short disclosure text is shown on the upload page. In the preview step, the candidate sees what data will be kept about them.
- **Blind screening:** Contact information is kept in a separate table and shown to the employer only at the shortlist stage.
- **Reasoning:** For every eliminated candidate a reason-coded decision record is kept (for the right to object under KVKK Art. 11).
- **Principle:** Objective criteria eliminate automatically, soft criteria only rank. A candidate with a low embedding score is not eliminated; they move down the list.
- **Deliberate trade-off:** The parse cache lives only as long as the candidate has an active application. When the profile is deleted, the cache goes with it. Privacy comes before cache gain.

### 8.2 Reason codes

- Stored in DecisionRecord: `LANG_MISSING`, `LOCATION_MISMATCH`, `MIN_YEARS`, `NOT_SELECTED`
- Response-only, never stored: `DUPLICATE_APPLICATION`, `DRAFT_EXPIRED`

### 8.3 Rules

`retention.run(session, now, policy)` applies the rules below. `policy` carries the three retention periods from settings. All are idempotent.

| Rule | Trigger | Action |
|---|---|---|
| R1 | `draft` application, `created_at + DRAFT_TTL_HOURS < now` | Application deleted, no DecisionRecord written, profile refcount |
| R2 | Elimination by hard filter (immediate) | DecisionRecord(reason, `application_token` = the application's `draft_token`), Application deleted, profile refcount |
| R3 | Posting `closed` | for `ranked` applications DecisionRecord(`NOT_SELECTED`, `application_token` = the application's `draft_token`), Application deleted, profile refcount. The posting's `draft` applications are deleted at the same time, with no DecisionRecord, since a draft of a closed posting can no longer be confirmed (SC-22) |
| R4 | Posting `finalized`, or `closed_at + SHORTLIST_RETENTION_DAYS < now` | `shortlisted` applications deleted, profile refcount |
| R5 | `DecisionRecord.expires_at < now` | Record deleted |

- `expires_at = decided_at + DECISION_RETENTION_DAYS`
- **Profile refcount:** If no Application remains attached to a profile, the profile, its sections and its contact information are deleted (cascade).
- Retention runs at these moments: at application startup, when a posting is closed, and via the "Run retention" button on the green panel (`POST /green/retention`, which requires the employer cookie; viewing `/green` stays public). For the demo, this button includes a "fast-forward +N days" field that shifts the `now` value by that amount. There is no separate scheduler dependency.
- After each run, the number of deleted records is added to `GreenCounter.records_deleted`. `VACUUM` runs only in the benchmark and via the panel button.

## 9. UI

### 9.1 Pages

| Path | Role | Content |
|---|---|---|
| `GET /` | Candidate | Open postings |
| `GET /jobs/{id}` | Candidate | Posting detail, disclosure text, CV upload form |
| `POST /jobs/{id}/apply` | Candidate | Upload → redirect to `/drafts/{token}` or error. Uploads larger than `MAX_UPLOAD_BYTES` get HTTP 413 with a Turkish error message (Starlette spools uploads above 1 MB to a temp file, which would violate the never-written-to-disk rule) |
| `GET/POST /drafts/{token}` | Candidate | Preview and edit form, confirmation |
| `GET /applications/{token}` | Candidate | Status: under review / eliminated (with reason text) / shortlisted. Looks up Application by token first, then DecisionRecord by `application_token`; if neither exists, shows a generic "record deleted" message |
| `GET /employer` | Employer | Postings, create a new posting |
| `GET/POST /employer/login` | Employer | Token form; sets the employer cookie if the token matches `EMPLOYER_TOKEN` |
| `GET/POST /employer/jobs/new` | Employer | Title, description, hard filters, requirement rows (text + weight) |
| `GET /employer/jobs/{id}` | Employer | Ranked list (paginated, anonymous), close the posting |
| `POST /employer/jobs/{id}/close` | Employer | Close the posting (runs retention, R3) |
| `POST /employer/jobs/{id}/finalize` | Employer | Finalize the posting (R4) |
| `GET /employer/applications/{id}` | Employer | Matrix, summary, shortlisting; contact information if shortlisted |
| `POST /employer/applications/{id}/shortlist` | Employer | Shortlist the application (reveals contact information) |
| `GET /green` | Everyone | Counters, benchmark table, page weight, retention button |
| `POST /green/retention` | Employer | Run retention (optional `+N days` shift); requires the employer cookie |

### 9.2 Green UI rules

- Per-page budget: HTML + CSS under 50 KB compressed.
- A single CSS file, `Cache-Control: public, max-age=86400`.
- FastAPI `GZipMiddleware` on.
- System font stack. No images. Small inline SVG if needed.
- Dark theme is supported through `prefers-color-scheme` but is not presented as a green claim.
- The `pageweight` middleware records the transferred bytes of every response per path. The green panel shows averages.

## 10. Configuration (`.env`)

```
DATABASE_URL=sqlite:///data/app.db
HASH_SECRET=change-me
EMPLOYER_TOKEN=change-me
EMBED_MODEL=intfloat/multilingual-e5-small
EMBED_DEVICE=cuda
OLLAMA_URL=http://localhost:11434
LLM_MODEL=qwen2.5:3b
TOP_K=5
MAX_CV_CHARS=6000
MAX_UPLOAD_BYTES=1048576
QUALITY_K=10
DRAFT_TTL_HOURS=24
DECISION_RETENTION_DAYS=30
SHORTLIST_RETENTION_DAYS=30
SECTION_ALPHA=experience:1.0,project:1.0,certification:0.8,education:0.8,summary:0.7,skills:0.6
COUNTRY_ISO=TUR
GREEN_TRACKING=true
BENCH_CV_REFERENCE_DATE=2018-01-01
```

## 11. Risks and open questions

- **Parse quality:** The heading structure of the PDFs in the dataset is checked in T-03 with 20 samples, and the heading list is extended accordingly.
- **Baseline duration:** Hundreds of LLM calls with a 3B model can take a long time. N=50 is used during development. The final measurement uses N=300. If the duration gets too long, N is reduced and this is stated in the report.
- **Small absolute difference:** The per-application gain will turn out small. A projection for 1 million applications per year is added to the report. It is stated explicitly that the projection is an estimate.
- **CPU measurement:** If there is no hardware counter, CodeCarbon estimates CPU consumption from TDP. This is written to the `measurement_notes` field and to the report.
- **e5 score band:** Raw scores stay in a narrow range. So absolute score comparison across postings is not done.