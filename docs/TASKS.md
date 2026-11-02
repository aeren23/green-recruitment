# Tasks

The order is the priority order. Each task is sized to fit a single Claude Code session. Do not move on to the next task before finishing the current one.

**Checkpoints:**
- **K1 (end of T-11):** The first real benchmark numbers must be taken with N=50. The project's main claim is verified here. If the numbers come out unexpected, stop before moving to the UI and find out why.
- **MVP line (end of T-16):** Everything needed for delivery. Anything after it is a bonus.

For the large tasks (T-03, T-09, T-10, T-13, T-14), first produce a plan in plan mode, then code after approval.

---

## Phase 0: Skeleton

- [x] **T-00 Repo skeleton**
  - `pyproject.toml` (approved dependencies, excluding torch), `.env.example`, `.gitignore` (`data/`, `.env`), `app/config.py`, `app/db.py` (pragmas on every connection), `app/main.py` (lifespan, `GET /health`), empty `tests/`.
  - Acceptance: `uvicorn app.main:app` starts, `/health` returns 200, `pytest -q` runs. One test verifies that `PRAGMA secure_delete` and `foreign_keys` are both 1.

- [x] **T-01 Dataset reader**
  - I will download the dataset and put it under `data/raw/`. `bench/dataset.py`: an iterator producing `(id, category, pdf_path, resume_str)` records, and stratified sampling per SPEC §7.3 (with `seed`).
  - The command `python -m bench dataset-stats` prints the category distribution, the PDF count and the missing PDF count.
  - Acceptance: If the structure differs from the expectation in SPEC §7.1, stop and tell me. The same seed gives the same sample.

## Phase 1: Ingest

- [x] **T-02 PDF, normalization, hash**
  - `ingest/pdf.py`: text extraction from in-memory bytes. A meaningful error if no text comes out or the text is too short (<200 characters).
  - `ingest/normalize.py`: SPEC §6.2.
  - Acceptance: Tests exist. The same PDF gives the same hash twice. Differences in case or whitespace do not change the hash. `C++` and `C#` are preserved.

- [x] **T-03 Section splitter and contact extraction**
  - `ingest/sections.py` and `ingest/contact.py`: SPEC §6.3. `years_experience` is computed by merging overlapping ranges.
  - `python -m bench parse-report --n 50`: for 50 random CVs, prints the section type distribution, the share of CVs without headings and the share where `years_experience` was found.
  - Acceptance: Unit tests (heading matching, entry splitting, 200-word chunking, year range merging, removal of contact information from section text). In the parse report, at least 80% of CVs have 2 or more section types found. If not, extend the heading list according to the dataset.

- [x] **T-04 Data model**
  - `app/models.py`: all entities from SPEC §5, with unique constraints and cascades. Benchmark tables in a separate metadata object.
  - Acceptance: The test DB is created. A `unique(job_id, raw_hash)` violation raises an error. Deleting a profile also deletes its sections and contact information.

## Phase 2: Measurement and scoring core

- [x] **T-05 Tracker**
  - `app/green/tracker.py`: a `stage(name)` context manager. It wraps the CodeCarbon task API and accumulates per-stage energy (Wh) and duration. With `GREEN_TRACKING=false` it becomes a zero-cost no-op. `python -m bench idle --seconds N` measures idle power.
  - Acceptance: The task API names are verified against the installed CodeCarbon version. In one test, two stages are reported separately using `time.sleep` (the tracker may be mocked in the test). The `measurement_notes` field contains the CPU measurement method.

- [x] **T-06 Embedding service**
  - `scoring/embed.py`: the model is loaded once in lifespan. `query:` and `passage:` prefixes, `normalize_embeddings=True`, batch processing, conversion to and from float32 BLOB. Reuse of sections by `text_hash` match.
  - Acceptance: On a second confirmation with the same section text, no embedding is computed (tested via the counter). Calling before the model is loaded raises a clear error.

- [x] **T-07 Hard filter and matrix score**
  - `scoring/filters.py`: SPEC §6.4, with the rule "unknown is not a reason for elimination". `scoring/matrix.py`: SPEC §6.5, pure numpy.
  - Acceptance: Tests with small hand-built vectors. An evidence section beats a claim section at the same similarity. Weights affect the score correctly. An empty section list gives score 0. For each filter, the elimination, pass and unknown cases are tested.

- [x] **T-08 LLM client**
  - `llm/ollama.py` and `llm/prompts.py`: the baseline scoring prompt (JSON output) and the green summary prompt (selected sections only). Token counters, timeout, graceful error when Ollama is down.
  - Acceptance: Tests with mocks. If malformed JSON comes back, the baseline score is retried once; if it still fails, it is recorded as `score=None` and counted.

## Phase 3: Benchmark

- [ ] **T-09 Green pipeline runner**
  - `bench/scenario.py` (SPEC §7.3) and `bench/green.py`: uses the `app/` modules directly and writes to a separate DB (`data/bench/bench_green.db`). At the end all postings are closed, retention and `VACUUM` run, and storage is measured at two points. A minimal version of retention is written for this task and completed in T-14.
  - Acceptance: `bench run --pipeline green --n 50` runs, and `BenchmarkRun` and `StageMeasurement` records are written.

- [ ] **T-10 Baseline runner**
  - `bench/baseline.py`: SPEC §7.4. Same scenario and same seed.
  - Acceptance: `bench run --pipeline baseline --n 50` runs. Total duration is printed to the console, and an estimated duration for N=300 is shown.

- [ ] **T-11 Quality and report**
  - `bench/quality.py` (precision@k, overlap@k) and `bench/report.py` (`bench compare` → `report.md`, `report.json`). Mean ± deviation is computed over repeated runs.
  - Acceptance: The report contains the tables in SPEC §7.8.
  - **→ K1: Show me the numbers.**

## Phase 4: Demo application

- [ ] **T-12 Seed and employer posting pages**
  - `app/seed.py` (the 3 postings in SPEC §7.2), template skeleton (`base.html`, `app.css`), `/employer` and `/employer/jobs/new`. Cookie check with `EMPLOYER_TOKEN`.
  - Acceptance: When a posting is created, requirement embeddings are computed once. Page weight is within budget.

- [ ] **T-13 Candidate flow**
  - Upload → draft → preview and edit → confirm → status page. Repeat-application block, cache-hit message, "Dosyanız saklanmadı" message. After confirmation, filtering and scoring run. `GreenCounter` is updated.
  - Acceptance: Steps 2, 3 and 4 of SPEC §3 work end to end by hand. That the PDF is not written to disk is verified with a test (a temporary directory is watched).

- [ ] **T-14 Retention and KVKK**
  - `privacy/retention.py`: R1–R5 from SPEC §8.3, refcount, the `now` parameter. The posting-close endpoint.
  - Acceptance: A test for each rule. It is tested that when a profile has applications to two postings and is eliminated from one, the profile is not deleted. Within the DecisionRecord period, a repeat application to the same posting is blocked.

- [ ] **T-15 Ranked list, matrix, shortlist**
  - `/employer/jobs/{id}` (paginated, anonymous, relative score, "missing information" marker) and `/employer/applications/{id}` (matrix table, evidence/claim marker, summary). Shortlisting and revealing contact information. When the top-k changes, summaries are generated only for the new top-k candidates.
  - Acceptance: Contact information appears in no employer response outside the shortlist (tested on the response HTML).

- [ ] **T-16 Green panel**
  - `/green`: live counters, the latest comparison table from `bench.db`, `pageweight` middleware averages, a "Run retention" button and a "fast-forward +N days" field.
  - Acceptance: The demo scenario in SPEC §3 runs from start to finish without problems.
  - **→ MVP line.**

## Phase 5: Final measurement and delivery

- [ ] **T-17 Final benchmark**
  - Idle measurement, two scenarios (SPEC §7.3), N=300 (or as much as time allows), 3 repeats each. A projection for 1 million applications per year is added to the report, stated as an estimate.
  - Acceptance: `report.md` is ready to go into the presentation.

- [ ] **T-18 README**
  - Installation, running, measurement protocol, limitations (TDP estimate, M left out, language and location filters off in the benchmark, small baseline model).

## Bonus (if time remains)

- [ ] **B-01 Embedding device experiment:** Run the same scenario with `EMBED_DEVICE=cpu` and `cuda` and compare energy. At this scale the GPU should not be assumed to be more efficient; go with whatever the measurement says.
- [ ] **B-02 Page carbon:** Take the constants of the Sustainable Web Design model from the CO2.js source code and cite the source in a comment. Show a "gCO2e per page view" value on the panel.
- [ ] **B-03 Model size experiment:** Run the baseline with `qwen2.5:1.5b` as well. How do energy and precision@k change together?