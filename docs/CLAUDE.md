# Green Hiring Pipeline (green-hiring)

Course project for the Green Software class. A single-service FastAPI application that makes CV pre-screening tiered, data-minimal and measurable, plus a benchmark CLI that runs on the same code.

- Full design: `docs/SPEC.md`
- Task order and acceptance criteria: `docs/TASKS.md`

## How to work

- In every session, take the first open task in `docs/TASKS.md`. When the task is done, tick its box and write in the commit message how you verified the acceptance criteria.
- If something conflicts with the SPEC, do not deviate silently. Stop, ask, and propose which line of the SPEC should change.
- If a task requires a decision the SPEC does not define (a new table, a new endpoint, a new dependency), ask first.
- Code, identifiers, commit messages and logs are in English. UI strings are in Turkish.

## Green rules

These are part of the project's claim. Do not relax them for the sake of "convenience".

1. Ask before adding a new dependency. Approved list: `fastapi`, `uvicorn`, `jinja2`, `python-multipart`, `sqlalchemy`, `pydantic-settings`, `pypdf`, `numpy`, `sentence-transformers`, `httpx`, `codecarbon`, `pytest`.
2. `torch` is already installed on the machine with CUDA. Do not add it to the dependency list, reinstall it, or change its version.
3. No frontend framework, no build step, no webfonts, no images. A single CSS file (`app/web/static/app.css`), system font. JavaScript only if unavoidable, and then as small inline snippets.
4. The embedding model is loaded once at application startup (lifespan). Never load a model inside a request or a loop.
5. Do not write a code path that bypasses the cache. Parse results are reused via `raw_hash`, section embeddings via `text_hash` (SPEC §6).
6. Every measured operation runs inside the `stage("<name>")` context manager in `app/green/tracker.py`. Stage names come from the list in SPEC §7.4.
7. In the green pipeline the LLM is called only for top-k summaries. Do not use an LLM in the parse, filter or scoring steps.
8. Uploaded PDFs are never written to disk. They are processed in memory, and the reference is dropped once the text is extracted.
9. Candidate lists are paginated (20 per page). Never render a whole table on one page.

## KVKK rules

- Contact information (`CandidateContact`) reaches employer views only when the application is `shortlisted`. `profile_json` and section texts contain no contact information.
- Deletions always go through `app/privacy/retention.py` (reference counting and `DecisionRecord`). Do not write a direct `DELETE` or `session.delete` anywhere else.
- Never write CV text, names, emails or phone numbers to logs. Logs may contain only ids and the first 8 characters of `raw_hash`.
- `raw_hash` is not a plain SHA-256 but an HMAC-SHA256 keyed with `HASH_SECRET`.

## Commands

```bash
uvicorn app.main:app --reload          # application
pytest -q                              # tests
python -m app.seed                     # load the 3 demo job postings
python -m bench idle --seconds 120     # idle power measurement
python -m bench run --pipeline green --n 50
python -m bench run --pipeline baseline --n 50
python -m bench compare                # group runs by scenario_json, mean ± std per group
ollama pull qwen2.5:3b                 # Ollama must be running locally on localhost:11434
```

## Layout

```
app/
  main.py            FastAPI app, lifespan (the embedding model is loaded here)
  config.py          pydantic-settings, .env
  db.py              SQLAlchemy engine, SQLite pragmas
  models.py          ORM entities (SPEC §5)
  seed.py            demo job postings
  pipeline.py        the SPEC §6.1 flow, shared by the web layer and bench
  ingest/            pdf.py, normalize.py, sections.py, contact.py
  scoring/           embed.py, filters.py, matrix.py
  llm/               ollama.py, prompts.py
  privacy/           retention.py, reasons.py
  green/             tracker.py, counters.py, pageweight.py
  web/               routes_candidate.py, routes_employer.py, routes_green.py, templates/, static/
bench/
  __main__.py        CLI
  dataset.py         Kaggle Resume Dataset reader and sampling
  scenario.py        application scenario generation
  runner.py          BenchmarkRunner base class
  green.py           green pipeline runner (uses app/ modules)
  baseline.py        baseline runner
  quality.py         precision@k, overlap@k
  report.py          JSON + Markdown output
tests/
data/                gitignored: dataset, SQLite files, bench outputs
```

## Test expectations

- Unit tests are mandatory for `ingest/normalize`, `ingest/sections`, `scoring/matrix`, `scoring/filters` and `privacy/retention`.
- `scoring/matrix` is a pure function. It knows nothing about the DB or the model: it takes numpy arrays and returns a result.
- `privacy/retention` functions take a `now: datetime` parameter. The clock is injected in tests; `datetime.now()` is never called directly.
- LLM and embedding calls are mocked in tests. Tests must run without a GPU and without Ollama.