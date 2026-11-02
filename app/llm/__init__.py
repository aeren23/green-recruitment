"""LLM access: the Ollama client, prompt templates and the two LLM tasks (SPEC 6.6, 7.4).

Nothing under `app/ingest/` or `app/scoring/` may import this package: the green
pipeline calls an LLM only for top-k summaries (Green rule 7).
"""
