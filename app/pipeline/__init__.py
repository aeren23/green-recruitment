"""The green pipeline flow of SPEC 6.1, shared by the web layer and the benchmark.

Two entry points, re-exported here:

* `submit_upload` (`upload.py`): PDF bytes in, a draft application out. Text
  extraction, the keyed hash, the duplicate check and the parse cache.
* `confirm_draft` (`confirm.py`): the candidate's (optional) corrections in, a
  ranked or eliminated application out. Profile edits (`profile_edits.py`),
  hard filter, embedding and scoring (`scoring.py`), re-scoring of the
  profile's other applications (SC-03) and the top-k summaries
  (`summaries.py`).

`context.py` holds the DTOs, errors and stage names, `shared.py` the small
lookups several modules need.

Commit policy: **no function in this package commits.** The caller owns the
transaction: the benchmark commits once per application, the web layer once
per request. A rolled-back unit of work therefore also rolls back its
counters and deletions.

Measurement: every measured operation runs inside `stage(<name>)` of the
injected tracker (SPEC 7.4). Stages run one after another and never nest. The
duplicate check and the parse-cache lookup run inside the `hash` stage, so the
green pipeline has no unmeasured lookups.

Green rules kept here: an uploaded PDF is processed in memory and the
pipeline drops its reference to the bytes as soon as the text is extracted; the
parse result is reused through `raw_hash` and the section embeddings through
`text_hash`; the LLM is called only to summarise top-k candidates.

KVKK: logs and exception messages carry ids and `short_hash(raw_hash)` only.
"""

from app.pipeline.confirm import confirm_draft
from app.pipeline.context import (
    ConfirmInput,
    ConfirmOutcome,
    ConfirmOutcomeKind,
    ContactEdit,
    DraftNotFoundError,
    PipelineContext,
    PostingClosedError,
    PostingWithoutRequirementsError,
    ProfileEdits,
    SectionEdit,
    UploadInput,
    UploadOutcome,
    UploadOutcomeKind,
)
from app.pipeline.upload import submit_upload

__all__ = [
    "ConfirmInput",
    "ConfirmOutcome",
    "ConfirmOutcomeKind",
    "ContactEdit",
    "DraftNotFoundError",
    "PipelineContext",
    "PostingClosedError",
    "PostingWithoutRequirementsError",
    "ProfileEdits",
    "SectionEdit",
    "UploadInput",
    "UploadOutcome",
    "UploadOutcomeKind",
    "confirm_draft",
    "submit_upload",
]
