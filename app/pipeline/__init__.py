"""The green pipeline flow of SPEC 6.1, shared by the web layer and the benchmark.

Entry points, re-exported here:

* `submit_upload` (`upload.py`): PDF bytes in, a draft application out. Text
  extraction, the keyed hash, the duplicate check and the parse cache.
* `confirm_draft` (`confirm.py`): the candidate's (optional) corrections in, a
  ranked or eliminated application out. Profile edits (`profile_edits.py`),
  hard filter, embedding and scoring (`scoring.py`), re-scoring of the
  profile's other applications (SC-03) and the top-k summaries
  (`summaries.py`).

`open_draft` (`draft_view.py`) and `look_up_status` (`application_status.py`)
serve the candidate's preview and status pages. `fill_top_k_summaries` is the
public hook that writes the top-k summaries after a confirmation that ran
without an LLM (the web layer calls it from a background task).

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

from app.pipeline.application_status import (
    ApplicationStatusView,
    StatusKind,
    look_up_status,
)
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
from app.pipeline.draft_view import DraftPreview, DraftState, PreviewSection, open_draft
from app.pipeline.summaries import fill_top_k_summaries
from app.pipeline.upload import submit_upload

__all__ = [
    "ApplicationStatusView",
    "ConfirmInput",
    "ConfirmOutcome",
    "ConfirmOutcomeKind",
    "ContactEdit",
    "DraftNotFoundError",
    "DraftPreview",
    "DraftState",
    "PipelineContext",
    "PostingClosedError",
    "PostingWithoutRequirementsError",
    "PreviewSection",
    "ProfileEdits",
    "SectionEdit",
    "StatusKind",
    "UploadInput",
    "UploadOutcome",
    "UploadOutcomeKind",
    "confirm_draft",
    "fill_top_k_summaries",
    "look_up_status",
    "open_draft",
    "submit_upload",
]
