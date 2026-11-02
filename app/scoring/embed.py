"""Embedding service and the section embedding cache (SPEC sections 5, 6.1 and 6.5).

`SentenceTransformerEmbedder` is the only place that builds a sentence-transformers
model. The model is loaded once at application start-up (lifespan) through
`load()`; embedding before that raises `EmbedderNotLoadedError`. The E5 prefixes
live inside the embedder, so callers cannot forget them.

`ensure_section_embeddings` is the single path through which a section gets its
vector: a section is never embedded without first looking for a stored vector
with the same `text_hash`.
"""

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.green.counters import CounterName, increment_counters
from app.ingest.normalize import HASH_TEXT_ENCODING, normalize
from app.models import Section

logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
EMBED_BATCH_SIZE = 32
EMBEDDING_DTYPE = np.dtype(np.float32)
FLOAT_BYTES = EMBEDDING_DTYPE.itemsize
VECTOR_DIMENSIONS = 1
MATRIX_DIMENSIONS = 2
DOWNLOAD_COMMAND_TEMPLATE = (
    ".venv/Scripts/python -c \"from huggingface_hub import snapshot_download as d; "
    "d('{model_name}')\""
)
MODEL_NOT_CACHED_MESSAGE = (
    "Embedding model {model_name!r} is not in the local Hugging Face cache. "
    "Download it once with:\n    {command}\n"
    "or set EMBED_LOCAL_FILES_ONLY=false to let the app download it at start-up."
)

FloatMatrix = NDArray[np.float32]


class EmbedderNotLoadedError(RuntimeError):
    """Raised when an embedding is requested before `load()` ran."""


class EmbeddingModelNotCachedError(RuntimeError):
    """Raised when local-only loading is on and the model is not in the cache."""


class MalformedEmbeddingBlobError(ValueError):
    """Raised when a stored embedding has a byte length that is not float32-shaped."""


class EmptySectionTextError(ValueError):
    """Raised when a section has no text left after normalization."""


class UnexpectedEmbeddingShapeError(RuntimeError):
    """Raised when an embedder returns something other than one row per text."""


class EmbeddingProvider(Protocol):
    """What the rest of the application needs from an embedding model."""

    @property
    def model_version(self) -> str:
        """Identifier of the model; part of every `text_hash`."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed requirement texts; returns a float32 matrix, one row per text."""
        ...

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed section texts; returns a float32 matrix, one row per text."""
        ...


class SentenceTransformerEmbedder:
    """`EmbeddingProvider` backed by a sentence-transformers model.

    Args:
        model_name: Hugging Face model id, also reported as `model_version`.
        device: Torch device the model runs on (`cuda` or `cpu`).
        local_files_only: Read the model from the local Hugging Face cache only,
            without any network request.
    """

    def __init__(self, model_name: str, device: str, local_files_only: bool = True) -> None:
        self._model_name = model_name
        self._device = device
        self._local_files_only = local_files_only
        self._model: Any = None

    @property
    def model_version(self) -> str:
        """The model name; available before `load()` because hashes need it."""
        return self._model_name

    def load(self) -> None:
        """Load the model once; later calls do nothing.

        The heavy import happens here, so importing this module (and running the
        tests) never pulls in torch.
        """
        if self._model is not None:
            return
        from huggingface_hub.errors import LocalEntryNotFoundError
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                local_files_only=self._local_files_only,
            )
        except OSError as error:
            if not isinstance(error.__cause__, LocalEntryNotFoundError):
                raise
            raise EmbeddingModelNotCachedError(self._not_cached_message()) from error
        logger.info(
            "Embedding model loaded: model=%s device=%s seconds=%.2f",
            self._model_name,
            self._device,
            time.perf_counter() - started,
        )

    def _not_cached_message(self) -> str:
        """Tell the user how to download the model once, or how to allow downloads."""
        command = DOWNLOAD_COMMAND_TEMPLATE.format(model_name=self._model_name)
        return MODEL_NOT_CACHED_MESSAGE.format(
            model_name=self._model_name, command=command
        )

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed requirement texts with the `query: ` prefix.

        Raises:
            EmbedderNotLoadedError: If `load()` has not run.
        """
        return self._encode(QUERY_PREFIX, texts)

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed section texts with the `passage: ` prefix.

        Raises:
            EmbedderNotLoadedError: If `load()` has not run.
        """
        return self._encode(PASSAGE_PREFIX, texts)

    def _encode(self, prefix: str, texts: Sequence[str]) -> FloatMatrix:
        """Prefix, batch-encode and L2-normalize the texts into a float32 matrix."""
        if self._model is None:
            raise EmbedderNotLoadedError(
                "The embedding model is not loaded; call load() at start-up."
            )
        if not texts:
            dimension = self._model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype=EMBEDDING_DTYPE)
        vectors = self._model.encode(
            [prefix + text for text in texts],
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=EMBEDDING_DTYPE)


def create_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    """Build an unloaded embedder from the settings.

    Reads `EMBED_MODEL`, `EMBED_DEVICE` and `EMBED_LOCAL_FILES_ONLY`.
    """
    return SentenceTransformerEmbedder(
        settings.embed_model,
        settings.embed_device,
        settings.embed_local_files_only,
    )


def embedding_to_blob(vector: NDArray[Any]) -> bytes:
    """Serialize one embedding as float32 bytes.

    Args:
        vector: A one-dimensional array; any float dtype is cast to float32.

    Raises:
        ValueError: If the array is not one-dimensional.
    """
    array = np.ascontiguousarray(vector, dtype=EMBEDDING_DTYPE)
    if array.ndim != VECTOR_DIMENSIONS:
        raise ValueError(f"Expected a 1-D embedding, got {array.ndim} dimensions.")
    return array.tobytes()


def blob_to_embedding(blob: bytes) -> FloatMatrix:
    """Read a float32 BLOB back into a writable 1-D array.

    Raises:
        MalformedEmbeddingBlobError: If the blob is empty or its length is not a
            multiple of four bytes.
    """
    if not blob or len(blob) % FLOAT_BYTES:
        raise MalformedEmbeddingBlobError(
            f"Embedding blob has {len(blob)} bytes; expected a positive "
            f"multiple of {FLOAT_BYTES}."
        )
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


def compute_text_hash(model_version: str, text: str) -> str:
    """SHA-256 hex digest of `model_version + normalize(text)` (SPEC section 5).

    Raises:
        EmptySectionTextError: If nothing is left of the text after normalization.
    """
    normalized = normalize(text)
    if not normalized:
        raise EmptySectionTextError("Text is empty after normalization.")
    payload = (model_version + normalized).encode(HASH_TEXT_ENCODING)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EmbedReport:
    """What `ensure_section_embeddings` did.

    Attributes:
        cache_hits: Sections that kept or reused a stored vector.
        embeddings_computed: Vectors the model actually computed.
        deduplicated: Sections that shared a vector computed in the same batch.
    """

    cache_hits: int
    embeddings_computed: int
    deduplicated: int


@dataclass(frozen=True)
class PendingSection:
    """A section that still needs a vector, with its freshly computed hash."""

    section: Section
    text_hash: str


def hash_section(section: Section, model_version: str) -> str:
    """Hash a section's text, naming the section (never its text) on failure."""
    try:
        return compute_text_hash(model_version, section.text)
    except EmptySectionTextError:
        raise EmptySectionTextError(
            f"Section {section.ordinal} ({section.type}) has no text after "
            "normalization."
        ) from None


def find_pending_sections(
    sections: Sequence[Section], model_version: str
) -> list[PendingSection]:
    """Hash every section (failing fast) and keep those without a matching vector."""
    hashed = [(section, hash_section(section, model_version)) for section in sections]
    return [
        PendingSection(section, text_hash)
        for section, text_hash in hashed
        if section.text_hash != text_hash or section.embedding is None
    ]


def load_stored_vectors(session: Session, text_hashes: set[str]) -> dict[str, bytes]:
    """Find stored vectors by hash across all sections (the global lookup).

    Args:
        session: Session to query.
        text_hashes: Hashes to look for.

    Returns:
        The vector of the oldest section per hash that has one.
    """
    if not text_hashes:
        return {}
    rows = session.execute(
        select(Section.text_hash, Section.embedding)
        .where(Section.text_hash.in_(text_hashes), Section.embedding.is_not(None))
        .order_by(Section.id)
    ).all()
    vectors: dict[str, bytes] = {}
    for text_hash, embedding in rows:
        vectors.setdefault(text_hash, embedding)
    return vectors


def compute_missing_vectors(
    missing: Sequence[PendingSection], embedder: EmbeddingProvider
) -> dict[str, bytes]:
    """Embed the missing sections in one batch, each distinct hash only once.

    Raises:
        UnexpectedEmbeddingShapeError: If the embedder returns the wrong number
            of rows.
    """
    texts_by_hash: dict[str, str] = {}
    for pending in missing:
        texts_by_hash.setdefault(pending.text_hash, pending.section.text)
    if not texts_by_hash:
        return {}
    matrix = embedder.embed_passages(list(texts_by_hash.values()))
    if matrix.ndim != MATRIX_DIMENSIONS or len(matrix) != len(texts_by_hash):
        raise UnexpectedEmbeddingShapeError(
            f"Embedder returned shape {matrix.shape} for {len(texts_by_hash)} texts."
        )
    return {
        text_hash: embedding_to_blob(row)
        for text_hash, row in zip(texts_by_hash, matrix, strict=True)
    }


def assign_vectors(
    pending: Sequence[PendingSection], vectors: dict[str, bytes], model_version: str
) -> None:
    """Write hash, vector and model version onto each pending section."""
    for item in pending:
        item.section.text_hash = item.text_hash
        item.section.embedding = vectors[item.text_hash]
        item.section.model_version = model_version


def ensure_section_embeddings(
    session: Session, sections: Sequence[Section], embedder: EmbeddingProvider
) -> EmbedReport:
    """Give every section its `text_hash`, `embedding` and `model_version`.

    A section that already holds the matching hash and a vector keeps its row. For
    the others, the vector is first looked up globally by hash (across profiles);
    only what is still missing is embedded, in one batch. Counters are updated in
    the caller's transaction; nothing is committed and no stage is opened (the
    caller measures embedding).

    Args:
        session: Session of the caller's transaction.
        sections: Sections of one profile.
        embedder: The loaded embedding provider.

    Returns:
        How many sections hit the cache, were computed, or were deduplicated.

    Raises:
        EmptySectionTextError: If any section is empty after normalization. This
            is checked before anything is changed.
    """
    model_version = embedder.model_version
    pending = find_pending_sections(sections, model_version)
    stored = load_stored_vectors(session, {item.text_hash for item in pending})
    missing = [item for item in pending if item.text_hash not in stored]
    computed = compute_missing_vectors(missing, embedder)
    assign_vectors(pending, {**stored, **computed}, model_version)
    report = EmbedReport(
        cache_hits=len(sections) - len(missing),
        embeddings_computed=len(computed),
        deduplicated=len(missing) - len(computed),
    )
    record_embed_counters(session, report)
    return report


def record_embed_counters(session: Session, report: EmbedReport) -> None:
    """Add the cache hits and computed vectors of a report to the green counters."""
    increments = {
        CounterName.EMBED_CACHE_HITS: report.cache_hits,
        CounterName.EMBEDDINGS_COMPUTED: report.embeddings_computed,
    }
    nonzero = {name: amount for name, amount in increments.items() if amount}
    increment_counters(session, nonzero)
