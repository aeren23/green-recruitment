"""Live counters of the green panel: the single `GreenCounter` row (SPEC section 5).

Increments are one SQL `UPDATE column = column + amount`, so they are atomic even
when two requests count at the same time. None of these functions commits: the
caller owns the transaction, which means a rolled-back unit of work also rolls
back its counts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import GREEN_COUNTER_ROW_ID, GreenCounter

DEFAULT_INCREMENT = 1


class CounterName(StrEnum):
    """The counters of SPEC section 5; the value is the `GreenCounter` column name."""

    PDF_BYTES_NOT_STORED = "pdf_bytes_not_stored"
    PARSE_CACHE_HITS = "parse_cache_hits"
    EMBED_CACHE_HITS = "embed_cache_hits"
    EMBEDDINGS_COMPUTED = "embeddings_computed"
    LLM_CALLS = "llm_calls"
    LLM_CALLS_AVOIDED = "llm_calls_avoided"
    RECORDS_DELETED = "records_deleted"


class UnknownCounterError(ValueError):
    """Raised when a counter name is not one of `CounterName`."""


class NegativeIncrementError(ValueError):
    """Raised when a counter would be decreased; counters only ever grow."""


@dataclass(frozen=True)
class CounterSnapshot:
    """Current value of every counter (all zero while the row does not exist yet)."""

    pdf_bytes_not_stored: int = 0
    parse_cache_hits: int = 0
    embed_cache_hits: int = 0
    embeddings_computed: int = 0
    llm_calls: int = 0
    llm_calls_avoided: int = 0
    records_deleted: int = 0


def resolve_counter_name(name: CounterName | str) -> CounterName:
    """Return the `CounterName` for a member or its string value.

    Args:
        name: A `CounterName` or the column name it stands for.

    Raises:
        UnknownCounterError: If the name is not a known counter.
    """
    try:
        return CounterName(name)
    except ValueError:
        known_names = ", ".join(member.value for member in CounterName)
        raise UnknownCounterError(
            f"Unknown counter {name!r}; expected one of: {known_names}"
        ) from None


def ensure_counter_row(session: Session) -> None:
    """Create the single counter row if it is missing; a no-op otherwise."""
    statement = (
        sqlite_insert(GreenCounter)
        .values(id=GREEN_COUNTER_ROW_ID)
        .on_conflict_do_nothing()
    )
    session.execute(statement)


def increment_counters(
    session: Session, increments: Mapping[CounterName | str, int]
) -> None:
    """Add to several counters with one atomic `UPDATE`, creating the row if needed.

    Everything is validated before the database is touched, so a bad name or
    amount leaves the counters exactly as they were.

    Args:
        session: Session whose transaction the caller owns; nothing is committed.
        increments: Amount to add per counter. Amounts must not be negative.

    Raises:
        UnknownCounterError: If a name is not a known counter.
        NegativeIncrementError: If an amount is negative.
    """
    resolved = {resolve_counter_name(name): amount for name, amount in increments.items()}
    for counter_name, amount in resolved.items():
        if amount < 0:
            raise NegativeIncrementError(
                f"Counter {counter_name.value!r} cannot be decreased (got {amount})."
            )
    if not resolved:
        return
    ensure_counter_row(session)
    session.execute(
        update(GreenCounter)
        .where(GreenCounter.id == GREEN_COUNTER_ROW_ID)
        .values(
            {
                counter_name.value: getattr(GreenCounter, counter_name.value) + amount
                for counter_name, amount in resolved.items()
            }
        )
    )


def increment_counter(
    session: Session, name: CounterName | str, amount: int = DEFAULT_INCREMENT
) -> None:
    """Add to one counter atomically, creating the row if needed.

    Args:
        session: Session whose transaction the caller owns; nothing is committed.
        name: The counter to increase.
        amount: How much to add; must not be negative.

    Raises:
        UnknownCounterError: If the name is not a known counter.
        NegativeIncrementError: If the amount is negative.
    """
    increment_counters(session, {name: amount})


def read_counters(session: Session) -> CounterSnapshot:
    """Read every counter without writing; all zero while the row is missing.

    Selecting columns (not the entity) keeps the values fresh even when a
    `GreenCounter` object sits in the session's identity map.
    """
    columns = [getattr(GreenCounter, member.value) for member in CounterName]
    row = session.execute(
        select(*columns).where(GreenCounter.id == GREEN_COUNTER_ROW_ID)
    ).one_or_none()
    if row is None:
        return CounterSnapshot()
    return CounterSnapshot(**dict(zip((m.value for m in CounterName), row, strict=True)))
