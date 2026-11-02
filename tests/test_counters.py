"""The green counters: atomic increments on the single `GreenCounter` row."""

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.db import create_session_factory
from app.green.counters import (
    CounterName,
    CounterSnapshot,
    NegativeIncrementError,
    UnknownCounterError,
    increment_counter,
    increment_counters,
    read_counters,
)
from app.models import GreenCounter

SMALL_AMOUNT = 3
LARGE_AMOUNT = 40


def count_rows(session: Session) -> int:
    """Number of rows in the counter table."""
    return session.execute(select(func.count()).select_from(GreenCounter)).scalar_one()


def test_reading_a_missing_row_gives_zeros_and_writes_nothing(
    demo_session: Session,
) -> None:
    """Reads never create the row."""
    assert read_counters(demo_session) == CounterSnapshot()
    assert count_rows(demo_session) == 0


def test_first_increment_creates_the_row(demo_session: Session) -> None:
    """The row appears on demand and only the named counter moves."""
    increment_counter(demo_session, CounterName.LLM_CALLS)
    snapshot = read_counters(demo_session)
    assert snapshot.llm_calls == 1
    assert snapshot.embed_cache_hits == 0
    assert count_rows(demo_session) == 1


def test_increments_accumulate_and_the_row_is_created_once(
    demo_session: Session,
) -> None:
    """Repeated increments add up on one row."""
    increment_counter(demo_session, CounterName.EMBED_CACHE_HITS, SMALL_AMOUNT)
    increment_counter(demo_session, CounterName.EMBED_CACHE_HITS, LARGE_AMOUNT)
    increment_counter(demo_session, CounterName.EMBED_CACHE_HITS)
    assert read_counters(demo_session).embed_cache_hits == SMALL_AMOUNT + LARGE_AMOUNT + 1
    assert count_rows(demo_session) == 1


def test_several_counters_move_in_one_call(demo_session: Session) -> None:
    """A mapping updates every named counter and leaves the others alone."""
    increment_counters(
        demo_session,
        {
            CounterName.EMBEDDINGS_COMPUTED: SMALL_AMOUNT,
            CounterName.PDF_BYTES_NOT_STORED: LARGE_AMOUNT,
        },
    )
    assert read_counters(demo_session) == CounterSnapshot(
        embeddings_computed=SMALL_AMOUNT, pdf_bytes_not_stored=LARGE_AMOUNT
    )


def test_a_counter_can_be_named_by_its_column_string(demo_session: Session) -> None:
    """The string value of a known counter is accepted."""
    increment_counter(demo_session, "records_deleted", SMALL_AMOUNT)
    assert read_counters(demo_session).records_deleted == SMALL_AMOUNT


def test_unknown_counter_fails_fast(demo_session: Session) -> None:
    """An unknown name raises and writes nothing, not even the row."""
    with pytest.raises(UnknownCounterError):
        increment_counter(demo_session, "not_a_counter")
    assert count_rows(demo_session) == 0


def test_bad_name_in_a_mapping_leaves_valid_counters_untouched(
    demo_session: Session,
) -> None:
    """Validation happens before any write."""
    with pytest.raises(UnknownCounterError):
        increment_counters(
            demo_session, {CounterName.LLM_CALLS: SMALL_AMOUNT, "bogus": 1}
        )
    assert read_counters(demo_session) == CounterSnapshot()


def test_negative_increment_is_rejected(demo_session: Session) -> None:
    """Counters never decrease."""
    increment_counter(demo_session, CounterName.LLM_CALLS, SMALL_AMOUNT)
    with pytest.raises(NegativeIncrementError):
        increment_counter(demo_session, CounterName.LLM_CALLS, -1)
    assert read_counters(demo_session).llm_calls == SMALL_AMOUNT


def test_rollback_undoes_the_increment(engine: Engine, demo_session: Session) -> None:
    """The caller owns the transaction: a rollback takes the counts with it."""
    increment_counter(demo_session, CounterName.EMBEDDINGS_COMPUTED, SMALL_AMOUNT)
    demo_session.commit()
    increment_counter(demo_session, CounterName.EMBEDDINGS_COMPUTED, LARGE_AMOUNT)
    assert read_counters(demo_session).embeddings_computed == SMALL_AMOUNT + LARGE_AMOUNT
    demo_session.rollback()
    assert read_counters(demo_session).embeddings_computed == SMALL_AMOUNT


def test_nothing_is_committed_by_the_functions(
    engine: Engine, demo_session: Session
) -> None:
    """Another session sees no counts until the caller commits."""
    increment_counter(demo_session, CounterName.PARSE_CACHE_HITS)
    with create_session_factory(engine)() as other_session:
        assert read_counters(other_session).parse_cache_hits == 0
    demo_session.commit()
    with create_session_factory(engine)() as other_session:
        assert read_counters(other_session).parse_cache_hits == 1


def test_read_is_fresh_even_when_the_entity_is_loaded(demo_session: Session) -> None:
    """A loaded `GreenCounter` object does not make the snapshot stale."""
    increment_counter(demo_session, CounterName.LLM_CALLS)
    demo_session.get(GreenCounter, 1)
    increment_counter(demo_session, CounterName.LLM_CALLS)
    assert read_counters(demo_session).llm_calls == 2
