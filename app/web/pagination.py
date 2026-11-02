"""Page-at-a-time listing (Green rule 9: 20 rows per page, never a whole table).

`paginate` works on any single-entity `select`, so the candidate list of T-15
can reuse it. A page number past the end is clamped to the last page, and an
unparsable one is treated as the first.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

PAGE_SIZE = 20
FIRST_PAGE = 1


@dataclass(frozen=True)
class Page[ItemT]:
    """One page of a listing.

    Attributes:
        items: The rows of this page (at most `PAGE_SIZE`).
        number: This page's number, starting at 1.
        page_count: Total number of pages (at least 1, even for no rows).
        total: Total number of rows over all pages.
    """

    items: Sequence[ItemT]
    number: int
    page_count: int
    total: int

    @property
    def has_previous(self) -> bool:
        """True when an earlier page exists."""
        return self.number > FIRST_PAGE

    @property
    def has_next(self) -> bool:
        """True when a later page exists."""
        return self.number < self.page_count

    @property
    def previous_number(self) -> int:
        """Number of the previous page (on the first page: the first page)."""
        return max(self.number - 1, FIRST_PAGE)

    @property
    def next_number(self) -> int:
        """Number of the next page (on the last page: the last page)."""
        return min(self.number + 1, self.page_count)


def parse_page_number(raw_value: str | None) -> int:
    """Read the `page` query value leniently; anything unusable is page 1."""
    try:
        return max(int(raw_value or FIRST_PAGE), FIRST_PAGE)
    except ValueError:
        return FIRST_PAGE


def paginate[ItemT](
    session: Session, statement: Select[tuple[ItemT]], page_number: int
) -> Page[ItemT]:
    """Run an ordered `select` for one page of `PAGE_SIZE` rows.

    Args:
        session: Session to run the statements on.
        statement: An ordered single-entity select, without limit or offset.
        page_number: Requested page, starting at 1; clamped to the last page.

    Returns:
        The page, with the total row count.
    """
    counting = select(func.count()).select_from(statement.order_by(None).subquery())
    total = session.execute(counting).scalar_one()
    page_count = max(-(-total // PAGE_SIZE), FIRST_PAGE)
    number = min(max(page_number, FIRST_PAGE), page_count)
    offset = (number - FIRST_PAGE) * PAGE_SIZE
    rows = session.scalars(statement.limit(PAGE_SIZE).offset(offset)).all()
    return Page(items=rows, number=number, page_count=page_count, total=total)
