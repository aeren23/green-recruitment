"""Build minimal valid PDFs from raw bytes for tests.

The text goes into a hand-written page content stream with a correct xref
table, so nothing here depends on pypdf's writer. Non-text properties (an Info
dictionary, the physical order of the objects) can vary, which produces a
"re-exported" file: different bytes, same extracted text.
"""

from collections.abc import Sequence
from dataclasses import dataclass

PDF_HEADER = b"%PDF-1.4\n"
PDF_ENCODING = "ascii"
FONT_NAME = "F1"
FONT_SIZE = 12
LINE_LEADING = 14
TEXT_ORIGIN_X = 50
TEXT_ORIGIN_Y = 750
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
XREF_FREE_ENTRY = b"0000000000 65535 f \n"
XREF_ENTRY_TEMPLATE = "{offset:010d} 00000 n \n"

CATALOG_OBJECT = 1
PAGES_OBJECT = 2
PAGE_OBJECT = 3
CONTENT_OBJECT = 4
FONT_OBJECT = 5
INFO_OBJECT = 6


@dataclass(frozen=True)
class PdfLayout:
    """Non-text properties that change the bytes but never the extracted text.

    Attributes:
        info_title: When set, adds an Info dictionary carrying this title.
        reverse_object_order: Write the objects in reverse physical order.
    """

    info_title: str | None = None
    reverse_object_order: bool = False


def escape_pdf_text(text: str) -> str:
    """Escape the characters that are special inside a PDF literal string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_content_stream(lines: Sequence[str]) -> bytes:
    """Return a page content stream that shows one text line per entry."""
    commands = [
        "BT",
        f"/{FONT_NAME} {FONT_SIZE} Tf",
        f"{LINE_LEADING} TL",
        f"{TEXT_ORIGIN_X} {TEXT_ORIGIN_Y} Td",
    ]
    for line in lines:
        commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode(PDF_ENCODING)


def build_object_bodies(content: bytes, layout: PdfLayout) -> dict[int, bytes]:
    """Map each object number to its body, without the `obj`/`endobj` wrapper."""
    bodies = {
        CATALOG_OBJECT: f"<< /Type /Catalog /Pages {PAGES_OBJECT} 0 R >>".encode(),
        PAGES_OBJECT: f"<< /Type /Pages /Kids [{PAGE_OBJECT} 0 R] /Count 1 >>".encode(),
        PAGE_OBJECT: (
            f"<< /Type /Page /Parent {PAGES_OBJECT} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Contents {CONTENT_OBJECT} 0 R "
            f"/Resources << /Font << /{FONT_NAME} {FONT_OBJECT} 0 R >> >> >>"
        ).encode(),
        CONTENT_OBJECT: (
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        ),
        FONT_OBJECT: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    if layout.info_title is not None:
        title = escape_pdf_text(layout.info_title)
        bodies[INFO_OBJECT] = f"<< /Title ({title}) >>".encode(PDF_ENCODING)
    return bodies


def build_xref_and_trailer(
    offsets: dict[int, int], xref_offset: int, has_info: bool
) -> bytes:
    """Return the xref table, trailer and end-of-file marker."""
    size = max(offsets) + 1
    table = b"xref\n" + f"0 {size}\n".encode() + XREF_FREE_ENTRY
    for object_number in range(1, size):
        entry = XREF_ENTRY_TEMPLATE.format(offset=offsets[object_number])
        table += entry.encode(PDF_ENCODING)
    info_reference = f" /Info {INFO_OBJECT} 0 R" if has_info else ""
    trailer = f"trailer\n<< /Size {size} /Root {CATALOG_OBJECT} 0 R{info_reference} >>\n"
    return table + trailer.encode() + f"startxref\n{xref_offset}\n%%EOF\n".encode()


def assemble_pdf(bodies: dict[int, bytes], reverse_order: bool) -> bytes:
    """Write the objects in the requested physical order and append the xref."""
    object_numbers = sorted(bodies, reverse=reverse_order)
    document = PDF_HEADER
    offsets: dict[int, int] = {}
    for object_number in object_numbers:
        offsets[object_number] = len(document)
        document += f"{object_number} 0 obj\n".encode() + bodies[object_number]
        document += b"\nendobj\n"
    return document + build_xref_and_trailer(
        offsets, len(document), INFO_OBJECT in bodies
    )


def build_pdf(lines: Sequence[str], layout: PdfLayout = PdfLayout()) -> bytes:
    """Build a one-page PDF whose text is the given ASCII lines, one per row.

    Args:
        lines: Text rows. Must be ASCII.
        layout: Non-text variations that change the bytes but not the text.

    Returns:
        The complete PDF file as bytes.
    """
    content = build_content_stream(lines)
    bodies = build_object_bodies(content, layout)
    return assemble_pdf(bodies, layout.reverse_object_order)


SAMPLE_CV_LINES = (
    "SUMMARY",
    "Backend developer with six years of experience building REST APIs.",
    "EXPERIENCE",
    "Software Engineer, Example Corp, 2018 - 2024",
    "Designed relational databases and wrote SQL reports for finance teams.",
    "Maintained Linux servers and container based deployments.",
    "SKILLS",
    "Python, Java, SQL, C++, C#, Docker",
    "EDUCATION",
    "BSc Computer Engineering, Example University, 2014 - 2018",
)
