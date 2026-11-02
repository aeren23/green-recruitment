"""Regenerate `docs/demo/demo-cv.pdf`, the synthetic CV used in the SPEC 3 demo.

Run from the repository root:

    .venv/Scripts/python docs/demo/make_demo_cv.py

The person, the e-mail address (example.com) and the phone number are all
fictional. The text is English and ASCII only because the test PDF builder
writes plain Helvetica. "Present" is clipped to the parse date by the parser,
so the experience adds up to roughly five years in 2026.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

from pdf_fixtures import build_pdf  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().with_name("demo-cv.pdf")

DEMO_CV_LINES = (
    "Ada Example",
    "ada.example@example.com  +90 555 010 20 30",
    "Summary",
    "Backend developer with about five years of experience designing REST APIs",
    "and relational databases for small teams. Comfortable owning a service",
    "from design to deployment.",
    "Experience",
    "Backend Developer, Fictional Labs, 2024 - Present",
    "Built and operated REST APIs in Python and FastAPI. Designed PostgreSQL",
    "schemas and wrote SQL reports. Added automated tests and Docker based",
    "deployments.",
    "Software Developer, Sample Software, 2023 - 2024",
    "Maintained internal web services and improved slow SQL queries.",
    "Junior Developer, Demo Systems, 2022 - 2023",
    "Implemented small features in a Python web application and fixed bugs.",
    "Education",
    "BSc Computer Engineering, Example University, 2017 - 2021",
    "Skills",
    "Python, FastAPI, SQL, PostgreSQL, Docker, Git, REST API design, unit testing",
    "Languages",
    "English, Turkish",
)


def main() -> None:
    """Write the demo CV next to this script."""
    OUTPUT_PATH.write_bytes(build_pdf(DEMO_CV_LINES))
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
