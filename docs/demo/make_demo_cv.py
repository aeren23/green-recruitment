"""Regenerate the synthetic demo CVs in `docs/demo/` (`demo-cv.pdf` and the ranking set).

Run from the repository root:

    .venv/Scripts/python docs/demo/make_demo_cv.py

Each persona shows one thing in the employer ranking; see `docs/demo/README.md`.
The people, the e-mail addresses (example.com) and the phone number are all
fictional. The text is English and ASCII only because the test PDF builder
writes plain Helvetica. "Present" is clipped to the parse date by the parser,
so the experience adds up to roughly five years in 2026.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

from pdf_fixtures import build_pdf  # noqa: E402

OUTPUT_DIRECTORY = Path(__file__).resolve().parent

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


PARAPHRASE_CV_LINES = (
    "Deniz Sample",
    "deniz.sample@example.com  +90 555 011 20 31",
    "Denizli, Turkey",
    "Summary",
    "Software engineer with five years of experience building server-side services",
    "for mobile and web clients. Owns systems from design to production.",
    "Experience",
    "Senior Software Engineer, Fictional Cloud, 2023 - Present",
    "Designed HTTP endpoints for mobile clients with Spring Boot and Java.",
    "Modelled PostgreSQL tables and tuned slow queries. Ran the services on",
    "Kubernetes on AWS with automated deployments and monitoring.",
    "Software Engineer, Sample Platforms, 2021 - 2023",
    "Wrote server-side services in Java that exchange JSON with client",
    "applications, and added integration tests for every service.",
    "Education",
    "BSc Computer Engineering, Example University, 2016 - 2020",
    "Skills",
    "Java, Spring Boot, PostgreSQL, Kubernetes, AWS, Git, Maven",
    "Languages",
    "English, Turkish",
)

KEYWORDS_CV_LINES = (
    "Mert Placeholder",
    "mert.placeholder@example.com  +90 555 012 20 32",
    "Denizli, Turkey",
    "Summary",
    "Marketing and sales professional with six years of experience in campaigns,",
    "customer relations and social media.",
    "Experience",
    "Marketing Manager, Fictional Brands, 2022 - Present",
    "Planned advertising campaigns, managed social media accounts and organised",
    "trade fair stands. Negotiated with agencies and reported sales results.",
    "Sales Representative, Sample Retail, 2019 - 2022",
    "Visited customers, prepared offers and met monthly sales targets.",
    "Education",
    "BA Business Administration, Example University, 2014 - 2018",
    "Skills",
    "REST API, SQL, Python, Java, Docker, Linux",
    "Languages",
    "English, Turkish",
)

FRONTEND_CV_LINES = (
    "Elif Mockup",
    "elif.mockup@example.com  +90 555 013 20 33",
    "Remote",
    "Summary",
    "Frontend developer with four years of experience building user interfaces.",
    "Experience",
    "Frontend Developer, Fictional Interfaces, 2023 - Present",
    "Built responsive user interfaces in React and CSS. Improved accessibility",
    "and page load times. Wrote a small Node.js script for static builds.",
    "UI Developer, Sample Studio, 2022 - 2023",
    "Turned design mockups into reusable components.",
    "Education",
    "BSc Computer Programming, Example University, 2017 - 2021",
    "Skills",
    "React, JavaScript, CSS, HTML, Node.js, Figma",
    "Languages",
    "English, Turkish",
)

JUNIOR_CV_LINES = (
    "Can Newgrad",
    "can.newgrad@example.com  +90 555 014 20 34",
    "Denizli, Turkey",
    "Summary",
    "Junior developer with about one year of experience in REST APIs and SQL.",
    "Experience",
    "Backend Developer, Fictional Labs, 2025 - Present",
    "Built small REST APIs in Python and FastAPI and wrote SQL queries for",
    "PostgreSQL reports. Packaged the services with Docker.",
    "Education",
    "BSc Computer Engineering, Example University, 2020 - 2024",
    "Skills",
    "Python, FastAPI, SQL, PostgreSQL, Docker, Git",
    "Languages",
    "English, Turkish",
)

DEMO_CVS = (
    ("demo-cv.pdf", DEMO_CV_LINES),
    ("demo-cv-paraphrase.pdf", PARAPHRASE_CV_LINES),
    ("demo-cv-keywords.pdf", KEYWORDS_CV_LINES),
    ("demo-cv-frontend.pdf", FRONTEND_CV_LINES),
    ("demo-cv-junior.pdf", JUNIOR_CV_LINES),
)


def main() -> None:
    """Write every demo CV next to this script."""
    for file_name, lines in DEMO_CVS:
        output_path = OUTPUT_DIRECTORY / file_name
        output_path.write_bytes(build_pdf(lines))
        print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
