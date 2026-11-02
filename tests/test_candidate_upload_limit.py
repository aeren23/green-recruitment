"""SC-02: the request-body limit, and proof that a CV never reaches the disk (Green rule 8)."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from candidate_support import (
    AsgiRecorder,
    call_asgi,
    create_open_posting,
    multipart_of_exact_size,
    stored_applications,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline_support import ControlledEmbedder, FakeLlm, build_cv_pdf
from starlette.formparsers import MultiPartParser

from app.config import Settings
from app.main import create_app
from app.web import messages

SMALL_LIMIT = 20_000
LARGE_LIMIT = 3 * 1024 * 1024
OVERSIZE_FILE_BYTES = 1_200_000
TOO_LARGE = 413
SEE_OTHER = 303


class DiskUseSpy:
    """Records every moment an upload could have touched the disk."""

    def __init__(self) -> None:
        self.rollovers = 0
        self.temporary_files = 0


@pytest.fixture
def disk_spy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[DiskUseSpy]:
    """Watch a temp directory, and spy on spooled-file rollover and temp-file creation."""
    watched = tmp_path / "watched-temp"
    watched.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(watched))
    spy = DiskUseSpy()
    original_rollover = tempfile.SpooledTemporaryFile.rollover
    original_mkstemp = tempfile._mkstemp_inner

    def counting_rollover(self: tempfile.SpooledTemporaryFile) -> None:
        spy.rollovers += 1
        original_rollover(self)

    def counting_mkstemp(*args: object, **kwargs: object) -> object:
        spy.temporary_files += 1
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", counting_rollover)
    monkeypatch.setattr(tempfile, "_mkstemp_inner", counting_mkstemp)
    yield spy
    assert list(watched.iterdir()) == []


def limited_app(test_settings: Settings, limit: int) -> FastAPI:
    """An application whose upload limit is `limit` bytes."""
    settings = test_settings.model_copy(update={"max_upload_bytes": limit})
    return create_app(settings, ControlledEmbedder(), FakeLlm())


@pytest.fixture
def small_limit_client(test_settings: Settings) -> Iterator[TestClient]:
    """A client of an app with a 20 kB limit, plus an open posting."""
    application = limited_app(test_settings, SMALL_LIMIT)
    with TestClient(application, follow_redirects=False) as client:
        client.job_id = create_open_posting(application, application.state.embedder, "Job")
        yield client


def post_body(client: TestClient, body: bytes, content_type: str):
    """POST a hand-made multipart body to the apply route."""
    return client.post(
        f"/jobs/{client.job_id}/apply", content=body, headers={"content-type": content_type}
    )


def test_the_limit_never_exceeds_starlettes_in_memory_threshold() -> None:
    """If Starlette lowers its spool size, this fails instead of the disk rule breaking silently."""
    default_limit = Settings(_env_file=None, hash_secret="x", employer_token="y").max_upload_bytes
    assert default_limit <= MultiPartParser.max_file_size


def test_a_small_pdf_is_read_in_memory_and_touches_no_temp_file(
    candidate_client: TestClient, candidate_app: FastAPI, disk_spy: DiskUseSpy
) -> None:
    """A normal upload creates no temporary file and never spools to disk."""
    job_id = create_open_posting(candidate_app, candidate_app.state.embedder, "Job")
    response = upload_cv(candidate_client, job_id, build_cv_pdf("alpha"))
    assert response.status_code == SEE_OTHER
    assert (disk_spy.rollovers, disk_spy.temporary_files) == (0, 0)


def test_exactly_the_limit_is_accepted_and_one_byte_over_is_rejected(
    small_limit_client: TestClient, disk_spy: DiskUseSpy
) -> None:
    """The boundary: a body of the limit passes, one more byte gives 413."""
    application = small_limit_client.app
    pdf = build_cv_pdf("alpha")
    body, content_type = multipart_of_exact_size(pdf, SMALL_LIMIT)
    accepted = post_body(small_limit_client, body, content_type)
    assert accepted.status_code == SEE_OTHER
    assert token_of(accepted)
    over, content_type = multipart_of_exact_size(build_cv_pdf("beta"), SMALL_LIMIT + 1)
    rejected = post_body(small_limit_client, over, content_type)
    assert rejected.status_code == TOO_LARGE
    assert messages.UPLOAD_TOO_LARGE_HEADING in rejected.text
    assert "19 KB" in rejected.text
    assert len(stored_applications(application)) == 1
    assert (disk_spy.rollovers, disk_spy.temporary_files) == (0, 0)


def test_an_oversized_body_is_cut_off_before_it_is_read_further(
    small_limit_client: TestClient, disk_spy: DiskUseSpy
) -> None:
    """The first chunk over the limit ends the request: later chunks are never requested."""
    application = small_limit_client.app
    recorder = AsgiRecorder([b"x" * (SMALL_LIMIT + 1), b"y" * 10, b"z" * 10])
    headers = {"content-type": "multipart/form-data; boundary=b"}
    call_asgi(application, f"/jobs/{small_limit_client.job_id}/apply", headers, recorder)
    assert recorder.status == TOO_LARGE
    assert recorder.receive_calls == 1
    assert sum(1 for item in recorder.sent if item["type"] == "http.response.start") == 1


def test_a_declared_content_length_over_the_limit_is_rejected_without_reading(
    small_limit_client: TestClient, disk_spy: DiskUseSpy
) -> None:
    """A lying or honest Content-Length over the limit is answered before any read."""
    recorder = AsgiRecorder([b"small"])
    headers = {
        "content-type": "multipart/form-data; boundary=b",
        "content-length": str(SMALL_LIMIT + 1),
    }
    call_asgi(small_limit_client.app, f"/jobs/{small_limit_client.job_id}/apply", headers, recorder)
    assert recorder.status == TOO_LARGE
    assert recorder.receive_calls == 0


def test_a_chunked_body_over_the_limit_is_rejected_too(
    small_limit_client: TestClient, disk_spy: DiskUseSpy
) -> None:
    """Without a Content-Length the byte count alone enforces the limit."""
    body, content_type = multipart_of_exact_size(build_cv_pdf("alpha"), SMALL_LIMIT + 1)
    chunks = (body[start : start + 4096] for start in range(0, len(body), 4096))
    response = small_limit_client.post(
        f"/jobs/{small_limit_client.job_id}/apply",
        content=chunks,
        headers={"content-type": content_type},
    )
    assert response.status_code == TOO_LARGE
    assert stored_applications(small_limit_client.app) == []


def test_the_confirmation_form_has_the_same_limit(
    small_limit_client: TestClient, disk_spy: DiskUseSpy
) -> None:
    """The preview form post is capped as well."""
    response = small_limit_client.post("/drafts/some-token", content=b"a=" + b"b" * SMALL_LIMIT)
    assert response.status_code == TOO_LARGE


def test_the_watcher_does_detect_spooling_when_the_limit_is_raised(
    test_settings: Settings, disk_spy: DiskUseSpy, tmp_path: Path
) -> None:
    """Positive control: above Starlette's threshold the spy sees the disk being used."""
    application = limited_app(test_settings, LARGE_LIMIT)
    with TestClient(application, follow_redirects=False) as client:
        job_id = create_open_posting(application, application.state.embedder, "Job")
        body, content_type = multipart_of_exact_size(build_cv_pdf("alpha"), OVERSIZE_FILE_BYTES)
        client.post(
            f"/jobs/{job_id}/apply", content=body, headers={"content-type": content_type}
        )
    assert disk_spy.rollovers >= 1
    # The positive control used the disk on purpose; the fixture's empty-directory
    # check would only pass because Windows deletes temp files on close.
