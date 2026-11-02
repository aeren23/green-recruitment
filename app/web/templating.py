"""Jinja2 environment (autoescaping on) and the page-rendering helper."""

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.web import messages, paths

WEB_DIRECTORY = Path(__file__).resolve().parent
TEMPLATE_DIRECTORY = WEB_DIRECTORY / "templates"
STATIC_DIRECTORY = WEB_DIRECTORY / "static"
AUTOESCAPED_EXTENSIONS = ("html",)


def build_environment() -> Environment:
    """The Jinja2 environment: autoescaping for `.html`, UTF-8 templates."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIRECTORY),
        autoescape=select_autoescape(AUTOESCAPED_EXTENSIONS),
    )
    environment.globals["paths"] = paths
    environment.globals["status_labels"] = messages.STATUS_LABELS
    environment.globals["section_options"] = [
        (section_type.value, label)
        for section_type, label in messages.SECTION_TYPE_LABELS.items()
    ]
    return environment


templates = Jinja2Templates(env=build_environment())


def render_page(
    request: Request,
    template_name: str,
    values: dict[str, Any],
    status_code: int = 200,
) -> HTMLResponse:
    """Render a template into an HTML response.

    Args:
        request: The current request.
        template_name: Template path under `app/web/templates`.
        values: Template variables.
        status_code: HTTP status of the response.
    """
    return templates.TemplateResponse(
        request, template_name, values, status_code=status_code
    )
