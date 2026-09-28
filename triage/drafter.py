"""Reply drafts from configurable templates.

When the mock model service is not asked for JSON, it always returns the same
fixed English text, ignoring both the prompt and the student's language. A
model-written draft could not meet "in the student's language". Templates
can: they are deterministic, translatable and cost zero tokens. That is why
the only engine implemented is 'template'.
"""
from .config import Config
from .schemas import Classification, Request, Student

# The only draft engine implemented. If cfg.draft_engine holds anything else
# (e.g. 'model'), it is rejected instead of pretending to support it.
TEMPLATE_ENGINE = "template"


class NoTemplate(Exception):
    """A draft cannot be written with the current configuration."""


class UnsupportedLanguage(NoTemplate):
    """There are templates for the category, but none in the student's language."""


def write_draft(req: Request, student: Student, classif: Classification,
                cfg: Config) -> str:
    if cfg.draft_engine != TEMPLATE_ENGINE:
        raise NoTemplate(
            f"Draft engine {cfg.draft_engine!r} not supported: only "
            f"{TEMPLATE_ENGINE!r} is implemented."
        )

    by_language = cfg.templates.get(classif.category)
    if not by_language:
        raise NoTemplate(
            f"No templates configured for category {classif.category!r}."
        )

    template = by_language.get(student.language)
    if not template:
        raise UnsupportedLanguage(
            f"No template in {student.language!r} for category "
            f"{classif.category!r}. We do not improvise a reply in a language "
            f"we have no template for."
        )

    data = {
        "name": student.name,
        "programme": student.programme,
        "enrolment_status": student.enrolment_status,
        "tutor": student.tutor,
        "progress_pct": student.progress_pct,
        "subject": req.subject,
        "id": req.id,
    }
    try:
        body = template.format(**data)
    except (KeyError, IndexError, ValueError) as e:
        raise NoTemplate(
            f"Malformed template for category {classif.category!r} and "
            f"language {student.language!r}: {e!r}."
        ) from e

    body = body.strip()
    if not body:
        raise NoTemplate(
            f"The template for category {classif.category!r} and language "
            f"{student.language!r} produces an empty body."
        )
    return body
