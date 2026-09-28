import dataclasses
import os

import pytest
from triage.config import Config, load_config
from triage.schemas import Student, Classification, Request
from triage.drafter import TEMPLATE_ENGINE, UnsupportedLanguage, NoTemplate, write_draft

# Config has 11 fields and is built by keyword. No value in this CFG matches
# its own default or a value the code could have hardcoded. The category
# "complaint" and the language "eu" are NOT in config/config.yaml on purpose:
# a category map or an es|en|ca language list hardcoded in production has to
# fail here.
TEMPLATES = {
    "complaint": {
        "es": "Hola {name} ({id}). Incidencia «{subject}» en {programme}. "
              "Estado: {enrolment_status}. Tutor: {tutor}. Progreso: {progress_pct}%.",
        "en": "Hi {name} ({id}). Issue \"{subject}\" on {programme}. "
              "Status: {enrolment_status}. Tutor: {tutor}. Progress: {progress_pct}%.",
        "eu": "Kaixo {name} ({id}). «{subject}» {programme} ikastaroan.",
    }
}
CFG = Config(
    categories=("complaint",),
    confidence_threshold=0.62,      # NOT the yaml's 0.75
    fast_model="cheap-model",
    precise_model="dear-model",
    http_timeout_s=3.5,
    http_retries=4,
    http_backoff_base_s=0.25,
    draft_engine=TEMPLATE_ENGINE,   # the only value that cannot vary
    templates=TEMPLATES,
    db_lock_wait_ms=1234,
    http_max_wait_s=7.0,
)

REQ = Request(id="R1", received_at="2026-09-01T09:00:00", channel="email",
              student_id="STU-1", subject="the video won't load", message="m")
CLASSIF = Classification(category="complaint", urgency=2, confidence=0.9, reason="ok")


def student(language):
    return Student(student_id="STU-1", name="Nuria", email="n@e.com",
                   programme="MSc in AI", enrolment_status="active",
                   start_date="2026-01-15", progress_pct=61, tutor="Fernando",
                   language=language)


def _cfg_with_template(language, text, category="complaint"):
    return dataclasses.replace(CFG, templates={category: {language: text}})


# --- The student's real language + the request's/student's real data -------

def test_uses_the_student_language_and_real_data():
    cases = [("es", "Incidencia"), ("en", "Issue"), ("eu", "ikastaroan")]
    assert len(cases) > 0  # the loop below would prove nothing with an empty list
    for language, fragment in cases:
        st = student(language)
        body = write_draft(REQ, st, CLASSIF, CFG)
        assert fragment in body
        assert st.name in body
        assert st.programme in body
        assert REQ.id in body
        assert REQ.subject in body
        if language in ("es", "en"):
            # Only this CFG's es/en templates use these placeholders; eu is
            # deliberately shorter.
            assert st.enrolment_status in body
            assert st.tutor in body
            assert str(st.progress_pct) in body


def test_includes_the_request_subject():
    assert REQ.subject in write_draft(REQ, student("en"), CLASSIF, CFG)


# Seven MUTUALLY DISTINCT fixture values, none a substring of another, none
# equal to any other fixture value in the whole file. 83 does not appear as a
# digit sequence anywhere else in the file.
UNIQUE_NAME = "Sapphire"
UNIQUE_PROGRAMME = "Quartz"
UNIQUE_STATUS = "Beryl"
UNIQUE_TUTOR = "Topaz"
UNIQUE_PROGRESS = 83
UNIQUE_SUBJECT = "Amethyst"
UNIQUE_ID = "Peridot"

LABELLED_TEMPLATE = (
    "N=[{name}] P=[{programme}] E=[{enrolment_status}] T=[{tutor}] "
    "G=[{progress_pct}] A=[{subject}] I=[{id}]"
)
LABELLED_CFG = dataclasses.replace(
    CFG, templates={"complaint": {"en": LABELLED_TEMPLATE}}
)
LABELLED_REQ = Request(
    id=UNIQUE_ID, received_at="2026-09-01T09:00:00", channel="email",
    student_id="STU-X", subject=UNIQUE_SUBJECT, message="m",
)
LABELLED_STUDENT = Student(
    student_id="STU-X", name=UNIQUE_NAME, email="sapphire@e.com",
    programme=UNIQUE_PROGRAMME, enrolment_status=UNIQUE_STATUS,
    start_date="2026-01-15", progress_pct=UNIQUE_PROGRESS, tutor=UNIQUE_TUTOR,
    language="en",
)


def test_each_placeholder_lands_in_its_exact_position():
    # `assert value in body` stays true even when the value lands in ANOTHER
    # field's slot (e.g. programme <-> tutor swapped): the value is still
    # present somewhere in the text. Only an == comparison against the full
    # expected string detects that a value landed in the wrong position.
    expected = (
        f"N=[{UNIQUE_NAME}] P=[{UNIQUE_PROGRAMME}] E=[{UNIQUE_STATUS}] "
        f"T=[{UNIQUE_TUTOR}] G=[{UNIQUE_PROGRESS}] A=[{UNIQUE_SUBJECT}] I=[{UNIQUE_ID}]"
    )
    body = write_draft(LABELLED_REQ, LABELLED_STUDENT, CLASSIF, LABELLED_CFG)
    assert body == expected


# --- Rejecting a language / category with no template ------------------------

def test_language_with_no_template_is_rejected():
    # "de" is not in "complaint"'s language dict NOR in the real yaml: it
    # cannot slip through by chance.
    with pytest.raises(UnsupportedLanguage):
        write_draft(REQ, student("de"), CLASSIF, CFG)


def test_unsupported_language_is_a_no_template():
    # A caller that only catches NoTemplate has to catch the language case
    # too. A named design decision; without this test, removing the
    # inheritance would break nothing.
    assert issubclass(UnsupportedLanguage, NoTemplate)


def test_category_with_no_template_is_rejected():
    # This is NoTemplate, NOT UnsupportedLanguage. The real cause is the
    # missing category, not the language.
    other = Classification(category="withdrawal", urgency=3, confidence=0.9, reason="ok")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), other, CFG)
    assert type(info.value) is NoTemplate
    assert not isinstance(info.value, UnsupportedLanguage)


# --- A badly written template in the YAML cannot bring the process down -----

def test_unknown_placeholder_in_template_does_not_break_the_process():
    cfg = _cfg_with_template("en", "Hi {name}, end date: {end_date}.")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), CLASSIF, cfg)
    assert type(info.value) is NoTemplate
    assert not isinstance(info.value, UnsupportedLanguage)
    assert "complaint" in str(info.value)
    assert "en" in str(info.value)


def test_positional_index_in_template_does_not_break_the_process():
    cfg = _cfg_with_template("en", "Hi {0}.")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), CLASSIF, cfg)
    assert type(info.value) is NoTemplate


def test_unbalanced_brace_in_template_does_not_break_the_process():
    cfg = _cfg_with_template("en", "Hi {name")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), CLASSIF, cfg)
    assert type(info.value) is NoTemplate


def test_template_empty_after_stripping_is_rejected():
    cfg = _cfg_with_template("en", "   \n  ")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), CLASSIF, cfg)
    assert type(info.value) is NoTemplate


# --- cfg.draft_engine really is read ------------------------------------------

def test_unknown_engine_is_rejected():
    cfg = dataclasses.replace(CFG, draft_engine="model")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), CLASSIF, cfg)
    assert type(info.value) is NoTemplate
    assert "model" in str(info.value)


# --- The real config has to render everything, with no live placeholders ----

def test_renders_every_template_in_the_real_config():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    cfg = load_config(path)
    yaml_languages = ("es", "en", "ca")

    assert len(cfg.categories) > 0
    assert len(yaml_languages) > 0

    bodies = []
    for category in cfg.categories:
        for language in yaml_languages:
            classif = Classification(category=category, urgency=1, confidence=0.99, reason="ok")
            body = write_draft(REQ, student(language), classif, cfg)
            assert body != ""
            assert "{" not in body
            bodies.append(body)

    assert len(bodies) > 0
    assert len(bodies) == len(cfg.categories) * len(yaml_languages)
    assert len(bodies) == 12


# --- The narrow except cannot become a broad relabelling of any failure ------
# `drafter.py` has a single try/except: this is its only sibling and it is
# already covered here.

class _ExplodingTemplate(str):
    """A 'str' whose .format() raises an error UNRELATED to template formatting.

    It simulates a real code bug (e.g. a TypeError from a future refactor)
    that has nothing to do with malformed template data.
    """

    def __new__(cls, text, exception):
        obj = super().__new__(cls, text)
        obj._exception = exception
        return obj

    def format(self, *args, **kwargs):
        raise self._exception("internal failure unrelated to the template")


@pytest.mark.parametrize("exception", [RuntimeError, TypeError])  # not just one type
def test_an_exception_unrelated_to_formatting_is_not_silenced(exception):
    # If write_draft()'s except were widened (to `except Exception`, or to a
    # tuple that simply includes one more type), this error would be caught
    # and relabelled as NoTemplate, hiding the real failure. With the narrow
    # except (KeyError, IndexError, ValueError), it has to propagate untouched,
    # whatever its type.
    cfg = dataclasses.replace(
        CFG, templates={"complaint": {"en": _ExplodingTemplate("irrelevant", exception)}}
    )
    with pytest.raises(exception):
        write_draft(REQ, student("en"), CLASSIF, cfg)


# --- Neither language nor category tolerates case variants

def test_upper_case_language_does_not_silently_fall_back():
    # "en" IS in CFG's templates; "EN" is not. A case-insensitive fallback
    # would send a student whose real language is something else an English
    # template without anyone noticing.
    with pytest.raises(UnsupportedLanguage):
        write_draft(REQ, student("EN"), CLASSIF, CFG)
    with pytest.raises(UnsupportedLanguage):  # nor does it tolerate spaces
        write_draft(REQ, student(" en"), CLASSIF, CFG)


def test_upper_case_category_does_not_silently_fall_back():
    other = Classification(category="COMPLAINT", urgency=2, confidence=0.9, reason="ok")
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), other, CFG)
    assert type(info.value) is NoTemplate
    assert not isinstance(info.value, UnsupportedLanguage)
    with_spaces = Classification(category=" complaint ", urgency=2, confidence=0.9, reason="ok")
    with pytest.raises(NoTemplate):  # nor does it tolerate spaces
        write_draft(REQ, student("en"), with_spaces, CFG)


# --- The engine has to be EXACT equality, not a near miss

@pytest.mark.parametrize("engine", [
    "template_experimental",  # .startswith(TEMPLATE_ENGINE) would accept it
    "Template",               # .lower() would accept it
    " template ",             # .strip() would accept it
    "template ",              # .rstrip() would accept it (space on the right only)
    " template",              # .lstrip() would accept it (space on the left only)
    "",                       # an inverted `in` check would accept it
])
def test_nearly_correct_engine_is_rejected(engine):
    # Pins that the check is an exact ==: neither startswith, strip, lower nor
    # an `in` can replace it without one of these variants slipping through.
    cfg = dataclasses.replace(CFG, draft_engine=engine)
    with pytest.raises(NoTemplate) as info:
        write_draft(REQ, student("en"), CLASSIF, cfg)
    assert type(info.value) is NoTemplate
