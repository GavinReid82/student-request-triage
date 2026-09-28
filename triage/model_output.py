"""Extraction and validation of the model's answer.

System rule: nothing is corrected here. If the output does not match the
schema, it is rejected and the request goes to human review. Turning "high"
into 3, or "sales" into the closest category, would be inventing the model's
judgement and persisting a value nobody decided.
"""
import json

from pydantic import ValidationError

from .schemas import Classification


class InvalidOutput(Exception):
    """The model's answer does not match the agreed schema."""


def extract_json(text: str) -> dict:
    """Returns the first balanced JSON object that appears in the text.

    The model sometimes wraps its answer in prose and ```json fences. The
    object is found by counting braces, ignoring those inside strings, rather
    than by trimming at markers that may not be there.
    """
    start = text.find("{")
    if start == -1:
        raise InvalidOutput("The answer contains no JSON object.")

    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start : i + 1]
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(data, dict):
                        break
                    return data
        start = text.find("{", start + 1)

    raise InvalidOutput("Incomplete JSON: the object never closes.")


# How much of the rejected value is shown. The reason ends up in a spreadsheet
# that a person reads: seeing what arrived is enough, not the whole object.
MAX_VALUE_IN_REASON = 40


def _summarise_errors(e: ValidationError) -> str:
    """One readable line per field: which field, what value arrived, and why it fails.

    Pydantic's default text runs over several lines and ends with a link to
    its documentation; as the reason for an escalation in a CSV it is noise.
    """
    parts = []
    for err in e.errors():
        field = ".".join(map(str, err["loc"])) or "model"
        if err["type"] == "missing":
            parts.append(f"{field} (missing)")
        else:
            value = repr(err.get("input"))[:MAX_VALUE_IN_REASON]
            parts.append(f"{field}={value} ({err['msg']})")
    return "; ".join(parts)


def validate_output(text: str, categories: tuple[str, ...]) -> Classification:
    """Extracts, parses and validates. Raises InvalidOutput with the specific reason."""
    data = extract_json(text)

    try:
        classification = Classification(**data)
    except ValidationError as e:
        raise InvalidOutput(f"Schema violated in {_summarise_errors(e)}.") from e

    if classification.category not in categories:
        raise InvalidOutput(
            f"Category outside the allowed enum: {classification.category!r}. "
            f"Allowed: {', '.join(categories)}."
        )
    return classification
