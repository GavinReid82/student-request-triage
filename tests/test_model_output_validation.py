"""The main test.

This is the one place in the system where a third party hands us garbage that
looks like valid data. A failure here breaks nothing visible: it persists
corrupt data and presents it in the summary as if it were true.
"""
import pytest
from triage.model_output import InvalidOutput, extract_json, validate_output

CATEGORIES = ("content", "technical", "admin", "withdrawal")

CLEAN = '{"category": "technical", "urgency": 3, "confidence": 0.95, "reason": "ok"}'
WITH_NOISE = (
    "Sure, here is the classification:\n```json\n"
    '{"category": "content", "urgency": 1, "confidence": 0.93, "reason": "ok"}'
    "\n```\nHope this helps."
)
URGENCY_AS_TEXT = '{"category": "content", "urgency": "high", "confidence": 0.92, "reason": "ok"}'
INVENTED_CATEGORY = '{"category": "sales", "urgency": 2, "confidence": 0.91, "reason": "ok"}'
BROKEN_JSON = '{"category": "content", "urgency": 1, "confidence":'

def test_clean_json_is_accepted():
    c = validate_output(CLEAN, CATEGORIES)
    assert (c.category, c.urgency, c.confidence) == ("technical", 3, 0.95)

def test_json_wrapped_in_prose_and_fences_is_extracted():
    c = validate_output(WITH_NOISE, CATEGORIES)
    assert c.category == "content"
    assert c.urgency == 1

@pytest.mark.parametrize("text,fragment", [
    (URGENCY_AS_TEXT, "urgency"),
    (INVENTED_CATEGORY, "sales"),
    (BROKEN_JSON, "JSON"),
])
def test_faulty_outputs_are_rejected_with_a_reason(text, fragment):
    with pytest.raises(InvalidOutput) as e:
        validate_output(text, CATEGORIES)
    assert fragment in str(e.value)

def test_textual_urgency_is_not_turned_into_a_number():
    """Explicit: 'high' is not translated to 3. Guessing is worse than failing.

    The reason fits on one line and says what value arrived: it ends up in the
    review queue that a person reads in a spreadsheet.
    """
    with pytest.raises(InvalidOutput) as e:
        validate_output(URGENCY_AS_TEXT, CATEGORIES)
    assert "urgency='high'" in str(e.value)
    assert "\n" not in str(e.value)

def test_category_outside_the_enum_is_not_approximated():
    """Explicit: 'sales' is not mapped to the closest category."""
    with pytest.raises(InvalidOutput):
        validate_output(INVENTED_CATEGORY, CATEGORIES)

def test_confidence_out_of_range_is_rejected():
    with pytest.raises(InvalidOutput):
        validate_output('{"category":"withdrawal","urgency":3,"confidence":1.7,"reason":"x"}', CATEGORIES)

def test_text_with_no_json_at_all_is_rejected():
    with pytest.raises(InvalidOutput):
        validate_output("Hello, thank you for writing to us.", CATEGORIES)

# --- Fix 1: confidence is not coerced from string to number ----------------

def test_confidence_as_string_is_rejected():
    """'0.9' as a string is not silently turned into a float."""
    with pytest.raises(InvalidOutput):
        validate_output(
            '{"category":"content","urgency":2,"confidence":"0.9","reason":"ok"}',
            CATEGORIES,
        )

def test_confidence_as_bare_integer_one_is_accepted():
    c = validate_output(
        '{"category":"content","urgency":2,"confidence":1,"reason":"ok"}',
        CATEGORIES,
    )
    assert c.confidence == 1

def test_confidence_as_bare_integer_zero_is_accepted():
    c = validate_output(
        '{"category":"content","urgency":2,"confidence":0,"reason":"ok"}',
        CATEGORIES,
    )
    assert c.confidence == 0

# --- Fix 2: the brace-counting logic has a direct test ---------------------

def test_closing_brace_inside_a_string_does_not_close_the_object():
    data = extract_json('{"reason": "uses } a brace"}')
    assert data == {"reason": "uses } a brace"}

def test_escaped_quote_inside_a_string_is_handled():
    data = extract_json('{"reason": "inner \\" quote"}')
    assert '"' in data["reason"]

def test_exact_wrapper_from_the_mock_service_is_extracted():
    text = (
        "Sure, here is the classification:\n```json\n"
        '{"category": "content", "urgency": 1, "confidence": 0.9, "reason": "ok"}'
        "\n```\nHope this helps."
    )
    data = extract_json(text)
    assert data == {
        "category": "content",
        "urgency": 1,
        "confidence": 0.9,
        "reason": "ok",
    }

def test_unbalanced_json_is_rejected():
    with pytest.raises(InvalidOutput):
        extract_json('{"a": 1')

def test_extract_json_with_no_json_at_all_is_rejected():
    with pytest.raises(InvalidOutput):
        extract_json("there is no json here")

# --- Fix 3: retry from the next brace if the first does not balance --------

def test_stray_brace_in_prose_does_not_stop_the_real_object_being_found():
    data = extract_json('Use { to open. Here: {"a": 1}')
    assert data == {"a": 1}

def test_two_complete_objects_returns_the_first():
    data = extract_json('{"a": 1} {"b": 2}')
    assert data == {"a": 1}
