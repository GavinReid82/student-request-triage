"""Classification with a two-tier ladder.

A single threshold drives two decisions: if the cheap model's confidence falls
below it, or its output does not match the schema (incomplete JSON, a
category outside the enum, a corrupted HTTP envelope...), that request, and
only that one, is reclassified with the precise model. If the precise model
also fails the schema, the request gives up: InvalidOutput is raised and the
caller decides what to do with a request no model could classify. If the
precise model validates but its confidence is still low, there is no third
attempt here: the result is returned as it is and recorded, because the next
decision belongs to a person, not to the system.

Nothing is ever corrected: neither an out-of-catalogue category, nor a
malformed HTTP envelope, nor a missing cost is turned into an invented value
without leaving a trace. Everything the model or the transport fails to
deliver in usable form ends up as an explicit incident.
"""
import math
import time

from .http import send
from .schemas import Classification, Request
from .model_output import InvalidOutput, validate_output

SYSTEM = (
    "You classify student requests for an online academy. "
    "Reply only with a JSON object with the keys: "
    "category (one of: {categories}), urgency (integer 1, 2 or 3), "
    "confidence (decimal between 0 and 1) and reason (short text)."
)

# Incident types born in this module, named here so the text that ends up in
# the database is not a loose literal repeated around the code.
TYPE_LOW_CONFIDENCE = "low_confidence"
TYPE_LOW_CONFIDENCE_PERSISTS = "low_confidence_persists"
TYPE_INVALID_OUTPUT = "invalid_model_output"
TYPE_COST_NOT_REPORTED = "cost_not_reported"

# Precision used to accumulate and return the cost. Kept as a constant so it
# is not a magic number repeated in `classify`.
COST_DECIMALS = 8


class ClassificationFailed(InvalidOutput):
    """No model produced a valid output. Carries the cost already spent.

    A subclass of InvalidOutput: anyone catching the generic exception keeps
    working, and anyone who needs the cost spent before giving up (so it is
    not lost from the cost summary) can catch this specific type.
    """

    def __init__(self, message: str, cost_eur: float):
        super().__init__(message)
        self.cost_eur = cost_eur


def build_prompt(req: Request, categories: tuple[str, ...]) -> list[dict]:
    """The model service matches on the subject and on the REQ-2026-NNNN id,
    so both have to appear verbatim in the prompt."""
    return [
        {"role": "system", "content": SYSTEM.format(categories=", ".join(categories))},
        {"role": "user", "content": (
            f"Request {req.id}\n"
            f"Channel: {req.channel}\n"
            f"Subject: {req.subject}\n"
            f"Message: {req.message}"
        )},
    ]


def _content(data: dict) -> str:
    """Extracts the answer text or raises InvalidOutput with the reason.

    A 200 with a corrupted envelope (empty `choices`, missing keys, a
    `content` that is not text) is as invalid as JSON that fails the schema:
    it must not escape as an uncaught KeyError/IndexError/TypeError and bring
    the process down for that request.
    """
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise InvalidOutput(
            f"The answer has no choices[0].message.content: {e}") from e
    if not isinstance(content, str):
        raise InvalidOutput(
            f"The content is not text but {type(content).__name__}.")
    return content


class Classifier:
    """Calls the cheap model and, when needed, the precise one.

    `sleep` exists because `send` binds its own `sleep=time.sleep` when the
    function is defined, so patching `triage.http.time.sleep` afterwards does
    not intercept it: a replaceable seam has to be exposed and forwarded
    explicitly on every call to `send`.
    """

    def __init__(self, client, env, cfg, store, sleep=time.sleep):
        self._client = client
        self._env = env
        self._cfg = cfg
        self._store = store
        self._sleep = sleep

    def _call(self, req: Request, model: str) -> tuple[dict, float]:
        """Sends the request to the given model. Returns (json_envelope, cost_eur).

        It does not extract the content here: that can fail in ways that have
        to be treated as invalid output, and that decision lives in
        `classify`, next to the one for the content's own JSON, so both
        failure paths (corrupted envelope, corrupted business JSON) are
        handled the same way.
        """
        response = send(
            self._client, "POST", f"{self._env.model_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._env.token}"},
            body={
                "model": model,
                "messages": build_prompt(req, self._cfg.categories),
                "response_format": {"type": "json_object"},
            },
            timeout_s=self._cfg.http_timeout_s,
            retries=self._cfg.http_retries,
            backoff_base_s=self._cfg.http_backoff_base_s,
            max_wait_s=self._cfg.http_max_wait_s,
            sleep=self._sleep,
            record=lambda kind, detail: self._store.incident(req.id, kind, detail),
        )
        data = response.json()
        cost = self._cost(data, model, req)
        return data, cost

    def _cost(self, data: dict, model: str, req: Request) -> float:
        """Cost of the call, or 0.0 with an incident if it is not usable.

        A `usage.cost_eur` that is missing, non-numeric or non-finite (NaN/inf
        parse as float without error) becomes 0.0, but never silently: the
        cost per 1,000 requests is a deliverable, and a call silently set to
        zero understates it without anyone noticing.
        """
        usage = data.get("usage")
        raw = usage.get("cost_eur") if isinstance(usage, dict) else None
        if (isinstance(raw, (int, float)) and not isinstance(raw, bool)
                and math.isfinite(raw)):
            return float(raw)
        self._store.incident(
            req.id, TYPE_COST_NOT_REPORTED,
            f"{model}: usage.cost_eur missing or non-numeric ({raw!r}); "
            "0.0 is counted for this call.",
        )
        return 0.0

    def classify(self, req: Request) -> tuple[Classification, str, float]:
        """Returns (classification, model_used, accumulated_cost).

        Raises ClassificationFailed (a subclass of InvalidOutput) if neither
        model produces an output that matches the schema; the exception
        carries the cost already spent on both calls, because the money was
        spent even though the classification is unusable.
        """
        total_cost = 0.0

        data, cost = self._call(req, self._cfg.fast_model)
        total_cost += cost
        try:
            classification = validate_output(_content(data), self._cfg.categories)
            if classification.confidence >= self._cfg.confidence_threshold:
                return (classification, self._cfg.fast_model,
                        round(total_cost, COST_DECIMALS))
            self._store.incident(
                req.id, TYPE_LOW_CONFIDENCE,
                f"{self._cfg.fast_model} returned confidence "
                f"{classification.confidence}, below the threshold "
                f"{self._cfg.confidence_threshold}; retrying with "
                f"{self._cfg.precise_model}.",
            )
        except InvalidOutput as e:
            self._store.incident(
                req.id, TYPE_INVALID_OUTPUT, f"{self._cfg.fast_model}: {e}")

        data, cost = self._call(req, self._cfg.precise_model)
        total_cost += cost
        try:
            classification = validate_output(_content(data), self._cfg.categories)
            # Symmetrical with the first tier (the twin rule): the persistent
            # low-confidence incident lives INSIDE the try, next to the
            # first tier's low-confidence one, so both post-validation
            # incident paths are treated exactly the same way by the except
            # below.
            if classification.confidence < self._cfg.confidence_threshold:
                self._store.incident(
                    req.id, TYPE_LOW_CONFIDENCE_PERSISTS,
                    f"{self._cfg.precise_model} returned confidence "
                    f"{classification.confidence}, still below the threshold "
                    f"{self._cfg.confidence_threshold} after reclassification; "
                    "left for human review.",
                )
        except InvalidOutput as e:
            self._store.incident(
                req.id, TYPE_INVALID_OUTPUT, f"{self._cfg.precise_model}: {e}")
            raise ClassificationFailed(
                f"No model produced a valid output: {e}",
                round(total_cost, COST_DECIMALS),
            ) from e

        return classification, self._cfg.precise_model, round(total_cost, COST_DECIMALS)
