# -*- coding: utf-8 -*-
"""
Mock model service.

Exposes an endpoint compatible with the OpenAI chat completions API, so any
HTTP client (or the official SDK) can talk to it by pointing base_url here.

Available models (see /v1/models for prices):
    sim-fast     — cheap and less reliable
    sim-precise  — expensive and more reliable

How to ask for each kind of output:
    response_format={"type": "json_object"}  -> structured classification
    (no response_format)                     -> reply draft as plain text

Usage and running cost: GET /v1/stats

The answers are canned per request subject, and some of them are broken on
purpose (JSON wrapped in prose, a wrong type, a category outside the list,
slow responses, an occasional 503) so the client has to cope with them.
"""
import asyncio, json, re, time, threading, itertools
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock model service", version="1.0.0")
TOKEN = "demo-token-2026"

# Simulated prices, in euros per million tokens.
PRICES = {
    "sim-fast":    {"input": 0.10, "output": 0.40},
    "sim-precise": {"input": 1.00, "output": 5.00},
}

_lock = threading.Lock()
_counter = itertools.count(1)
_log = []   # one entry per call: model, tokens, cost, detected request

# ---------------------------------------------------------------------------
# Behaviour per request.
# ---------------------------------------------------------------------------
CATEGORIES = ("content", "technical", "admin", "withdrawal")

# subject (lower case) -> (category, urgency, base_confidence)
CASES = {
 "invoice for september":                           ("admin", 2, 0.96),
 "invoice in my company's name":                    ("admin", 2, 0.95),
 "enrolment certificate":                           ("admin", 2, 0.96),
 "course completion certificate":                   ("admin", 2, 0.94),
 "duplicate invoice":                               ("admin", 3, 0.91),
 "certificate of teaching hours":                   ("admin", 2, 0.93),
 "i can't log in to the platform":                  ("technical", 3, 0.95),
 "the video for class 12 won't load":               ("technical", 2, 0.93),
 "error uploading my assignment":                   ("technical", 3, 0.94),
 "i'm not getting platform emails":                 ("technical", 2, 0.90),
 "tool licence expired":                            ("technical", 3, 0.92),
 "my progress isn't showing":                       ("technical", 2, 0.89),
 "two-factor authentication not working":           ("technical", 3, 0.93),
 "problema con el aula virtual en directo":         ("technical", 2, 0.88),
 "question about exercise 4 in module 2":           ("content", 1, 0.94),
 "difference between two concepts in the syllabus": ("content", 1, 0.92),
 "can my final project be about my company?":       ("content", 1, 0.90),
 "reading list for module 5":                       ("content", 1, 0.93),
 "marking of assignment 2":                         ("content", 2, 0.86),
 "optional exercise in module 6":                   ("content", 1, 0.91),
 "materials from yesterday's session":              ("content", 1, 0.92),
 "question about a case study":                     ("content", 1, 0.90),
 "is there a review session before the exam?":      ("content", 1, 0.91),
 "i want to withdraw from the programme":           ("withdrawal", 3, 0.97),
 "withdrawal for work reasons":                     ("withdrawal", 3, 0.96),
 "cancel enrolment":                                ("withdrawal", 3, 0.95),
 "withdrawal and refund":                           ("withdrawal", 3, 0.96),
 "no puedo continuar":                              ("withdrawal", 3, 0.87),
 "question about the final project deadline":       ("content", 2, 0.89),
 "consulta sobre el calendari del mòdul 4":         ("content", 1, 0.88),
 "pending matter":                                  ("content", 1, 0.31),   # ambiguous
 "quick thing":                                     ("content", 1, 0.24),   # ambiguous
 "cambio de datos bancarios":                       ("admin", 2, 0.93),
 "postpone a submission":                           ("admin", 2, 0.85),
 "access to materials after finishing":             ("content", 1, 0.88),
 "the platform is very slow":                       ("technical", 2, 0.91),
 "can i change group?":                             ("admin", 2, 0.89),
 "question about credit transfer":                  ("admin", 1, 0.87),
 "question about payment":                          ("admin", 2, 0.90),
}

# Deliberately faulty answers, by subject.
JSON_WITH_NOISE   = {"reading list for module 5"}           # JSON wrapped in prose
URGENCY_AS_TEXT   = {"materials from yesterday's session"}  # wrong type
INVENTED_CATEGORY = {"the platform is very slow"}           # outside the enum
SLOW              = {"error uploading my assignment", "withdrawal and refund"}
# Cheap model only: one extra case returns broken JSON.
BROKEN_FAST_ONLY  = {"question about a case study"}

def _detect(text: str):
    t = text.lower()
    m = re.search(r"req-2026-\d{4}", t)
    req_id = m.group(0).upper() if m else None
    for subject in CASES:
        if subject in t:
            return subject, req_id
    return None, req_id

def _tok(s: str) -> int:
    return max(1, len(s) // 4)

@app.get("/v1/models", summary="Available models and their prices")
async def models():
    return {"object": "list", "data": [
        {"id": "sim-fast", "object": "model", "owned_by": "mock",
         "description": "Cheaper and faster. Less reliable on hard cases.",
         "price_eur_per_million": PRICES["sim-fast"]},
        {"id": "sim-precise", "object": "model", "owned_by": "mock",
         "description": "More expensive and slower. More reliable.",
         "price_eur_per_million": PRICES["sim-precise"]},
    ]}

@app.get("/v1/stats", summary="Usage and running cost of this instance")
async def stats():
    with _lock:
        log = list(_log)
    by_model = {}
    for r in log:
        d = by_model.setdefault(r["model"], {"calls": 0, "input_tokens": 0,
                                             "output_tokens": 0, "cost_eur": 0.0})
        d["calls"] += 1
        d["input_tokens"] += r["input_tokens"]
        d["output_tokens"] += r["output_tokens"]
        d["cost_eur"] = round(d["cost_eur"] + r["cost_eur"], 6)
    return {"total_calls": len(log),
            "total_cost_eur": round(sum(r["cost_eur"] for r in log), 6),
            "by_model": by_model,
            "detail": log}

@app.post("/v1/chat/completions", summary="Compatible with the OpenAI API")
async def completions(request: Request, authorization: str | None = Header(default=None)):
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, detail="Missing or wrong token.")
    body = await request.json()

    model = body.get("model", "sim-fast")
    if model not in PRICES:
        return JSONResponse(status_code=400, content={"error": {
            "message": f"Unknown model: {model}. See /v1/models.",
            "type": "invalid_request_error"}})

    messages = body.get("messages") or []
    prompt = "\n".join(str(m.get("content", "")) for m in messages)
    wants_json = (body.get("response_format") or {}).get("type") == "json_object"
    subject, req_id = _detect(prompt)
    n = next(_counter)

    # The provider fails now and then, like any real provider.
    if n == 9:
        return JSONResponse(status_code=503, content={"error": {
            "message": "The service is temporarily unavailable.",
            "type": "server_error"}}, headers={"Retry-After": "1"})

    if subject in SLOW:
        await asyncio.sleep(3)
    else:
        await asyncio.sleep(0.25)

    if subject is None:
        category, urgency, confidence = "content", 1, 0.42
    else:
        category, urgency, confidence = CASES[subject]
        if model == "sim-fast":
            confidence = round(max(0.05, confidence - 0.15), 2)

    if wants_json:
        obj = {"category": category, "urgency": urgency,
               "confidence": confidence,
               "reason": f"The request text fits the {category} category."}
        if subject in INVENTED_CATEGORY:
            obj["category"] = "sales"                # outside the allowed enum
        if subject in URGENCY_AS_TEXT:
            obj["urgency"] = "high"                  # wrong type
        content = json.dumps(obj, ensure_ascii=False)
        if subject in JSON_WITH_NOISE or (model == "sim-fast" and subject in BROKEN_FAST_ONLY):
            content = ("Sure, here is the classification:\n```json\n"
                       + content + "\n```\nHope this helps.")
    else:
        content = (
            "Hello,\n\nThank you for writing to us. We have received your request and "
            "classified it as {cat}. A member of the relevant team will be in touch "
            "shortly.\n\nBest regards,\nThe Academic Team"
        ).format(cat=category)

    t_in, t_out = _tok(prompt), _tok(content)
    p = PRICES[model]
    cost = round(t_in / 1e6 * p["input"] + t_out / 1e6 * p["output"], 8)
    with _lock:
        _log.append({"n": n, "model": model, "request": req_id or subject,
                     "kind": "classification" if wants_json else "draft",
                     "input_tokens": t_in, "output_tokens": t_out,
                     "cost_eur": cost})

    return {
        "id": f"chatcmpl-mock-{n}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": t_in, "completion_tokens": t_out,
                  "total_tokens": t_in + t_out, "cost_eur": cost},
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
