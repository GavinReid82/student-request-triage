# Student Request Triage

An AI automation pipeline that triages student support requests for an online
academy. For each request it validates the input, looks the student up in a
student records API, classifies the request with an LLM (category, urgency,
confidence and a short reason), and then either writes a reply draft in the
student's own language or sends the request to a human review queue. Everything
is stored in SQLite, and every run ends with a summary and the measured cost.

The interesting part is not the happy path. The two external services it talks to
are deliberately unreliable: the student API rate-limits, returns a 500 and
answers slowly on some calls, and the LLM returns a 503, wraps its JSON in prose,
sends a string where a number should be and invents a category that does not
exist. The project is about handling that properly: idempotent re-runs, fault
tolerance, schema-validated model output, configuration outside the code, and a
cost per 1,000 requests that is measured rather than guessed.

**Stack:** Python 3.12, httpx, Pydantic, SQLite, PyYAML, pytest + respx, Docker Compose.
The LLM and the student API are local mock services (FastAPI), so the whole thing
runs offline with no API keys.

All the data is fictional.

---

## 1. Run it

```bash
cp .env.example .env          # once: secrets and paths, kept out of the repo
docker compose up --exit-code-from triage
```

This starts the student API and the model service, waits until both answer,
processes the whole input file, exports the CSVs and ends by printing the summary
and the cost. The first start is slower because all three containers install their
dependencies. Plain `docker compose up` also works; it just leaves the two services
running afterwards.

What ends up in `output/`:

| File | Purpose |
|---|---|
| `triage.db` | SQLite, the source of truth: requests, drafts and incidents. |
| `results.csv` | Every request. It opens in Excel or imports into Google Sheets. |
| `review_queue.csv` | The human review queue, most urgent first, with a column saying why each row is there. |

**Running it twice** shows the idempotency: the second run makes no model calls,
writes no new drafts and adds no cost. To start from scratch, run `rm -rf output/*`.

Exit codes: `0` the run completed (even if some requests ended as `failed`, because
they are recorded and the next run picks them up), `1` store failure, `2` invalid
configuration, `3` a service never came up.

**Tests** (Python 3.12, no Docker or network needed):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q           # 190 tests
```

**Without Docker**, start the two mock services and point the pipeline at them:

```bash
.venv/bin/pip install fastapi==0.115.6 "uvicorn[standard]==0.34.0"
(cd mock_student_api && ../.venv/bin/uvicorn main:app --port 8080 &)
(cd mock_llm && ../.venv/bin/uvicorn main:app --port 9000 &)
TRIAGE_TOKEN=demo-token-2026 TRIAGE_STUDENT_API_URL=http://127.0.0.1:8080 \
TRIAGE_MODEL_URL=http://127.0.0.1:9000 TRIAGE_REQUESTS_PATH=data/requests.json \
TRIAGE_DB_PATH=output/triage.db TRIAGE_OUTPUT_DIR=output \
.venv/bin/python -m triage
```

---

## 2. What it does and where

```
config/config.yaml        categories, threshold, models, retries, reply templates
triage/
  config.py               loads the YAML and the environment variables
  schemas.py              strict schemas: request, student record, classification
  model_output.py         extracts and validates the model's output
  http.py                 retries, backoff, Retry-After, timeouts
  students.py             student API client, with a cache
  classifier.py           two-tier classification (cheap model, then precise)
  drafter.py              template drafts in the student's language
  store.py                SQLite and the state machine
  pipeline.py             the steps, one request at a time
  summary.py              summary, CSVs and cost
  __main__.py             entry point
mock_student_api/         mock student records API, with planted faults
mock_llm/                 mock OpenAI-compatible LLM, with planted faults
data/requests.json        43 incoming requests (English, Spanish and Catalan)
scripts/measure_cost.sh   measures the three cost strategies
```

Each request moves through `received → validated → enriched → classified → drafted →
completed`, with three other exits: `invalid`, `escalated` and `failed`. Each step
is committed to the database before the next one starts.

Result on the supplied file: 43 records, of which 3 are exact duplicates, leaving
**40 requests**. 33 are completed with a draft, 4 are escalated to a person, 3 are
invalid (one has no `student_id`, one has empty fields and one is for a student who
does not exist), and none fail.

---

## 3. Design decisions and why

**Idempotency is guaranteed by the database, not by the code.** `requests.id` and
`drafts.request_id` are primary keys, so re-running the file cannot duplicate a
request or write a second draft, even if the code had a bug. A hash of each record
is stored as well. The same id with the same content is a harmless duplicate, which
is ignored and counted. The same id with **different content** is a conflict: it is
never overwritten, and it goes to the human review queue.

**A crash is never paid for twice.** State is committed after every step. If the
process dies after classifying a request, the next run reuses the stored
classification instead of calling the model again. It decides this by checking
whether the classification is present in the row, not by looking at the state name,
because a `failed` request no longer records which step it reached, but whatever was
already paid for is still there.

**A failure stops one request, never the whole run, and never silently.** Once the
retries are used up, the request is marked `failed` with the reason and a row in
`incidents`, and processing moves on. It is not retried in the same run, which keeps
the run bounded in time; the next run picks it up. The one exception is the database
itself failing: carrying on when nothing can be recorded would make every later
failure silent, so the process stops there.

**My own HTTP client on `httpx`, not the `openai` SDK.** The SDK has its own retry
layer, which would sit between my code and the behaviour I most want to control:
how it reacts to a 429 or a 503. My client retries 429, 5xx, timeouts and connection
errors; honours `Retry-After` up to a 30-second cap; and never retries 400, 401 or
404. The timeout is 10 seconds per attempt, because some responses are deliberately
slow (3 to 4 seconds) and slow does not mean down.

**The student lookup is cached, including students who do not exist.** There are 40
requests from 15 students, so the pipeline makes 15 calls instead of 40. This is not
only cheaper: the mock API fails based on a call counter, so every call that is not
made is a failure that cannot happen.

**Model output is never corrected.** The first JSON object is extracted even when it
is wrapped in prose or code fences, and then validated against a strict schema.
`"urgency": "high"` is not turned into `3`, an invented category is not mapped to the
nearest real one, and `"confidence": "0.9"` as a string is rejected. If the cheap
model returns something invalid, the request is retried on the precise model. If
that also fails, the request is escalated with the reason and **the money already
spent is recorded**. Extra fields are ignored, because a valid classification is not
worth sending to a person just because the model added a comment.

**Two model tiers with a single threshold (0.75).** Every request is classified with
`sim-fast`. Only the ones that fall below the threshold are reclassified with
`sim-precise`, and if they are still below it, they are escalated. One threshold
drives both decisions, which means fewer settings to explain and no way to leave them
inconsistent. Section 4 shows why this is the right trade-off.

**Reply drafts come from templates, not from the model.** The mock model's free-text
answer is the same English text whatever the student's language, so it cannot
produce a reply in the student's language. The templates live in
`config/config.yaml` (English, Spanish and Catalan, for each of the four categories)
and use the real data from the student record: name, programme, enrolment status,
progress and tutor. A language with no template is escalated rather than improvised.

**No agent framework.** The flow is a fixed sequence of steps with no routing
decisions at runtime. A framework would add a layer between the code and the error
handling without providing anything that a plain function does not.

**SQLite as the source of truth, CSV for people.** A spreadsheet would be easier for
staff to open, but idempotency depends on a primary key and a transaction, which a
spreadsheet does not offer. So SQLite holds the data and the CSV exports are there
for people to read.

**Configuration lives outside the code.** Categories, threshold, models, retries,
waits and templates are in `config/config.yaml`. The token and paths are in `.env`,
which is not committed (`.env.example` is the template). Changing the threshold or a
reply text does not touch the logic.

---

## 4. Cost per 1,000 requests

Measured with [`scripts/measure_cost.sh`](scripts/measure_cost.sh) on the whole file,
starting each strategy from an empty database. Each run's cost is the **difference**
between two readings of `GET /v1/stats` (one before, one after), because the provider
keeps a running total from the moment it starts.

| Strategy | Model calls | Run cost | Cost per 1,000 | Need a person (at 0.75) |
|---|---|---|---|---|
| All `sim-fast` | 39 | €0.000875 | €0.022 | 13 |
| All `sim-precise` | 39 | €0.009813 | €0.245 | 4 |
| **Two tiers (chosen)** | 50 | €0.004086 | **€0.102** | **4** |

**The arithmetic:** run cost ÷ requests handled × 1,000.
For two tiers: €0.004086 ÷ 40 × 1,000 = **€0.1022**. That is 37 calls to `sim-fast`
and 13 calls to `sim-precise`. The program prints this same calculation at the end
of every run.

**What it means.** Two tiers costs 42% of always using the precise model and leaves
exactly the same human queue, because the expensive model is only paid for on the 13
requests where the cheap one is unsure. Using only the cheap model is 4.7 times
cheaper again, but `sim-fast` knocks 0.15 off every confidence it returns, so at the
same threshold it more than triples the human queue (13 against 4). At scale, that is
about 225 more manual reviews per 1,000 requests in order to save about €0.08. The
expensive resource is a person's time, not the model.

For the single-model strategies, the number who need a person is counted afterwards:
the requests whose stored confidence is below 0.75, plus the requests that no model
could classify. (Those runs use a threshold of 0, so nothing is escalated for low
confidence during the run itself.) The costs come from a simulator with canned
answers, so the ratio between the strategies is the meaningful part; the absolute
figures would depend on real text lengths.

---

## 5. The two tests that matter most

There are 190 tests. These are the two I would point to first.

**[`tests/test_model_output_validation.py`](tests/test_model_output_validation.py).**
This is the one place where an external system hands over garbage that looks like
valid data, and the failure is silent: nothing breaks, a corrupt classification is
stored and shown in the summary as if it were true. The tests prove that `"high"` is
not turned into `3`, that a category outside the list is not approximated, that a
confidence given as text or out of range is rejected, and that the JSON is found even
when it is wrapped in prose or contains braces inside strings.

**[`tests/test_idempotency.py`](tests/test_idempotency.py).** Idempotency is the
requirement that cannot be checked by eye. The test runs the full pipeline twice over
the same database and checks that the second run makes no model calls, adds no cost,
writes or rewrites no draft (the texts are byte-identical) and counts duplicates
correctly.

The rest of the suite exists because, during development, every guarantee was tested
against deliberate sabotage of the code: if breaking a guarantee made no test fail,
that test was not really testing anything. The aim is not coverage for its own sake,
but that every guarantee that matters has something that fails when it breaks.

---

## 6. Limitations

**Left out on purpose:**

- **No real email sending.** Drafts are stored; sending them would need SMTP
  credentials and a bounce policy, and would add nothing to the parts that matter
  here.
- **Template engine only.** `draft: engine: model` is rejected when a draft is
  generated, rather than half supported (see section 3).
- **No concurrency.** The student API limits by number of calls, so running requests
  in parallel would only produce more 429s. At higher volume, the right answer is a
  rate-limited queue.
- **Failed requests are not retried in the same run.** They are recorded, and the
  next run picks them up.
- **No user interface.**

**Known and not yet fixed:**

- If `sim-fast` is paid for and `sim-precise` then fails **on the network** (rather
  than returning invalid output), the first call's cost does not reach that request's
  cost column. The section 4 figures come from the provider, so they do not have this
  problem.
- If the database fails, the run stops (a deliberate decision, see section 3). A
  failure while creating the schema at start-up also falls outside the exit-code
  scheme.
- `http.timeout_s` has no lower limit. Below the 3 to 4 second slow responses,
  retried POSTs would repeat real model calls and multiply the cost.
- 3xx responses are treated as success. `Retry-After` is only understood in seconds,
  not as a date. When the 30-second cap applies, the log does not say what the server
  asked for. The exponential backoff base of 2 is fixed in the code.
- The CSVs do not neutralise cells that start with `=`, even though the reason text
  is written by the model and opened in Excel.
- The container also installs the test dependencies (there is a single
  `requirements.txt`). The optional `env_file` setting needs Docker Compose 2.24 or
  later.
- There is no test for a conflict against a request that never finished, nor for the
  alternative-date-format incident (the only such record in the file is rejected
  earlier for having empty fields).

---

## 7. Use of AI tools

I used **Claude Code** throughout, for design and planning, to implement each module
and to review it. The process was a task-by-task plan, where each task was
implemented by one agent and reviewed by a second, independent agent, which also
deliberately sabotaged the code to check that the tests caught it. Every decision
that came out of a review was made and recorded by me.

I have reviewed the code and can explain and defend every decision in this document.
