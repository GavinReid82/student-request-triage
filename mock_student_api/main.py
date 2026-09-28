# -*- coding: utf-8 -*-
"""
Mock student API.

A read-only service that returns one student's record.
Interactive docs at http://localhost:8080/docs

It behaves like an ordinary production service rather than a perfect one:
some calls are slow, some are rate-limited and some fail. The faults are
fixed by call number, so every run of the pipeline meets the same ones.
"""
import asyncio, itertools, threading
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock student API", version="1.0.0")

TOKEN = "demo-token-2026"
_counter = itertools.count(1)
_lock = threading.Lock()

# Call numbers where the service misbehaves on purpose.
RATE_LIMIT_ON = {12}
ERROR_500_ON = {27}
SLOW_ON = {7, 19, 33}

STUDENTS = {
 "STU-01402": dict(name="Nuria Beltrán Sáez", email="nuria.beltran@example.com",
   programme="Máster en IA Aplicada",
   enrolment_status="active", start_date="2026-01-15", progress_pct=61, tutor="Fernando Ruiz", language="es"),
 "STU-01755": dict(name="Óscar Prieto Lamas", email="oscar.prieto@example.com",
   programme="MSc in Automation Engineering",
   enrolment_status="active", start_date="2026-02-01", progress_pct=44, tutor="Fernando Ruiz", language="en"),
 "STU-02233": dict(name="Lucía Cambra Ortiz", email="lucia.cambra@example.com",
   programme="MSc in Generative AI",
   enrolment_status="completed", start_date="2025-09-10", progress_pct=100, tutor="Marta Salas", language="en"),
 "STU-02891": dict(name="Iván Rodero Gil", email="ivan.rodero@example.com",
   programme="MSc in Applied AI",
   enrolment_status="active", start_date="2026-01-15", progress_pct=38, tutor="Marta Salas", language="en"),
 "STU-03340": dict(name="Paula Ferrer Bosch", email="paula.ferrer@example.com",
   programme="MSc in Generative AI",
   enrolment_status="active", start_date="2026-03-02", progress_pct=22, tutor="Fernando Ruiz", language="en"),
 # This record arrives with the status field named differently from the rest.
 "STU-03871": dict(name="Sergio Valdés Mora", email="sergio.valdes@example.com",
   programme="MSc in Applied AI",
   status_enrolment="active", start_date="2026-02-01", progress_pct=55, tutor="Marta Salas", language="en"),
 "STU-04120": dict(name="Elena Quiroga Nieto", email="elena.quiroga@example.com",
   programme="MSc in Automation Engineering",
   enrolment_status="active", start_date="2026-01-15", progress_pct=71, tutor="Fernando Ruiz", language="en"),
 "STU-04412": dict(name="Andrés Lima Cordero", email="andres.lima@example.com",
   programme="MSc in Applied AI",
   enrolment_status="active", start_date="2026-02-01", progress_pct=49, tutor="Marta Salas", language="en"),
 "STU-04980": dict(name="Rocío Nájera Pons", email="rocio.najera@example.com",
   programme="MSc in Generative AI",
   enrolment_status="suspended", start_date="2025-11-03", progress_pct=17, tutor="Fernando Ruiz", language="en"),
 "STU-05120": dict(name="Tomás Herrera Vidal", email="tomas.herrera@example.com",
   programme="MSc in Applied AI",
   enrolment_status="active", start_date="2026-03-02", progress_pct=12, tutor="Marta Salas", language="en"),
 "STU-05677": dict(name="Alba Cifuentes Rey", email="alba.cifuentes@example.com",
   programme="MSc in Automation Engineering",
   enrolment_status="active", start_date="2026-02-01", progress_pct=66, tutor="Fernando Ruiz", language="en"),
 "STU-06001": dict(name="Diego Palau Mestre", email="diego.palau@example.com",
   programme="MSc in Generative AI",
   enrolment_status="active", start_date="2026-03-02", progress_pct=29, tutor="Marta Salas", language="en"),
 "STU-07001": dict(name="Sarah Whitfield", email="sarah.whitfield@example.com",
   programme="MSc in Generative AI",
   enrolment_status="active", start_date="2026-02-01", progress_pct=40, tutor="Marta Salas", language="en"),
 "STU-07002": dict(name="Jordi Sallent Roca", email="jordi.sallent@example.com",
   programme="Màster en IA Aplicada",
   enrolment_status="active", start_date="2026-02-01", progress_pct=33, tutor="Fernando Ruiz", language="ca"),
 "STU-07003": dict(name="Miriam Tejada Coll", email="miriam.tejada@example.com",
   programme="Máster en IA Generativa",
   enrolment_status="active", start_date="2026-01-15", progress_pct=58, tutor="Marta Salas", language="es"),
}

def _next():
    with _lock:
        return next(_counter)

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}

@app.get("/students/{student_id}", summary="One student's record")
async def student(student_id: str, authorization: str | None = Header(default=None)):
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, detail="Missing or wrong token. Use: Authorization: Bearer <token>")

    n = _next()

    if n in SLOW_ON:
        await asyncio.sleep(4)

    if n in RATE_LIMIT_ON:
        return JSONResponse(status_code=429,
            content={"error": "Too many requests", "retry_in_s": 2},
            headers={"Retry-After": "2"})

    if n in ERROR_500_ON:
        return JSONResponse(status_code=500,
            content={"error": "Internal service error"})

    record = STUDENTS.get(student_id)
    if record is None:
        raise HTTPException(404, detail=f"No student exists with id {student_id}")

    return {"student_id": student_id, **record}
