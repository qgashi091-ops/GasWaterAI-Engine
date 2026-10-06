# Multi-Agent v1 — Asynchronous Job Mode: Base44 Integration Contract

Written for the job-mode epic: Base44's first live call to
`POST /multi_agent_v1/analyze` succeeded (HTTP 200) but took ≈340.7s,
while Base44 itself gives up waiting after ≈120s ("Multi-Agent v1 nicht
verfügbar / Engine nicht rechtzeitig geantwortet"). A multi-minute plan
analysis can no longer run as a single synchronous HTTP request from
Base44 — it must be submitted as a job and polled. This document is the
one place Base44's integration code should be written against.

`POST /multi_agent_v1/analyze` (synchronous) still exists, unchanged, for
diagnosis/development only (section 3 of the job-mode spec) — Base44's
production integration must use the job pair below, never the
synchronous endpoint.

## 1. Submit the job

```
POST /multi_agent_v1/jobs
X-API-Key: <same key as /multi_agent_v1/analyze, if GASWATERAI_MULTI_AGENT_API_KEY is configured>
Content-Type: multipart/form-data
  file: <the PDF, identical upload shape to /multi_agent_v1/analyze>
```

Returns immediately (does not wait for the analysis):

```
202 Accepted
{"job_id": "<opaque id>", "status": "queued"}
```

`status` in this response reflects the job's REAL current state at the
moment of the call — usually `"queued"`, but see idempotency below: if an
identical plan is already being analyzed, this call returns that job's
current status (which may already be `"processing"`) together with its
existing `job_id`, never a second job.

Error responses from this endpoint itself (never queued, no job_id
returned):
  - `400` — the uploaded bytes are not a PDF.
  - `401` — missing/invalid `X-API-Key` (only once
    `GASWATERAI_MULTI_AGENT_API_KEY` is configured on the Engine).
  - `413` — the upload exceeds the job endpoint's size limit
    (`GASWATERAI_JOB_MAX_UPLOAD_BYTES`, default 25 MB — see section 6 of
    this contract).
  - `503` — the job queue is at capacity
    (`GASWATERAI_JOB_MAX_QUEUE_LENGTH`, default 20); retry later.

## 2. Store `job_id`

Base44 must persist `job_id` against the plan/check it belongs to (same
place it already tracks a check's own identity) so it can resume polling
even if the user navigates away and comes back, or the Base44 process
itself restarts.

## 3. Show "Plan wird analysiert"

The UI shows a durable "plan is being analyzed" state immediately after
the 202 — this is expected to take minutes, not seconds, exactly like the
original live run (≈340.7s) that motivated this epic.

## 4. Poll `GET /multi_agent_v1/jobs/{job_id}`

```
GET /multi_agent_v1/jobs/{job_id}
X-API-Key: <same key>
```

Always 200 once `job_id` exists, with exactly one of four `status`
values. The response always carries all of these keys (nulled where not
applicable, so Base44 can parse one stable shape regardless of status):

```json
{
  "job_id": "...",
  "status": "queued" | "processing" | "completed" | "failed",
  "created_at": "<ISO-8601 UTC>",
  "started_at": "<ISO-8601 UTC or null>",
  "completed_at": "<ISO-8601 UTC or null>",
  "result": <the same body POST /multi_agent_v1/analyze returns, or null>,
  "error": "<technical message, or null>"
}
```

  - `queued` / `processing`: `result` and `error` are both `null`;
    `started_at` is `null` while still `queued`, populated once a worker
    actually begins processing.
  - `completed`: `result` holds the full analysis body — identical in
    shape and content to what `POST /multi_agent_v1/analyze` would have
    returned for the same input (`engine_version`, `document_fingerprint`,
    `canonical_plan_understanding`, `observations`, `diagnostics`).
    `error` is `null`.
  - `failed`: `result` is `null`. `error` is a technical message only —
    never plan content, never a secret, never a stack trace.

`GET /multi_agent_v1/jobs/{unknown-id}` returns `404`.

## 5. Keep waiting while `processing`

Recommended polling interval: **5–10 seconds**. This is frequent enough
that the UI feels responsive once the job finishes, and far too
infrequent to meaningfully load a single Render Free/Starter instance
(the Engine's own GET handler does only a cheap in-memory lookup — no PDF
re-parsing, no model call). Do not poll faster than every 5 seconds in
production.

## 6. Show the result when `completed`

Render `result.canonical_plan_understanding` (and the rest of `result`)
exactly as Base44 already renders `POST /multi_agent_v1/analyze`'s
response today — no new parsing logic needed, since it is the same body.

## 7. Show a technical error when `failed`

Surface `error` as a technical/diagnostic message (e.g. in a collapsed
"details" section), not as the primary user-facing failure reason —
it intentionally never contains plan content or secrets, so it is safe to
display, but it is not written to be customer-friendly prose.

## 8. NEVER start a second model run just because of a UI timeout

This is the whole point of this epic: if Base44's own UI/request layer
times out while polling (e.g. one `GET` call itself takes too long over
the network), **do not** fall back to calling
`POST /multi_agent_v1/jobs` again for the same plan "just in case" —
that pays for a second, redundant multi-minute analysis. The correct
behavior on any transient polling failure is to retry the *same*
`GET /multi_agent_v1/jobs/{job_id}` call after the normal 5–10s interval;
the job keeps running on the Engine regardless of whether Base44's poll
succeeded. The Engine's own idempotency check (document_fingerprint, see
below) is a safety net for accidental double-clicks on a NEW submission,
not a substitute for correct polling behavior on an already-known
`job_id`.

## Idempotency (informational — Base44 does not need to implement this)

The Engine itself de-duplicates `POST /multi_agent_v1/jobs` calls for the
identical PDF (SHA-256 `document_fingerprint`) while a job for it is
still `queued`/`processing`: a repeated submission (e.g. a user
double-clicking "Prüfen") returns the SAME `job_id` instead of starting a
second, costly analysis. This is a backstop against accidental duplicate
submissions, not a reason to skip storing `job_id` client-side.

## Known POC limits Base44 should be aware of

  - Job state lives in the Engine's own process memory
    (`InMemoryJobStore` — see `app/multi_agent_v1/job_store.py`'s module
    docstring for the full boundary). A Render restart/redeploy of the
    Engine loses all in-flight and recently-completed job records; a
    `job_id` from before such a restart will come back `404`. Base44
    should treat a `404` for a `job_id` it previously saw as "that job no
    longer exists, resubmit if the user still wants this plan analyzed" —
    never as a transient error worth blindly retrying forever.
  - Job records are cleaned up automatically after
    `GASWATERAI_JOB_TTL_SECONDS` (default 1 hour) past completion/failure.
    Poll promptly once the UI is actively showing a job's progress;
    don't rely on a `job_id` still being resolvable hours later.
