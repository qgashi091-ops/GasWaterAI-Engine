# Deployment — container, access control, Base44 model-call adapter

This covers the deployment PREREQUISITES added to the repository (Dockerfile,
`.dockerignore`, version pin, shared-secret protection, Base44 provider
adapter). It does **not** cover actually deploying anywhere — no hosting
account was created, nothing was ordered, nothing runs outside this repo.

## Container

- **Dockerfile**: `python:3.11-slim` base (not an exact patch pin — see
  below), installs `tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-deu`
  via `apt-get` (required by `pytesseract`'s OCR fallback in
  `app/plan_analysis/text.py`; it is a system binary, never a pip package,
  and is absent from a plain Python base image). Copies `app/`,
  `requirements.txt`, and the two files the running service actually reads
  from under `data/` (`data/dev_plans_v04/detector_v1/{model.joblib,
  training_config.json}` — confirmed by a repo-wide search to be the only
  runtime data dependency; everything else under `data/` is dev-only).
  Binds to `$PORT` when the host provides one (e.g. Render), else `8000`.

- **`.python-version`**: pinned to `3.11.15` (this repository's actual
  development version) for local tooling (pyenv, etc.). The Docker image
  intentionally floats on the `3.11` minor line rather than pinning the
  same exact patch, so security patches to the base image keep arriving
  without a manual bump; re-pin to an exact `python:3.11.x-slim` tag instead
  if byte-identical container builds become a requirement.

- **`.dockerignore`**: explicit blocklist of the known dev-only
  subdirectories under `data/dev_plans_v04/` (annotation datasets, benchmark
  crops, microagent POC output, the symbol-reidentification experiment,
  etc. — ~196 MB total), plus `tests/`, `docs/`, `scripts/`, `.git/`, and
  `.env*`. `data/dev_plans_v04/detector_v1/` is deliberately NOT listed (the
  Dockerfile's `COPY` needs it in the build context) — a blocklist was used
  instead of an allowlist-with-negation specifically to avoid Docker's
  well-known negation-pattern gotchas (re-including a path whose parent
  directory is excluded silently fails); verified empirically by building
  the image and inspecting its contents (see the build log this epic's
  final report links).

## `/multi_agent_v1/analyze` access control

`app/main.py::_verify_multi_agent_api_key` checks an `X-API-Key` header
against the `GASWATERAI_MULTI_AGENT_API_KEY` environment variable, using
`hmac.compare_digest` (constant-time comparison). **Fails open when the
variable is unset** — today's default, and every test's environment — so
this change does not lock anyone out until the variable is actually set on
a host. `/analyze` and `/check` carry no such dependency at all and are
completely unaffected (see `tests/test_multi_agent_v1_api.py`).

To activate: set `GASWATERAI_MULTI_AGENT_API_KEY` to a random secret on the
hosting platform's environment-variable store (never in git), and have
Base44 send it as `X-API-Key` on every call to `/multi_agent_v1/analyze`.

## Model calls without a paid Anthropic key

`app/multi_agent_v1/provider.py`'s `AgentModelProvider` abstraction was
already generic (confirmed, not redesigned) — every one of the 12 agents
only ever calls `provider.call(AgentModelRequest) -> AgentModelResponse`, so
a second, alternate provider can be swapped in purely via configuration,
with zero agent/prompt/schema changes.

`app/multi_agent_v1/base44_provider.py`'s `Base44AgentModelProvider`
implements that same interface by calling a Base44-operated HTTP endpoint
instead of Anthropic directly. `app/main.py::_select_agent_model_provider`
prefers it whenever `BASE44_AI_GATEWAY_URL` is configured, falling back to
the existing `AnthropicAgentModelProvider` otherwise (local dev, or before
Base44's gateway exists) — with neither configured, behavior is byte-for-
byte what it was before this change.

See `docs/base44-ai-gateway-contract.md` for the exact JSON request/response
shape Base44's gateway must implement.

## What was NOT changed

- No agent logic, system prompt, tool schema, or claim-type contract.
- No change to `/analyze` or `/check` request/response shape or behavior.
- No change to PlanFacts, the rule engine, or the golden-holdout guard.
- No deployment performed; no hosting account created; no cost incurred.
