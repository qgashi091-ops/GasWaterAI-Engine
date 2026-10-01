# Base44 AI Gateway — Request/Response Contract

This is the exact contract `app/multi_agent_v1/base44_provider.py`'s
`Base44AgentModelProvider` requires from a Base44-operated endpoint, so that
the multi-agent pipeline's model-dependent agents (SymbolAgent, TextAgent,
AnschlussAgent, and the model-fallback paths of PlanstrukturAgent,
LeitungsAgent, SicherungsAgent, ZirkulationsHydraulikAgent, ProbenahmeAgent)
can run **without requiring a separately-paid Anthropic API key for product
operation**.

No agent, prompt, or tool schema changed to make this possible. Every agent
already only talks to the `AgentModelProvider` abstraction
(`app/multi_agent_v1/provider.py`) via one method —
`call(AgentModelRequest) -> AgentModelResponse` — so this contract is simply
that same request/response shape carried over HTTP+JSON, field for field.
`Base44AgentModelProvider` is a drop-in alternate to the existing
`AnthropicAgentModelProvider`, selected by configuration
(`app/main.py::_select_agent_model_provider`), never by agent code.

## Configuration (this engine's side)

| Environment variable | Meaning |
|---|---|
| `BASE44_AI_GATEWAY_URL` | The full HTTPS endpoint Base44 exposes. This engine makes no assumption about its path — Base44 decides the URL. |
| `BASE44_AI_GATEWAY_API_KEY` | The bearer credential this engine presents **to** Base44, sent as `Authorization: Bearer <key>`. |

Both must be set (as platform secrets, never committed) for this engine to
prefer Base44's gateway over the direct Anthropic path; with neither set,
behavior is unchanged from before this change (model calls report
`available: false`, nothing fails).

## Request — POST `BASE44_AI_GATEWAY_URL`

Headers:

```
Content-Type: application/json
Authorization: Bearer <BASE44_AI_GATEWAY_API_KEY>
```

Body (every field always present; `images` may be an empty list):

```json
{
  "system_prompt": "string — the agent's system prompt, verbatim",
  "tool_name": "string — the name of the single tool the model must call",
  "tool_schema": {
    "name": "string, equal to tool_name",
    "description": "string",
    "input_schema": { "...": "a JSON Schema object (Anthropic tool-schema shape)" }
  },
  "text": "string — the agent's user-turn text content",
  "images": ["string, ... — each a base64-encoded PNG, no data: URI prefix"],
  "model": "string or null — the agent's requested model id, or null to let Base44 choose a default",
  "temperature": 0.0,
  "max_tokens": 300
}
```

Field meanings map 1:1 to `AgentModelRequest` (`provider.py`):

| Field | Type | Notes |
|---|---|---|
| `system_prompt` | string | Never modify — these prompts are the product's reviewed behavior. |
| `tool_name` | string | The gateway must force the underlying model to call exactly this tool (e.g. Anthropic's `tool_choice: {"type": "tool", "name": tool_name}`), not answer in free text. |
| `tool_schema` | object | A single Anthropic-style tool definition. Pass through unmodified to whatever model Base44's gateway calls. |
| `text` | string | Plain text content, appended after any images in the user turn. |
| `images` | array of base64 strings | Each one PNG image. Order matters (if present, decode and pass as `image` content blocks in the same order, before the text block). |
| `model` | string \| null | A hint, not a requirement — Base44 may substitute its own default model if `null` or if it chooses to route differently. |
| `temperature` | number | Always `0.0` from every current agent (determinism) — pass through. |
| `max_tokens` | integer | Upper bound on the model's response size. |

## Response — 200 OK, JSON body

```json
{
  "available": true,
  "tool_input": { "...": "the parsed arguments the model's tool call produced" },
  "model": "string — the model actually used",
  "raw_response_id": "string or null — an opaque id for audit, never the raw prompt/response text",
  "error": null
}
```

Field meanings map 1:1 to `AgentModelResponse` (`provider.py`):

| Field | Required | Meaning |
|---|---|---|
| `available` | yes | `true` if the model call succeeded and produced a usable tool call; `false` for any failure (quota, refusal, timeout, parse failure on Base44's side). |
| `tool_input` | yes (may be `null`) | The tool call's parsed arguments as a JSON object, exactly as the model produced them — **never** reformatted, renamed, or filled in by Base44's gateway. `null` when `available` is `false`. |
| `model` | yes | The model id actually used to serve the request. |
| `raw_response_id` | no (may be omitted or `null`) | An opaque identifier for audit/debugging. Must never carry prompt or response text (these agents' crops/plan text can contain real, potentially identifying content — see `app/vision_fallback/interface.py`'s own privacy note, which applies here too). |
| `error` | no (may be omitted or `null`) | A short, human-readable reason when `available` is `false`. |

### On failure

Return **HTTP 200** with `"available": false` and a populated `"error"`
whenever possible — this lets `Base44AgentModelProvider` record a clean,
typed `AgentModelResponse` instead of treating it as a transport failure.
A non-2xx HTTP status or a connection failure is also handled (mapped to
`available: false` with a generic `"Base44 AI Gateway request failed: ..."`
message), but carries less diagnostic detail than a 200 with an explicit
`error` field.

### Malformed responses

If the response body is not valid JSON, or is missing the `available` key,
`Base44AgentModelProvider` returns `available: false` with a parse-error
message — it never raises an exception into the calling agent (see
`tests/multi_agent_v1/test_base44_provider.py`).

## What never changes on Base44's side

- No agent's `system_prompt` or `tool_schema` may be edited, summarized, or
  "optimized" by the gateway — forward them to the model exactly as
  received. This epic's instruction was explicitly "keine Promptoptimierung".
- The gateway must not invent a `tool_input` value when the underlying model
  call fails or refuses — return `available: false` instead. Every agent in
  this codebase already treats an unavailable call as an honest
  "unresolved", never a fabricated answer (see `evidence_merger.py`'s
  `DETERMINISTIC_FACT > AGENT_OBSERVATION` priority rule, which depends on
  agents never bluffing when they have nothing real to say).
- `images` must be passed through as genuine image content to the model
  Base44 calls (not OCR'd, not described in text, not dropped) — several
  agents (SymbolAgent, AnschlussAgent, SicherungsAgent's position/Zuordnung
  path, ...) depend on the model actually seeing the crop.

## Example: a minimal reference implementation sketch (Base44 side, pseudocode)

```text
POST /ai-gateway
  verify Authorization: Bearer === configured secret, else 401
  body = parse JSON
  result = call_underlying_model(
      system=body.system_prompt,
      tools=[body.tool_schema],
      tool_choice=body.tool_name,
      temperature=body.temperature,
      max_tokens=body.max_tokens,
      content=[*decode_images(body.images), {type: text, text: body.text}],
  )
  if result.ok:
      return 200 { available: true, tool_input: result.tool_call.input,
                   model: result.model_used, raw_response_id: result.id }
  else:
      return 200 { available: false, tool_input: null,
                   model: body.model, error: result.failure_reason }
```
