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
| `BASE44_AI_GATEWAY_URL` | The full HTTPS endpoint Base44 exposes (confirmed live: `https://gaswaterai.base44.app/functions/aiGateway`). |
| `BASE44_AI_GATEWAY_API_KEY` | The shared secret this engine presents **to** Base44, sent as the `x-gateway-secret` header (confirmed against Base44's live, tested endpoint). |

Both must be set (as platform secrets, never committed) for this engine to
prefer Base44's gateway over the direct Anthropic path; with neither set,
behavior is unchanged from before this change (model calls report
`available: false`, nothing fails).

## Request — POST `BASE44_AI_GATEWAY_URL`

Headers:

```
Content-Type: application/json
x-gateway-secret: <BASE44_AI_GATEWAY_API_KEY>
```

Body (every field always present; `images` may be an empty list):

```json
{
  "system_prompt": "string — the agent's system prompt, verbatim",
  "tool_name": "string — the name of the single tool the model must call",
  "tool_schema": {
    "type": "object",
    "properties": { "...": "one entry per expected output field" },
    "required": ["...the required output field names..."]
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
| `tool_schema` | object | A raw JSON Schema, root `"type": "object"` with `"properties"` (and `"required"` where applicable) -- live-confirmed Base44 requirement (HTTP 400: `"tool_schema" muss vom Typ "object" sein (JSON-Schema-Root)`), used as Base44's structured-output schema. Every agent still builds its tool definition in Anthropic's `{"name", "description", "input_schema"}` shape (needed unchanged for `AnthropicAgentModelProvider`'s `tools` parameter); `Base44AgentModelProvider` unwraps `input_schema` before sending (`_to_base44_json_schema` in `base44_provider.py`), never the agent or `AgentModelRequest` itself. |
| `text` | string | Plain text content, appended after any images in the user turn. |
| `images` | array of base64 strings | Each one PNG image. Order matters (if present, decode and pass as `image` content blocks in the same order, before the text block). |
| `model` | string \| null | A hint, not a requirement — Base44 may substitute its own default model if `null` or if it chooses to route differently. |
| `temperature` | number | Always `0.0` from every current agent (determinism) — pass through. |
| `max_tokens` | integer | Upper bound on the model's response size. |

## Image attachment handling inside aiGateway (live fix: "Invalid file attachment")

Live-confirmed symptom: after the `tool_schema` fix, model-dependent agents
that send a crop (SymbolAgent, AnschlussAgent, LeitungsAgent's vision
fallback, ...) get `available: false, error: "Invalid file attachment"` from
Base44's `InvokeLLM`. Root cause: `InvokeLLM` (Base44's
`integrations.Core.InvokeLLM`) does **not** accept inline image data of any
kind -- not raw bytes, not base64, not a `data:` URI. It accepts only
`file_urls`: an array of URLs pointing at files Base44 already has in its
own storage. This is confirmed, not assumed -- every existing, working
`InvokeLLM` call in this product's own Base44 app (`planpruefung`,
`visualFirstPoc`, `richtlinienChat`, `extractDocumentText`,
`verifyCorrection`) uses exactly this pattern and none of them ever passes
image/file bytes directly:

```js
const { signed_url } = await base44.asServiceRole.integrations.Core.CreateFileSignedUrl({
  file_uri, expires_in: 300,
});
const res = await base44.asServiceRole.integrations.Core.InvokeLLM({
  prompt, file_urls: [signed_url, ...], response_json_schema,
});
```

This engine's side of the contract does **not** change: `images` stays an
array of base64-encoded PNGs, no `data:` prefix, exactly as documented
above -- `Base44AgentModelProvider` and every agent stay exactly as they
are. **aiGateway** is the only place that needs to change: it must turn
each incoming base64 image into a short-lived, private file Base44's
`InvokeLLM` can reference by URL, before calling `InvokeLLM`. No agent, and
no part of this engine, needs to know this happens.

Required aiGateway-side steps per request that carries `images`:

1. For each base64 string in `body.images`, decode it to raw bytes and wrap
   it as a `File` (PNG, matching this engine's actual output --
   `new File([bytes], \`attachment-${i}.png\`, { type: "image/png" })`).
2. Upload each as a **private, temporary** file -- never a permanent public
   one, never written to any product entity (no `PlanCheck`, no
   `PlanInventory`, no customer/document entity -- this file exists solely
   to let `InvokeLLM` see the crop for this one call). Use whichever of
   Base44's own upload primitives yields a **private** reference rather
   than a public URL (the existing `feedbackPlanUrl`/`planAnalyseExtern`
   functions already treat `file_uri` as the private form, in contrast to
   the public `file_url` `UploadFile` can also return -- prefer that
   private path here, consistent with those).
3. Obtain a short-lived signed URL for each uploaded file via
   `CreateFileSignedUrl({ file_uri, expires_in: 300 })` -- the same 300s
   expiry every existing InvokeLLM-with-attachments call in this app
   already uses. No longer-lived or permanent link is needed; the URL only
   has to survive the single `InvokeLLM` call that follows immediately.
4. Pass the resulting signed URLs as `file_urls` to the single `InvokeLLM`
   call this request makes -- never more than one `InvokeLLM` call per
   gateway request, matching the existing no-retry-loop behavior.
5. Never log `body.images`, the decoded bytes, the uploaded file's
   contents, or the signed URL itself (a signed URL is a bearer credential
   for that file) -- log at most a count (`images.length`) and the HTTP
   outcome, the same discipline `Base44AgentModelProvider`'s own error
   logging already follows on this engine's side.
6. A request with an empty `images` array (every current text-only agent
   path, and the deterministic-only agents that never call a model at all)
   must skip all of the above and call `InvokeLLM` exactly as it does
   today, with no `file_urls` -- this fix only adds a step, it must not
   change behavior for a request with no images.

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

Concrete to Base44's own `integrations.Core` API (the shape every other
function in this app's own codebase already uses), including the
image-attachment fix above:

```js
export default async function (req) {
  const secret = req.headers.get('x-gateway-secret');
  if (!secret || secret !== Deno.env.get('GATEWAY_SHARED_SECRET')) {
    return Response.json({ error: 'unauthorized' }, { status: 401 });
  }
  const body = await req.json();

  let fileUrls = [];
  if (Array.isArray(body.images) && body.images.length > 0) {
    fileUrls = await Promise.all(body.images.map(async (b64, i) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const file = new File([bytes], `attachment-${i}.png`, { type: 'image/png' });
      // private upload -- never a permanent public file_url, never an entity write
      const { file_uri } = await base44.asServiceRole.integrations.Core.UploadFile({ file, private: true });
      const { signed_url } = await base44.asServiceRole.integrations.Core.CreateFileSignedUrl({
        file_uri, expires_in: 300,
      });
      return signed_url;
    }));
  }

  let result, error = null;
  try {
    result = await base44.asServiceRole.integrations.Core.InvokeLLM({
      prompt: body.system_prompt + '\n\n' + body.text,
      file_urls: fileUrls,
      response_json_schema: body.tool_schema,
    });
  } catch (e) {
    error = e.message || String(e);
  }

  if (error) {
    return Response.json({ available: false, tool_input: null, model: body.model, error }, { status: 200 });
  }
  return Response.json({ available: true, tool_input: result, model: body.model, raw_response_id: null });
}
```

Notes on this sketch: `UploadFile({ file, private: true })` stands in for
whichever of Base44's own upload primitives yields a private file
reference rather than a permanent public `file_url` -- the exact call
shape needs confirming against Base44's current `integrations.Core` API
surface (not independently verified in this epic, since aiGateway's actual
source is not in any repository this engine's own development has access
to). The `response_json_schema: body.tool_schema` line relies on the
`tool_schema` fix above (a bare JSON-Schema-root object, not the Anthropic
tool-definition wrapper).
