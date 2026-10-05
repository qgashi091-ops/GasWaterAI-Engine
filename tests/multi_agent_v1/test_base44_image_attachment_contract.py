"""Live follow-up: after the tool_schema fix, model-dependent agents that
attach a crop get `available: false, error: "Invalid file attachment"` from
Base44's InvokeLLM. Root cause (confirmed against every real, working
InvokeLLM call already in the connected gaswaterai-base44-app repo --
planpruefung, visualFirstPoc, richtlinienChat, extractDocumentText,
verifyCorrection): InvokeLLM accepts only `file_urls`, never inline image
bytes/base64/data-URLs -- so the fix belongs entirely inside Base44's own
`aiGateway` function (not present in any repository this engine's
development has access to; see docs/base44-ai-gateway-contract.md's new
"Image attachment handling inside aiGateway" section for the exact,
documented fix).

This engine's own side of the contract is UNCHANGED -- `images` stays a
list of base64-encoded PNGs, no data: prefix (base44_provider.py was never
touched for this epic). These tests lock in exactly the properties this
epic had to verify stayed true on the engine side: text-only requests are
unaffected, images are transported format-agnostically (so a future JPEG
would work exactly like today's PNGs), a Base44-side "Invalid file
attachment" failure surfaces cleanly with no image data ever leaking into
`AgentModelResponse.error`, the shared secret and tool_schema fixes from
the previous two epics are untouched, and exactly one HTTP request (hence
at most one InvokeLLM call on Base44's side) is made per `.call()`."""
from __future__ import annotations

import base64
import io
import json
import urllib.error
from unittest import mock

from app.multi_agent_v1.base44_provider import Base44AgentModelProvider
from app.multi_agent_v1.provider import AgentModelRequest

_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload"
_FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"fake-jpeg-payload"


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _ok_payload(tool_input=None):
    return json.dumps({"available": True, "tool_input": tool_input or {}, "model": "m"}).encode("utf-8")


def test_text_only_request_is_unaffected_and_sends_an_empty_images_list():
    """Requirement 1: no images attached -> `images` is sent as an empty
    list, exactly as before this epic -- the image-attachment fix is
    entirely Base44-side and must change nothing for a text-only call."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
        text="hello", images=[], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_ok_payload())

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        response = provider.call(request)

    assert response.available is True
    assert captured["body"]["images"] == []


def test_a_png_image_is_base64_encoded_with_no_data_uri_prefix():
    """Requirement 2: a real PNG (magic bytes included) is prepared exactly
    per the documented contract -- base64, no `data:` prefix, order
    preserved."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
        text="hello", images=[_FAKE_PNG], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_ok_payload())

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(request)

    sent_images = captured["body"]["images"]
    assert sent_images == [base64.b64encode(_FAKE_PNG).decode("ascii")]
    assert not sent_images[0].startswith("data:")
    assert base64.b64decode(sent_images[0]) == _FAKE_PNG


def test_image_transport_is_format_agnostic_a_jpeg_would_work_identically():
    """Requirement 3: no agent currently produces JPEG (confirmed: zero
    `jpeg`/`jpg` references anywhere under app/multi_agent_v1/ or
    app/vision_fallback/ -- every crop comes from
    PlanAgentContext.render_crop_png()'s `pixmap.tobytes("png")`). The
    provider itself never inspects or assumes image format -- it only
    base64-encodes whatever bytes `AgentModelRequest.images` carries -- so
    this proves a future JPEG-producing agent would already be transported
    correctly with no provider change needed."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
        text="hello", images=[_FAKE_JPEG], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_ok_payload())

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(request)

    assert captured["body"]["images"] == [base64.b64encode(_FAKE_JPEG).decode("ascii")]


def test_invalid_file_attachment_error_surfaces_cleanly_without_image_data():
    """Requirements 4+5: the live-confirmed Base44 failure
    ("Invalid file attachment") must surface through
    `AgentModelResponse.error` without crashing, and even if Base44's own
    error body echoes back part of the rejected attachment, no base64/
    image content ever reaches the error text (the existing
    `_safe_http_error_detail` redaction from the previous epic already
    covers this -- this pins it to the exact, now-confirmed live error)."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
        text="hello", images=[_FAKE_PNG], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    leaked_b64_fragment = base64.b64encode(_FAKE_PNG * 10).decode("ascii")  # long enough to trip the base64 scrub
    error_body = json.dumps({
        "available": False,
        "error": f"Invalid file attachment: could not process {leaked_b64_fragment}",
    }).encode("utf-8")

    def _raise(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://base44.example/ai-gateway", 400, "Bad Request",
            {"Content-Type": "application/json"}, io.BytesIO(error_body),
        )

    with mock.patch("urllib.request.urlopen", side_effect=_raise):
        response = provider.call(request)

    assert response.available is False
    assert "Invalid file attachment" in response.error
    assert leaked_b64_fragment not in response.error
    assert "[IMAGE_DATA_REDACTED]" in response.error


def test_shared_secret_header_is_unchanged_by_this_epic():
    """Requirement 6: x-gateway-secret is still sent exactly as before --
    this epic touched no auth logic."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
        text="hello", images=[_FAKE_PNG], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="unchanged-secret")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return _FakeResp(_ok_payload())

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(request)

    assert captured["headers"]["X-gateway-secret"] == "unchanged-secret"


def test_tool_schema_unwrap_is_unchanged_by_this_epic():
    """Requirement 7: the tool_schema fix from the previous epic
    (`_to_base44_json_schema`) is untouched -- an Anthropic-wrapped schema
    still arrives at Base44 unwrapped to its bare JSON-Schema root."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="classify_medium",
        tool_schema={
            "name": "classify_medium", "description": "d",
            "input_schema": {"type": "object", "properties": {"medium": {"type": "string"}}, "required": ["medium"]},
        },
        text="hello", images=[_FAKE_PNG], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(_ok_payload())

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(request)

    assert captured["body"]["tool_schema"] == {
        "type": "object", "properties": {"medium": {"type": "string"}}, "required": ["medium"],
    }


def test_call_makes_exactly_one_http_request_no_retry_loop():
    """Requirement 8: one `.call()` makes exactly one HTTP request to the
    gateway URL -- the engine-side half of "a gateway request produces at
    most one InvokeLLM call" (the other half -- aiGateway itself never
    retrying InvokeLLM -- is documented as a requirement in
    docs/base44-ai-gateway-contract.md but outside this engine's own code,
    since aiGateway's source isn't in any repository this engine's
    development has access to)."""
    request = AgentModelRequest(
        system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
        text="hello", images=[_FAKE_PNG], model=None, temperature=0.0, max_tokens=300,
    )
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    call_count = {"n": 0}

    def _fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        return _FakeResp(_ok_payload())

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(request)

    assert call_count["n"] == 1
