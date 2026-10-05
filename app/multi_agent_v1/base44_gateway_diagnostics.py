"""TEMPORARY diagnostic probe for the live Render -> Base44 AI Gateway 403.

Not part of the agent pipeline: calls no agent, runs no plan check, invokes
no AgentModelProvider. It is a raw, standalone HTTP probe that sends the
SAME minimal text-only request shape the real Base44 AI Gateway contract
defines (docs/base44-ai-gateway-contract.md), from wherever this code
actually runs -- i.e. from inside the Render service itself when called via
the `/diagnostics/base44-gateway` endpoint (app/main.py), which is the only
way to reproduce an IP-/edge-specific block (Render's own egress network).

Sends AT MOST TWO requests total, no retry loop:
  1. Always: default urllib User-Agent (matches exactly what
     Base44AgentModelProvider sends today -- see base44_provider.py).
  2. ONLY if request 1 returned exactly HTTP 403: the same request again
     with a browser-like User-Agent, for comparison.

Never returns BASE44_AI_GATEWAY_API_KEY, GASWATERAI_MULTI_AGENT_API_KEY, any
Authorization-style header value, or cookies -- only the safe diagnostic
fields this epic asked for. A defensive redaction pass also scrubs the
gateway secret from any captured header/body text as a second safety net,
in case it is ever echoed back.

DELETE THIS MODULE (and its route in app/main.py) once the Base44 403 is
resolved -- it exists solely to diagnose that one live incident.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from .base44_provider import GATEWAY_API_KEY_ENV_VAR, GATEWAY_URL_ENV_VAR

REQUEST_TIMEOUT_SECONDS = 30

# Headers safe to return -- deliberately excludes Authorization, Set-Cookie,
# Cookie, and anything else that could carry a credential.
SAFE_RESPONSE_HEADERS = [
    "content-type", "server", "via", "cf-ray", "cf-cache-status", "x-render-origin-server",
    "x-powered-by", "location", "x-request-id", "date", "content-length",
    "x-frame-options", "strict-transport-security", "x-cache", "x-amz-cf-id",
]

MINIMAL_BODY = {
    "system_prompt": "You are a connectivity diagnostic. Respond only via the ping tool.",
    "tool_name": "diagnostic_ping",
    "tool_schema": {
        "name": "diagnostic_ping",
        "description": "Diagnostic no-op tool.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "text": "ping",
    "images": [],
    "model": None,
    "temperature": 0.0,
    "max_tokens": 16,
}

BROWSER_LIKE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text


def _classify(content_type: str, body_text: str) -> str:
    ct = (content_type or "").lower()
    stripped = body_text.strip()[:200].lower()
    if "application/json" in ct:
        try:
            json.loads(body_text)
            return "BASE44_JSON"
        except (json.JSONDecodeError, ValueError):
            return "OTHER"
    if "text/html" in ct or stripped.startswith("<!doctype") or stripped.startswith("<html"):
        return "HTML_EDGE_WAF"
    return "OTHER"


def _send_one(url: str, secret: str, user_agent: str | None) -> dict:
    body_bytes = json.dumps(MINIMAL_BODY).encode("utf-8")
    headers = {"content-type": "application/json", "x-gateway-secret": secret}
    if user_agent is not None:
        headers["user-agent"] = user_agent
    sent_user_agent = user_agent or "Python-urllib/{}.{}".format(*sys.version_info[:2])

    req = urllib.request.Request(url, data=body_bytes, method="POST", headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            status = resp.status
            final_url = resp.geturl()
            resp_headers = dict(resp.headers.items())
            raw_body = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        final_url = exc.geturl() if hasattr(exc, "geturl") else url
        resp_headers = dict(exc.headers.items()) if exc.headers else {}
        raw_body = exc.read() if hasattr(exc, "read") else b""
    except urllib.error.URLError as exc:
        return {
            "status": None, "content_type": None, "headers": {}, "body": None,
            "final_url": url, "redirected": False, "user_agent": sent_user_agent,
            "classification": "OTHER", "transport_error": str(exc),
        }

    body_text = _redact(raw_body.decode("utf-8", errors="replace"), secret)[:2000]
    content_type = resp_headers.get("Content-Type") or resp_headers.get("content-type") or ""
    safe_headers = {}
    for wanted in SAFE_RESPONSE_HEADERS:
        for actual_key in resp_headers:
            if actual_key.lower() == wanted:
                safe_headers[actual_key] = _redact(resp_headers[actual_key], secret)

    return {
        "status": status,
        "content_type": content_type,
        "headers": safe_headers,
        "body": body_text,
        "final_url": final_url,
        "redirected": final_url != url,
        "user_agent": sent_user_agent,
        "classification": _classify(content_type, body_text),
        "transport_error": None,
    }


def run_diagnostics(url: str | None = None, secret: str | None = None) -> dict:
    """Runs request 1 always, request 2 only if request 1's status is
    exactly 403. Returns a dict with no secret values anywhere in it."""
    url = url or os.environ.get(GATEWAY_URL_ENV_VAR)
    secret = secret or os.environ.get(GATEWAY_API_KEY_ENV_VAR)

    if not url or not secret:
        return {
            "configured": False,
            "error": f"{GATEWAY_URL_ENV_VAR} / {GATEWAY_API_KEY_ENV_VAR} not set -- no request sent.",
            "request_1": None, "request_2": None, "request_2_skipped_reason": "not configured",
        }

    request_1 = _send_one(url, secret, user_agent=None)

    request_2 = None
    skipped_reason = None
    if request_1["status"] == 403:
        request_2 = _send_one(url, secret, user_agent=BROWSER_LIKE_USER_AGENT)
    else:
        skipped_reason = f"request 1 status was {request_1['status']!r}, not 403"

    return {
        "configured": True,
        "request_1": request_1,
        "request_2": request_2,
        "request_2_skipped_reason": skipped_reason,
    }
