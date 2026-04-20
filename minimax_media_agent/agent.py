"""
agents/minimax_media_agent/agent.py — MiniMax media generation embedded agent.

Drop-in embedded agent for Backplaned.  Accepts an LLMData prompt plus
optional ProxyFile references, runs an LLM tool loop backed by MiniMax's
Anthropic-compatible endpoint, and invokes MiniMax media-generation APIs
(image, and in future: speech, video, music) to produce media files that
are returned as ProxyFile attachments.

No additional dependencies — uses only httpx + fastapi, both already
required by Backplaned itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from helper import (
    AgentInfo,
    AgentOutput,
    LLMData,
    ProxyFile,
    ProxyFileManager,
    build_result_request,
)

from tools import (
    MINIMAX_TOOLS,
    ToolContext,
    execute_tool,
)

logger = logging.getLogger("minimax_media_agent")


# ---------------------------------------------------------------------------
# Configuration (re-read on every request for hot-reload)
# ---------------------------------------------------------------------------

_CONFIG_PATH = _AGENT_DIR / "data" / "config.json"
_OUR_AGENT_ID = "minimax_media_agent"


def _load_config() -> dict[str, Any]:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# Module-level defaults — refreshed from config.json on every request.
MINIMAX_API_KEY: str = ""
MINIMAX_API_BASE: str = "https://api.minimax.io"
LLM_MODEL: str = "MiniMax-M2.7"
LLM_MAX_TOKENS: int = 4096
LLM_TEMPERATURE: float = 1.0
IMAGE_MODEL: str = "image-01"
OUTPUT_DIR: str = str(_AGENT_DIR / "data" / "output")
AGENT_TIMEOUT: float = 180.0
HTTP_TIMEOUT: float = 120.0
MAX_ITERATIONS: int = 6
MAX_TOOL_CALLS: int = 8
ROUTER_URL: str = os.environ.get("ROUTER_URL", "http://localhost:8000")


def _refresh_config() -> None:
    global MINIMAX_API_KEY, MINIMAX_API_BASE, LLM_MODEL, LLM_MAX_TOKENS
    global LLM_TEMPERATURE, IMAGE_MODEL, OUTPUT_DIR
    global AGENT_TIMEOUT, HTTP_TIMEOUT, MAX_ITERATIONS, MAX_TOOL_CALLS
    cfg = _load_config()
    _s = lambda v, d: d if v is None or v == "" else str(v)
    _si = lambda v, d: d if v is None or v == "" else int(v)
    _sf = lambda v, d: d if v is None or v == "" else float(v)
    MINIMAX_API_KEY = _s(cfg.get("MINIMAX_API_KEY"), "")
    MINIMAX_API_BASE = _s(cfg.get("MINIMAX_API_BASE"), "https://api.minimax.io").rstrip("/")
    LLM_MODEL = _s(cfg.get("LLM_MODEL"), "MiniMax-M2.7")
    LLM_MAX_TOKENS = _si(cfg.get("LLM_MAX_TOKENS"), 4096)
    LLM_TEMPERATURE = _sf(cfg.get("LLM_TEMPERATURE"), 1.0)
    IMAGE_MODEL = _s(cfg.get("IMAGE_MODEL"), "image-01")
    OUTPUT_DIR = _s(cfg.get("OUTPUT_DIR"), str(_AGENT_DIR / "data" / "output"))
    AGENT_TIMEOUT = _sf(cfg.get("AGENT_TIMEOUT"), 180.0)
    HTTP_TIMEOUT = _sf(cfg.get("HTTP_TIMEOUT"), 120.0)
    MAX_ITERATIONS = _si(cfg.get("MAX_ITERATIONS"), 6)
    MAX_TOOL_CALLS = _si(cfg.get("MAX_TOOL_CALLS"), 8)


_refresh_config()


# ---------------------------------------------------------------------------
# Output-directory GC (throttled)
# ---------------------------------------------------------------------------

_OUTPUT_MAX_AGE_SECS: int = 6 * 3600  # 6 hours
_CLEANUP_INTERVAL_SECS: float = 300.0
_last_cleanup: float = 0.0


def _cleanup_output_dir() -> None:
    import time
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CLEANUP_INTERVAL_SECS:
        return
    _last_cleanup = now
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.is_dir():
        return
    threshold = now - _OUTPUT_MAX_AGE_SECS
    for f in out_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < threshold:
            try:
                f.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# AgentInfo
# ---------------------------------------------------------------------------

AGENT_GROUPS = (["tool"], ["tool"])

AGENT_INFO = AgentInfo(
    agent_id=_OUR_AGENT_ID,
    description=(
        "MiniMax media-generation suite. Generates images from a text prompt "
        "(and, in future releases, speech / music / video). Put the creative "
        "request in llmdata.prompt, background constraints (style, mood, "
        "aspect ratio hints) in llmdata.context, and optional reference "
        "images in files. Returns the generated media as ProxyFile "
        "attachments plus a short summary of what was produced."
    ),
    input_schema="llmdata: LLMData, files: Optional[List[ProxyFile]]",
    output_schema="content: str, files: Optional[List[ProxyFile]]",
    required_input=["llmdata"],
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are MiniMax Media Agent, a specialist that turns a natural-language \
creative brief into generated media using MiniMax's media APIs.

## Process
1. Read the user's brief and any context carefully.
2. Translate it into a vivid, concrete prompt suited to the target model. \
Expand sparse briefs with sensory detail (subject, setting, lighting, \
composition, style), but stay faithful to the user's intent.
3. Pick appropriate parameters (aspect ratio, count, reference images if \
provided) and call the matching tool.
4. After tools return, write a brief final message (1-3 sentences) describing \
what you produced. Do NOT re-describe the prompt verbatim — the media itself \
is returned separately.

## Tools
You currently have image generation. Additional tools (speech, music, video) \
may be added over time; use whichever tools are available for the request.

## Rules
- Call a generation tool at least once unless the request is clearly \
impossible or unsafe.
- Do NOT make multiple redundant calls for the same asset. If the user asks \
for "a few variations", pass n>1 to a single tool call.
- If reference images are supplied via the input `files`, they will be listed \
in the context. Pass their URLs to the tool's reference-image parameter when \
the user wants visual consistency with them.
- Keep your final text short. The generated files speak for themselves.
"""


# ---------------------------------------------------------------------------
# MiniMax Anthropic-compatible LLM call
# ---------------------------------------------------------------------------


async def _call_minimax_llm(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """POST to MiniMax's Anthropic-compatible /v1/messages endpoint.

    Returns the parsed JSON response body.  Raises RuntimeError on
    non-2xx responses or transport failures.
    """
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY is not configured")

    url = f"{MINIMAX_API_BASE}/anthropic/v1/messages"
    headers = {
        "x-api-key": MINIMAX_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body: dict[str, Any] = {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "system": system,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        raise RuntimeError(
            f"MiniMax LLM error {r.status_code}: {r.text[:500]}"
        )
    try:
        return r.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse MiniMax response: {exc}") from exc


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------


async def _run(data: dict[str, Any]) -> dict[str, Any]:
    task_id: str = data.get("task_id", "")
    parent_task_id: Optional[str] = data.get("parent_task_id")
    raw_payload: dict[str, Any] = data.get("payload", {})

    llmdata_raw = raw_payload.get("llmdata")
    if not llmdata_raw or not llmdata_raw.get("prompt"):
        return build_result_request(
            agent_id=_OUR_AGENT_ID,
            task_id=task_id,
            parent_task_id=parent_task_id,
            status_code=400,
            output=AgentOutput(content="Error: payload.llmdata.prompt is required"),
        )
    llmdata = LLMData.model_validate(llmdata_raw)

    if not MINIMAX_API_KEY:
        return build_result_request(
            agent_id=_OUR_AGENT_ID,
            task_id=task_id,
            parent_task_id=parent_task_id,
            status_code=500,
            output=AgentOutput(
                content=(
                    "Error: MINIMAX_API_KEY is not configured. Set it in "
                    "agents/minimax_media_agent/data/config.json."
                ),
            ),
        )

    # Per-request ProxyFileManager — keeps inbox/output scoped per invocation
    # but reuses the shared agent directory.
    pfm = ProxyFileManager(
        inbox_dir=_AGENT_DIR / "data" / "inbox",
        router_url=ROUTER_URL,
    )

    # ------------------------------------------------------------------
    # Resolve inbound reference files
    # ------------------------------------------------------------------
    ref_files_raw: list[dict[str, Any]] = raw_payload.get("files") or []
    ref_entries: list[dict[str, str]] = []
    for pf in ref_files_raw:
        try:
            local_path = await pfm.fetch(pf, task_id)
        except Exception as exc:
            logger.warning("Failed to fetch reference file %s: %s", pf, exc)
            continue
        # If the original is already accessible as an http URL, the LLM can
        # pass it to MiniMax directly.  Otherwise we hand the local path to
        # the tool and let it encode to a data-URI at call time.
        original_url = pf.get("path") if pf.get("protocol") == "http" else None
        ref_entries.append({
            "filename": Path(local_path).name,
            "local_path": local_path,
            "url": original_url or "",
        })

    # ------------------------------------------------------------------
    # Build message history
    # ------------------------------------------------------------------
    system_parts = [_SYSTEM_PROMPT]
    if llmdata.agent_instruction:
        system_parts.append(
            f"## Additional Instructions\n{llmdata.agent_instruction}"
        )
    if llmdata.context:
        system_parts.append(f"## Context\n{llmdata.context}")
    if ref_entries:
        lines = ["## Reference Files Provided"]
        for i, e in enumerate(ref_entries, 1):
            if e["url"]:
                lines.append(f"{i}. {e['filename']} — url: {e['url']}")
            else:
                lines.append(f"{i}. {e['filename']} — local: {e['local_path']}")
        lines.append(
            "Pass one of these URLs (or local paths) via the "
            "`reference_image` parameter when the user wants visual "
            "consistency with a reference."
        )
        system_parts.append("\n".join(lines))

    system = "\n\n".join(system_parts)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": llmdata.prompt}]}
    ]

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------
    ctx = ToolContext(
        api_key=MINIMAX_API_KEY,
        api_base=MINIMAX_API_BASE,
        image_model=IMAGE_MODEL,
        output_dir=Path(OUTPUT_DIR),
        http_timeout=HTTP_TIMEOUT,
        pfm=pfm,
    )

    produced_files: list[ProxyFile] = []
    total_tool_calls = 0
    final_text_parts: list[str] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        try:
            resp = await asyncio.wait_for(
                _call_minimax_llm(system, messages, MINIMAX_TOOLS),
                timeout=AGENT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return build_result_request(
                agent_id=_OUR_AGENT_ID,
                task_id=task_id,
                parent_task_id=parent_task_id,
                status_code=504,
                output=AgentOutput(
                    content="MiniMax LLM call timed out.",
                    files=produced_files or None,
                ),
            )
        except Exception as exc:
            logger.exception("MiniMax LLM call failed")
            return build_result_request(
                agent_id=_OUR_AGENT_ID,
                task_id=task_id,
                parent_task_id=parent_task_id,
                status_code=502,
                output=AgentOutput(
                    content=f"MiniMax LLM error: {exc}",
                    files=produced_files or None,
                ),
            )

        content_blocks: list[dict[str, Any]] = resp.get("content") or []
        stop_reason: str = resp.get("stop_reason") or ""

        # Append the assistant turn verbatim (required for multi-turn tool use).
        messages.append({"role": "assistant", "content": content_blocks})

        # Extract tool_use + text
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
        iter_text = "".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        )
        if iter_text:
            final_text_parts.append(iter_text)

        if not tool_uses or stop_reason != "tool_use":
            break  # assistant is done

        # Execute each tool call, build tool_result blocks for the next turn.
        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            total_tool_calls += 1
            if total_tool_calls > MAX_TOOL_CALLS:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id", ""),
                    "content": "Error: tool call limit reached. "
                               "Respond with your final summary now.",
                    "is_error": True,
                })
                continue
            name = tu.get("name", "")
            args = tu.get("input", {}) or {}
            try:
                result_text, new_files = await execute_tool(name, args, ctx)
            except Exception as exc:
                logger.exception("Tool '%s' raised", name)
                result_text = f"Tool error: {exc}"
                new_files = []
            produced_files.extend(new_files)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.get("id", ""),
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})

        # Gentle nudge as we approach the iteration cap.
        remaining = MAX_ITERATIONS - iteration
        if remaining <= 1:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"[System] {remaining} iteration(s) remaining. "
                        "Write your final short summary now; do not call "
                        "more tools."
                    ),
                }],
            })

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    summary = "\n".join(p for p in final_text_parts if p).strip()
    if not summary:
        if produced_files:
            summary = f"Generated {len(produced_files)} media file(s)."
        else:
            summary = "No media was generated."

    return build_result_request(
        agent_id=_OUR_AGENT_ID,
        task_id=task_id,
        parent_task_id=parent_task_id,
        status_code=200,
        output=AgentOutput(
            content=summary,
            files=produced_files or None,
        ),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MiniMax Media Agent")


@app.post("/receive")
async def receive(request: Request) -> JSONResponse:
    """Called by the router via in-process ASGI transport."""
    _refresh_config()
    _cleanup_output_dir()
    data = await request.json()
    try:
        result = await _run(data)
        return JSONResponse(status_code=200, content=result)
    except Exception as exc:
        logger.exception("Unhandled error in minimax_media_agent")
        task_id = data.get("task_id", "")
        parent_task_id = data.get("parent_task_id")
        return JSONResponse(
            status_code=200,
            content=build_result_request(
                agent_id=_OUR_AGENT_ID,
                task_id=task_id,
                parent_task_id=parent_task_id,
                status_code=500,
                output=AgentOutput(content=f"minimax_media_agent error: {exc}"),
            ),
        )
