"""
agents/minimax_media_agent/tools.py — MiniMax media-generation tool
definitions and executors.

Only image generation is implemented in this release; speech, music, and
video tools will follow the same pattern (Anthropic-style schema entry
in MEDIA_TOOLS + matching dispatch arm in execute_tool()).

Each executor returns ``(result_text, files)`` where:

- ``result_text``  — a short status string fed back to the LLM as the
  ``tool_result`` content.
- ``files``        — list of :class:`ProxyFile` entries to append to the
  agent's outbound ``AgentOutput.files``.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from helper import ProxyFile, ProxyFileManager

logger = logging.getLogger("minimax_media_agent.tools")


# ---------------------------------------------------------------------------
# Shared execution context
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Dependencies shared by all tool executors.

    ``ref_map`` maps LLM-visible filenames (as presented in the system
    prompt) to ``{"local_path": ..., "url": ...}``.  ``inbox_resolver`` is
    a callable that safely resolves any other filename the LLM provides
    to an absolute path inside the per-task inbox, or returns None on
    miss / path-traversal attempt.
    """
    api_key: str
    api_base: str
    image_model: str
    output_dir: Path
    http_timeout: float
    pfm: ProxyFileManager
    ref_map: dict[str, dict[str, str]] = field(default_factory=dict)
    inbox_resolver: Optional[Callable[[str], Optional[str]]] = None


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool-schema format)
# ---------------------------------------------------------------------------


MINIMAX_TOOLS: list[dict[str, Any]] = [
    {
        "name": "generate_image",
        "description": (
            "Generate one or more images from a text prompt using the MiniMax "
            "image-generation API. Supports optional reference images for "
            "subject/character consistency (image-to-image).  Returns a "
            "summary; the generated images are attached to the agent's "
            "final response as ProxyFile objects — do NOT ask the user to "
            "wait for a URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Vivid, concrete description of the desired image. "
                        "Include subject, setting, lighting, composition, "
                        "and style cues."
                    ),
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": [
                        "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
                        "21:9",
                    ],
                    "description": "Image aspect ratio. Default: 1:1.",
                },
                "n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Number of images to generate (1-4). Default: 1.",
                },
                "reference_image": {
                    "type": "string",
                    "description": (
                        "Optional reference image for subject/character "
                        "consistency. Preferred form: a filename exactly as "
                        "listed under 'Reference Files' in the system prompt "
                        "(e.g. 'hero.jpg'). An http(s) URL is also accepted."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def execute_tool(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    """Route a tool_use invocation to the matching executor."""
    if name == "generate_image":
        return await _generate_image(args, ctx)
    return (f"Error: unknown tool '{name}'", [])


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------


def _file_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _ref_to_image_file(reference: str, ctx: ToolContext) -> str:
    """Convert an LLM-supplied reference into MiniMax's ``image_file`` value.

    Accepted inputs:

    - Http(s) URL or ``data:`` URI — passed through unchanged.
    - A filename listed in ``ctx.ref_map`` — if that entry has a usable
      http URL we forward it, otherwise the local file is base64-encoded
      into a data URI.
    - Any other bare name — resolved against the per-task inbox (with a
      path-traversal check) via ``ctx.inbox_resolver``, then data-URI
      encoded.

    Absolute filesystem paths are rejected: the LLM must use filenames.
    """
    reference = reference.strip()
    if not reference:
        raise ValueError("reference_image is empty")
    if reference.startswith(("http://", "https://", "data:")):
        return reference
    if reference.startswith(("/", "\\")) or ":" in reference[:3]:
        raise ValueError(
            "reference_image must be a filename from the Reference Files "
            "list, or an http(s) URL — absolute paths are not allowed"
        )

    entry = ctx.ref_map.get(reference)
    if entry:
        if entry.get("url"):
            return entry["url"]
        lp = entry.get("local_path", "")
        if lp and Path(lp).is_file():
            return _file_to_data_uri(Path(lp))

    if ctx.inbox_resolver is not None:
        resolved = ctx.inbox_resolver(reference)
        if resolved:
            return _file_to_data_uri(Path(resolved))

    raise ValueError(f"reference_image '{reference}' not found in inbox")


async def _generate_image(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    prompt = str(args.get("prompt", "")).strip()
    if not prompt:
        return ("Error: prompt is required", [])

    aspect_ratio = str(args.get("aspect_ratio") or "1:1")
    n_raw = args.get("n", 1)
    try:
        n = max(1, min(4, int(n_raw)))
    except (TypeError, ValueError):
        n = 1

    payload: dict[str, Any] = {
        "model": ctx.image_model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "n": n,
        "response_format": "base64",
    }

    reference = args.get("reference_image")
    if reference:
        try:
            image_file = _ref_to_image_file(str(reference), ctx)
        except Exception as exc:
            return (f"Error: invalid reference_image: {exc}", [])
        payload["subject_reference"] = [{
            "type": "character",
            "image_file": image_file,
        }]

    url = f"{ctx.api_base}/v1/image_generation"
    headers = {
        "Authorization": f"Bearer {ctx.api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
    except Exception as exc:
        return (f"Error: MiniMax image API request failed: {exc}", [])

    if r.status_code >= 400:
        return (
            f"Error: MiniMax image API {r.status_code}: {r.text[:400]}",
            [],
        )

    try:
        data = r.json()
    except Exception as exc:
        return (f"Error: unparseable MiniMax image response: {exc}", [])

    # MiniMax returns base64 payloads under data.image_base64 (list).
    images_b64 = (data.get("data") or {}).get("image_base64")
    if not images_b64:
        # Some error bodies still return 200; surface the raw payload.
        return (f"Error: no images returned: {str(data)[:400]}", [])

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    files: list[ProxyFile] = []
    saved_names: list[str] = []
    for idx, b64 in enumerate(images_b64):
        try:
            raw = base64.b64decode(b64)
        except Exception as exc:
            logger.warning("Failed to decode image %d: %s", idx, exc)
            continue
        fname = f"minimax_image_{uuid.uuid4().hex[:10]}.jpeg"
        out_path = ctx.output_dir / fname
        out_path.write_bytes(raw)
        pf_dict = ctx.pfm.resolve(str(out_path.resolve()))
        if pf_dict is not None:
            files.append(ProxyFile(**pf_dict))
            saved_names.append(fname)

    if not files:
        return ("Error: image generation returned no usable images.", [])

    summary = (
        f"Generated {len(files)} image(s) at {aspect_ratio}: "
        + ", ".join(saved_names)
    )
    return (summary, files)
