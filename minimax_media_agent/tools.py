"""
agents/minimax_media_agent/tools.py — MiniMax media-generation tool
definitions and executors.

Currently ships image generation plus the speech suite (voice listing,
asynchronous T2A, and rapid voice cloning).  Music and video will slot in
behind the same MINIMAX_TOOLS list + execute_tool dispatcher.

Each executor returns ``(result_text, files)`` where:

- ``result_text``  — a short status string fed back to the LLM as the
  ``tool_result`` content.
- ``files``        — list of :class:`ProxyFile` entries to append to the
  agent's outbound ``AgentOutput.files``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from helper import ProxyFile, ProxyFileManager
from voices import SYSTEM_VOICES, format_voices

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
    speech_model: str
    output_dir: Path
    http_timeout: float
    pfm: ProxyFileManager
    # Speech-specific: long audio tasks are asynchronous.
    speech_poll_interval: float = 3.0
    speech_max_wait: float = 600.0
    ref_map: dict[str, dict[str, str]] = field(default_factory=dict)
    inbox_resolver: Optional[Callable[[str], Optional[str]]] = None


# ---------------------------------------------------------------------------
# MiniMax HTTP helpers
# ---------------------------------------------------------------------------


def _auth_headers(api_key: str, *, json_body: bool = False) -> dict[str, str]:
    h = {"Authorization": f"Bearer {api_key}"}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _base_resp_ok(body: dict[str, Any]) -> tuple[bool, str]:
    """Inspect MiniMax's ``base_resp`` envelope.  Returns (ok, message)."""
    br = body.get("base_resp") or {}
    try:
        code = int(br.get("status_code", 0) or 0)
    except (TypeError, ValueError):
        code = -1
    msg = str(br.get("status_msg") or "")
    return code == 0, msg


async def _mm_upload_file(
    ctx: ToolContext,
    local_path: Path,
    purpose: str,
    display_name: Optional[str] = None,
) -> str:
    """POST /v1/files/upload; return the resulting ``file_id`` as a string."""
    url = f"{ctx.api_base}/v1/files/upload"
    name = display_name or local_path.name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with local_path.open("rb") as fh:
        files = {"file": (name, fh, mime)}
        data = {"purpose": purpose}
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                url, headers=_auth_headers(ctx.api_key), data=data, files=files,
            )
    if r.status_code >= 400:
        raise RuntimeError(f"upload failed {r.status_code}: {r.text[:300]}")
    body = r.json()
    ok, msg = _base_resp_ok(body)
    file_id = ((body.get("file") or {}).get("file_id"))
    if not ok or file_id is None:
        raise RuntimeError(f"upload rejected: {msg or body}")
    return str(file_id)


async def _mm_retrieve_file_bytes(ctx: ToolContext, file_id: str) -> bytes:
    """GET /v1/files/retrieve_content; return raw audio bytes."""
    url = f"{ctx.api_base}/v1/files/retrieve_content"
    async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
        r = await client.get(
            url, headers=_auth_headers(ctx.api_key), params={"file_id": file_id},
        )
    if r.status_code >= 400:
        raise RuntimeError(f"retrieve failed {r.status_code}: {r.text[:300]}")
    # retrieve_content returns binary directly.
    return r.content


async def _mm_poll_async_t2a(ctx: ToolContext, task_id: str) -> str:
    """Poll /v1/query/t2a_async_query_v2 until the task succeeds or fails.

    Returns the resulting ``file_id`` string.  Raises RuntimeError on
    timeout or task-level failure.
    """
    url = f"{ctx.api_base}/v1/query/t2a_async_query_v2"
    deadline = asyncio.get_running_loop().time() + ctx.speech_max_wait
    async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
        while True:
            r = await client.get(
                url,
                headers=_auth_headers(ctx.api_key),
                params={"task_id": task_id},
            )
            if r.status_code >= 400:
                raise RuntimeError(
                    f"poll failed {r.status_code}: {r.text[:300]}"
                )
            body = r.json()
            status = str(body.get("status") or "").strip()
            if status == "Success":
                file_id = body.get("file_id")
                if file_id is None:
                    raise RuntimeError(f"task {task_id} success but no file_id: {body}")
                return str(file_id)
            if status in ("Failed", "Expired"):
                raise RuntimeError(
                    f"task {task_id} {status.lower()}: {body}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"task {task_id} timed out after {ctx.speech_max_wait}s"
                )
            await asyncio.sleep(ctx.speech_poll_interval)


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
    {
        "name": "list_system_voices",
        "description": (
            "List MiniMax's built-in system voice IDs.  Call this BEFORE "
            "generate_speech when you need to pick a voice — the full list "
            "is long, so keep the `language` filter as specific as possible "
            "(e.g. 'english', 'korean', 'chinese'). Returns voice_id + "
            "human name, grouped by language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": (
                        "Case-insensitive language substring filter "
                        "(e.g. 'english'). Omit to dump the entire catalogue."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_speech",
        "description": (
            "Synthesize speech from text with a MiniMax voice. Uses the "
            "asynchronous long-form T2A pipeline (create → poll → download) "
            "so input up to 1,000,000 characters is supported. The returned "
            "audio file is attached to the agent's final response as a "
            "ProxyFile — do NOT ask the user to wait for a URL.\n\n"
            "`voice_id` is REQUIRED. It is either a system voice (see "
            "list_system_voices) or a custom voice_id returned by "
            "clone_voice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to synthesize. Up to 1,000,000 characters.",
                },
                "voice_id": {
                    "type": "string",
                    "description": (
                        "MiniMax voice_id. Use list_system_voices to pick a "
                        "built-in one, or pass a voice_id returned from a "
                        "previous clone_voice call."
                    ),
                },
                "model": {
                    "type": "string",
                    "enum": [
                        "speech-2.8-hd", "speech-2.8-turbo",
                        "speech-2.6-hd", "speech-2.6-turbo",
                        "speech-02-hd", "speech-02-turbo",
                    ],
                    "description": (
                        "Speech model. *-hd favours fidelity, *-turbo favours "
                        "speed/cost. Default comes from agent config."
                    ),
                },
                "language_boost": {
                    "type": "string",
                    "description": (
                        "Optional language hint (e.g. 'English', 'Korean', "
                        "'auto'). Default: 'auto'."
                    ),
                },
                "speed": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 2.0,
                    "description": "Speech rate multiplier. Default: 1.",
                },
                "vol": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Volume 0-10. Default: 1.",
                },
                "pitch": {
                    "type": "number",
                    "minimum": -12,
                    "maximum": 12,
                    "description": "Pitch shift in semitones (-12 to +12). Default: 0.",
                },
                "format": {
                    "type": "string",
                    "enum": ["mp3", "wav", "flac", "pcm"],
                    "description": "Output audio format. Default: mp3.",
                },
            },
            "required": ["text", "voice_id"],
        },
    },
    {
        "name": "clone_voice",
        "description": (
            "Clone a voice from a user-supplied source audio file. Returns "
            "the voice_id that the caller should then pass to "
            "generate_speech. Optionally also synthesises a short preview "
            "audio if `preview_text` is provided, returned as a ProxyFile.\n\n"
            "Source audio requirements (MiniMax): mp3/m4a/wav, 10 s to 5 min, "
            "≤ 20 MB. Supply it via the `source_audio` filename — it must "
            "be listed under 'Reference Files' in the system prompt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_audio": {
                    "type": "string",
                    "description": (
                        "Filename of the source audio (from Reference Files)."
                    ),
                },
                "voice_id": {
                    "type": "string",
                    "description": (
                        "Custom voice_id to assign to the clone. Must be "
                        "unique per MiniMax account; typically 8+ chars, "
                        "alphanumeric plus underscore/hyphen."
                    ),
                },
                "preview_text": {
                    "type": "string",
                    "description": (
                        "Optional preview text. If provided, MiniMax returns "
                        "a short audio sample of the cloned voice; this "
                        "agent will attach it as a ProxyFile."
                    ),
                },
                "model": {
                    "type": "string",
                    "enum": [
                        "speech-2.8-hd", "speech-2.8-turbo",
                        "speech-2.6-hd", "speech-2.6-turbo",
                        "speech-02-hd", "speech-02-turbo",
                    ],
                    "description": "Model used for the preview synthesis.",
                },
            },
            "required": ["source_audio", "voice_id"],
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
    if name == "list_system_voices":
        return await _list_system_voices(args, ctx)
    if name == "generate_speech":
        return await _generate_speech(args, ctx)
    if name == "clone_voice":
        return await _clone_voice(args, ctx)
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


# ---------------------------------------------------------------------------
# list_system_voices
# ---------------------------------------------------------------------------


async def _list_system_voices(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    language = args.get("language")
    return (format_voices(str(language) if language else None), [])


# ---------------------------------------------------------------------------
# generate_speech (async T2A v2)
# ---------------------------------------------------------------------------


# MiniMax async T2A defaults.  Kept close to the tool so callers can audit.
_SPEECH_DEFAULT_FORMAT: str = "mp3"
_SPEECH_DEFAULT_SAMPLE_RATE: int = 32000
_SPEECH_DEFAULT_BITRATE: int = 128000
_SPEECH_DEFAULT_CHANNEL: int = 2


def _ext_for_format(fmt: str) -> str:
    fmt = (fmt or "mp3").lower()
    if fmt in ("mp3", "wav", "flac"):
        return fmt
    if fmt == "pcm":
        return "pcm"
    return "mp3"


async def _generate_speech(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    text = str(args.get("text", "")).strip()
    voice_id = str(args.get("voice_id", "")).strip()
    if not text:
        return ("Error: text is required", [])
    if not voice_id:
        return (
            "Error: voice_id is required. Call list_system_voices first "
            "or pass a voice_id returned from clone_voice.",
            [],
        )

    model = str(args.get("model") or ctx.speech_model).strip() or ctx.speech_model
    language_boost = str(args.get("language_boost") or "auto")
    fmt = str(args.get("format") or _SPEECH_DEFAULT_FORMAT).lower()

    def _num(key: str, default: float) -> float:
        v = args.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    payload: dict[str, Any] = {
        "model": model,
        "text": text,
        "language_boost": language_boost,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": _num("speed", 1.0),
            "vol": _num("vol", 1.0),
            "pitch": _num("pitch", 0.0),
        },
        "audio_setting": {
            "audio_sample_rate": _SPEECH_DEFAULT_SAMPLE_RATE,
            "bitrate": _SPEECH_DEFAULT_BITRATE,
            "format": fmt if fmt in ("mp3", "wav", "flac", "pcm") else "mp3",
            "channel": _SPEECH_DEFAULT_CHANNEL,
        },
    }

    # 1. Create async task.
    create_url = f"{ctx.api_base}/v1/t2a_async_v2"
    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                create_url,
                headers=_auth_headers(ctx.api_key, json_body=True),
                json=payload,
            )
    except Exception as exc:
        return (f"Error: T2A create request failed: {exc}", [])
    if r.status_code >= 400:
        return (f"Error: T2A create {r.status_code}: {r.text[:400]}", [])
    try:
        create_body = r.json()
    except Exception as exc:
        return (f"Error: unparseable T2A create response: {exc}", [])
    ok, msg = _base_resp_ok(create_body)
    task_id = create_body.get("task_id")
    if not ok or task_id is None:
        return (f"Error: T2A create rejected: {msg or create_body}", [])

    # 2. Poll until done.
    try:
        file_id = await _mm_poll_async_t2a(ctx, str(task_id))
    except Exception as exc:
        return (f"Error: T2A task failed: {exc}", [])

    # 3. Retrieve audio bytes.
    try:
        audio_bytes = await _mm_retrieve_file_bytes(ctx, file_id)
    except Exception as exc:
        return (f"Error: could not retrieve T2A output: {exc}", [])
    if not audio_bytes:
        return ("Error: T2A returned empty audio.", [])

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ext = _ext_for_format(fmt)
    fname = f"minimax_speech_{uuid.uuid4().hex[:10]}.{ext}"
    out_path = ctx.output_dir / fname
    out_path.write_bytes(audio_bytes)
    pf_dict = ctx.pfm.resolve(str(out_path.resolve()))
    if pf_dict is None:
        return ("Error: failed to register generated audio as a ProxyFile.", [])

    summary = (
        f"Generated speech with voice '{voice_id}' using {model} "
        f"({len(audio_bytes)} bytes, {ext}): {fname}"
    )
    return (summary, [ProxyFile(**pf_dict)])


# ---------------------------------------------------------------------------
# clone_voice
# ---------------------------------------------------------------------------


_CLONEABLE_EXTS: set[str] = {".mp3", ".m4a", ".wav"}
_CLONE_MAX_BYTES: int = 20 * 1024 * 1024  # 20 MB per MiniMax docs


async def _clone_voice(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    source_name = str(args.get("source_audio", "")).strip()
    voice_id = str(args.get("voice_id", "")).strip()
    if not source_name:
        return ("Error: source_audio filename is required", [])
    if not voice_id:
        return ("Error: voice_id is required", [])

    # Resolve filename → local path (prefer ref_map, fall back to inbox).
    entry = ctx.ref_map.get(source_name)
    local_path_str: Optional[str] = None
    if entry:
        lp = entry.get("local_path", "")
        if lp and Path(lp).is_file():
            local_path_str = lp
    if local_path_str is None and ctx.inbox_resolver is not None:
        local_path_str = ctx.inbox_resolver(source_name)
    if not local_path_str:
        return (
            f"Error: source_audio '{source_name}' not found in inbox. "
            "Provide it via the agent's `files` input.",
            [],
        )

    local_path = Path(local_path_str)
    if local_path.suffix.lower() not in _CLONEABLE_EXTS:
        return (
            f"Error: unsupported source format '{local_path.suffix}'. "
            "MiniMax accepts mp3, m4a, or wav.",
            [],
        )
    try:
        size = local_path.stat().st_size
    except OSError as exc:
        return (f"Error: cannot stat source audio: {exc}", [])
    if size > _CLONE_MAX_BYTES:
        return (
            f"Error: source audio is {size} bytes; MiniMax limit is 20 MB.",
            [],
        )

    # 1. Upload source audio.
    try:
        file_id = await _mm_upload_file(ctx, local_path, purpose="voice_clone")
    except Exception as exc:
        return (f"Error: upload failed: {exc}", [])

    # 2. Call voice_clone.  Example (prompt) audio intentionally omitted —
    # see agent spec: only source audio is populated for simplicity.
    clone_payload: dict[str, Any] = {
        "file_id": file_id,
        "voice_id": voice_id,
        "model": str(args.get("model") or ctx.speech_model),
    }
    preview_text = str(args.get("preview_text") or "").strip()
    if preview_text:
        clone_payload["text"] = preview_text

    clone_url = f"{ctx.api_base}/v1/voice_clone"
    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                clone_url,
                headers=_auth_headers(ctx.api_key, json_body=True),
                json=clone_payload,
            )
    except Exception as exc:
        return (f"Error: voice_clone request failed: {exc}", [])
    if r.status_code >= 400:
        return (
            f"Error: voice_clone {r.status_code}: {r.text[:400]}",
            [],
        )
    try:
        body = r.json()
    except Exception as exc:
        return (f"Error: unparseable voice_clone response: {exc}", [])
    ok, msg = _base_resp_ok(body)
    if not ok:
        return (f"Error: voice_clone rejected: {msg or body}", [])

    # 3. Optionally attach preview audio.  MiniMax returns a public URL in
    # ``demo_audio`` when ``text`` was supplied.
    files: list[ProxyFile] = []
    demo_url = body.get("demo_audio") or body.get("demo_audio_url")
    if preview_text and isinstance(demo_url, str) and demo_url.startswith("http"):
        try:
            async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
                dr = await client.get(demo_url)
                dr.raise_for_status()
                audio_bytes = dr.content
        except Exception as exc:
            logger.warning("Failed to download preview audio: %s", exc)
            audio_bytes = b""
        if audio_bytes:
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            ext = Path(demo_url.split("?")[0]).suffix or ".mp3"
            fname = f"minimax_clone_preview_{uuid.uuid4().hex[:10]}{ext}"
            out_path = ctx.output_dir / fname
            out_path.write_bytes(audio_bytes)
            pf_dict = ctx.pfm.resolve(str(out_path.resolve()))
            if pf_dict is not None:
                files.append(ProxyFile(**pf_dict))

    suffix = f" preview attached as {files[0].original_filename}" if files else ""
    summary = (
        f"Voice cloned successfully. Use voice_id='{voice_id}' in "
        f"generate_speech.{suffix}"
    )
    return (summary, files)
