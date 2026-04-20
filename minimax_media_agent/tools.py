"""
agents/minimax_media_agent/tools.py — MiniMax media-generation tool
definitions and executors.

Tools shipped:

- generate_image                — text-to-image / image-to-image.
- list_voices / delete_voice    — voice catalogue management.
- generate_speech               — async long-form T2A.
- clone_voice                   — quick voice cloning (source audio only).
- generate_music                — vocal songs, instrumentals, covers.
- generate_video                — text, image, first-last-frame, subject-
                                  reference video.

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
    music_model: str
    video_model: str
    output_dir: Path
    http_timeout: float
    pfm: ProxyFileManager
    # Speech-specific: long audio tasks are asynchronous.
    speech_poll_interval: float = 3.0
    speech_max_wait: float = 600.0
    # Video-specific: tasks take minutes, poll slower, wait longer.
    video_poll_interval: float = 10.0
    video_max_wait: float = 1800.0
    ref_map: dict[str, dict[str, str]] = field(default_factory=dict)
    inbox_resolver: Optional[Callable[[str], Optional[str]]] = None
    # Voice_ids minted during this request (clone_voice, etc.).  The agent
    # reads this after the tool loop and surfaces the IDs in its final
    # message so the user can reuse them later.
    created_voice_ids: list[str] = field(default_factory=list)


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
    """GET /v1/files/retrieve_content; return raw file bytes.

    Used by the async T2A path.  Video uses /v1/files/retrieve instead —
    see :func:`_mm_get_file_download_url`.
    """
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


async def _mm_poll_video_task(ctx: ToolContext, task_id: str) -> str:
    """Poll /v1/query/video_generation until the task succeeds or fails.

    Uses longer interval / cap than the speech poller because video
    generation is measured in minutes, not seconds.
    """
    url = f"{ctx.api_base}/v1/query/video_generation"
    deadline = asyncio.get_running_loop().time() + ctx.video_max_wait
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
                if not file_id:
                    raise RuntimeError(
                        f"video task {task_id} success but no file_id: {body}"
                    )
                return str(file_id)
            if status == "Fail":
                err = body.get("error_message") or body
                raise RuntimeError(f"video task {task_id} failed: {err}")
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"video task {task_id} timed out after {ctx.video_max_wait}s"
                )
            await asyncio.sleep(ctx.video_poll_interval)


async def _mm_get_file_download_url(ctx: ToolContext, file_id: str) -> str:
    """GET /v1/files/retrieve; return the file's download URL.

    This is the JSON-wrapped retrieval path used by the video workflow,
    distinct from /v1/files/retrieve_content which streams bytes
    directly (used for async T2A).
    """
    url = f"{ctx.api_base}/v1/files/retrieve"
    async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
        r = await client.get(
            url,
            headers=_auth_headers(ctx.api_key),
            params={"file_id": file_id},
        )
    if r.status_code >= 400:
        raise RuntimeError(f"retrieve failed {r.status_code}: {r.text[:300]}")
    body = r.json()
    download_url = (body.get("file") or {}).get("download_url")
    if not download_url:
        raise RuntimeError(f"retrieve returned no download_url: {body}")
    return str(download_url)


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
        "name": "list_voices",
        "description": (
            "List voices available to this MiniMax account via the "
            "/v1/get_voice API. Use BEFORE generate_speech when picking a "
            "voice. Covers four categories selectable by `voice_type`:\n"
            "  - 'system'           : MiniMax's built-in voices.\n"
            "  - 'voice_cloning'    : voices the user cloned via clone_voice.\n"
            "  - 'voice_generation' : voices created via voice-design (TTV).\n"
            "  - 'all'              : everything the account can use.\n\n"
            "NOTE (MiniMax quirk): a newly cloned voice does NOT appear in "
            "this list until it has been used for speech synthesis at least "
            "once. If clone_voice just returned a voice_id, trust that ID "
            "and call generate_speech with it directly — do not wait for "
            "it to show up here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "voice_type": {
                    "type": "string",
                    "enum": ["system", "voice_cloning", "voice_generation", "all"],
                    "description": (
                        "Which category to list (default: 'system')."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring filter applied "
                        "to system voice_id / voice_name only (e.g. "
                        "'english'). The system voice catalogue is large — "
                        "use this to keep output focused. Ignored for "
                        "cloned / generated voices (those are typically "
                        "short lists already)."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "delete_voice",
        "description": (
            "Delete a voice the current account owns. Only voices created "
            "via clone_voice ('voice_cloning') or voice-design "
            "('voice_generation') can be deleted — system voices are not "
            "deletable. WARNING: the voice_id cannot be reused after "
            "deletion. Only call when the user explicitly asks to remove a "
            "voice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "voice_id": {
                    "type": "string",
                    "description": "The voice_id to delete.",
                },
                "voice_type": {
                    "type": "string",
                    "enum": ["voice_cloning", "voice_generation"],
                    "description": "Category of the voice.",
                },
            },
            "required": ["voice_id", "voice_type"],
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
            "list_voices with voice_type='system') or a custom voice_id "
            "returned by clone_voice."
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
                        "MiniMax voice_id. Use list_voices to pick a "
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
    {
        "name": "generate_music",
        "description": (
            "Generate a song (vocal or instrumental) or cover using "
            "MiniMax's music API. Three modes are selected by `model`:\n"
            "  - 'music-2.6' / 'music-2.6-free': compose an original song "
            "from a `prompt` (style/mood) and `lyrics`. Set "
            "`is_instrumental: true` for a purely instrumental piece "
            "(then `prompt` is required, `lyrics` ignored). Set "
            "`lyrics_optimizer: true` to have MiniMax auto-write lyrics "
            "from the prompt when you don't have any.\n"
            "  - 'music-cover' / 'music-cover-free': generate a cover of "
            "`reference_audio` in the style described by `prompt` "
            "(10-300 chars). `lyrics` is optional — omit to ASR them from "
            "the reference.\n\n"
            "Returns a single audio file attached as a ProxyFile. The "
            "'-free' models are available to all API key holders with "
            "lower RPM; the paid variants have higher rate limits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Description of style / mood / scenario, e.g. "
                        "'Soulful blues, rainy night, slow tempo'. Up to "
                        "2000 chars. Required for instrumental and cover "
                        "modes."
                    ),
                },
                "lyrics": {
                    "type": "string",
                    "description": (
                        "Song lyrics, newline-separated. Supports structure "
                        "tags [Intro] [Verse] [Pre Chorus] [Chorus] "
                        "[Interlude] [Bridge] [Outro] [Post Chorus] "
                        "[Transition] [Break] [Hook] [Build Up] [Inst] "
                        "[Solo]. Required for vocal music-2.6 songs unless "
                        "`lyrics_optimizer` is true. Up to 3500 chars."
                    ),
                },
                "model": {
                    "type": "string",
                    "enum": [
                        "music-2.6", "music-2.6-free",
                        "music-cover", "music-cover-free",
                    ],
                    "description": "Music model. Default comes from agent config.",
                },
                "is_instrumental": {
                    "type": "boolean",
                    "description": (
                        "Instrumental-only (no vocals). Only valid with "
                        "music-2.6 / music-2.6-free. Default: false."
                    ),
                },
                "lyrics_optimizer": {
                    "type": "boolean",
                    "description": (
                        "Auto-generate lyrics from `prompt` when `lyrics` "
                        "is empty. Only valid with music-2.6 / "
                        "music-2.6-free. Default: false."
                    ),
                },
                "reference_audio": {
                    "type": "string",
                    "description": (
                        "Required for music-cover / music-cover-free. A "
                        "filename exactly as listed under 'Reference Files' "
                        "(mp3/wav/flac/m4a, 6 s-6 min, ≤ 50 MB), or an "
                        "http(s) URL."
                    ),
                },
                "format": {
                    "type": "string",
                    "enum": ["mp3", "wav", "pcm"],
                    "description": "Output audio format. Default: mp3.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_video",
        "description": (
            "Generate a video with MiniMax. Four modes, auto-selected by "
            "which parameters you pass:\n"
            "  - text-to-video: `prompt` only.\n"
            "  - image-to-video: `prompt` + `first_frame_image`.\n"
            "  - first-last-frame: `prompt` + `first_frame_image` + "
            "`last_frame_image`.\n"
            "  - subject-reference: `prompt` + `subject_reference` "
            "(one face photo; keeps the subject consistent across the "
            "clip).\n\n"
            "Each mode has a default model:\n"
            "  - text / image-to-video: MiniMax-Hailuo-2.3\n"
            "  - first-last-frame    : MiniMax-Hailuo-02\n"
            "  - subject-reference   : S2V-01\n"
            "Override via `model` only if you know the specific model you "
            "want. Video generation is asynchronous and typically takes "
            "1-5 minutes; the agent polls until done and attaches the "
            "resulting mp4 as a ProxyFile — do NOT warn the user that a "
            "URL is coming later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What should happen in the video. For image-to-"
                        "video, describe how the scene evolves from the "
                        "first frame. Some models support camera hints "
                        "like [pan], [zoom], [static] after key phrases."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Optional model override, e.g. 'MiniMax-Hailuo-2.3', "
                        "'MiniMax-Hailuo-02', 'S2V-01'. Leave unset to pick "
                        "the default for the inferred mode."
                    ),
                },
                "duration": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Video length in seconds. Default: 6.",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["512P", "768P", "1080P"],
                    "description": "Output resolution. Default: 1080P.",
                },
                "first_frame_image": {
                    "type": "string",
                    "description": (
                        "Filename from Reference Files (jpg/png/webp), or "
                        "an http(s) URL. Triggers image-to-video mode "
                        "(plus first-last-frame if `last_frame_image` is "
                        "also set)."
                    ),
                },
                "last_frame_image": {
                    "type": "string",
                    "description": (
                        "Filename or URL of the desired ending frame. "
                        "Requires `first_frame_image`; triggers first-last-"
                        "frame mode."
                    ),
                },
                "subject_reference": {
                    "type": "string",
                    "description": (
                        "One face photo for subject-reference mode. "
                        "Filename from Reference Files or http(s) URL. "
                        "Cannot be combined with first/last_frame_image."
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
    if name == "list_voices":
        return await _list_voices(args, ctx)
    if name == "delete_voice":
        return await _delete_voice(args, ctx)
    if name == "generate_speech":
        return await _generate_speech(args, ctx)
    if name == "clone_voice":
        return await _clone_voice(args, ctx)
    if name == "generate_music":
        return await _generate_music(args, ctx)
    if name == "generate_video":
        return await _generate_video(args, ctx)
    return (f"Error: unknown tool '{name}'", [])


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------


def _file_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _ref_to_image_file(reference: str, ctx: ToolContext) -> str:
    """Resolve an LLM-supplied image reference to a value MiniMax accepts.

    Used for every tool parameter that carries an image: image-gen's
    ``reference_image`` and video's ``first_frame_image``,
    ``last_frame_image``, ``subject_reference``.

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
        raise ValueError("reference is empty")
    if reference.startswith(("http://", "https://", "data:")):
        return reference
    if reference.startswith(("/", "\\")) or ":" in reference[:3]:
        raise ValueError(
            "reference must be a filename from Reference Files, or an "
            "http(s) URL — absolute paths are not allowed"
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

    raise ValueError(f"'{reference}' not found in inbox")


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
# list_voices / delete_voice
# ---------------------------------------------------------------------------


_VOICE_CATEGORY_LABELS: dict[str, str] = {
    "system_voice": "System voices",
    "voice_cloning": "Cloned voices",
    "voice_generation": "Voice-design (TTV) voices",
}


def _format_voice_list(
    body: dict[str, Any],
    language: Optional[str],
) -> str:
    """Render the /v1/get_voice response as grouped markdown.

    Applies the optional ``language`` substring filter to system voices
    only (cloned / generated voices carry no language metadata).
    """
    needle = (language or "").strip().lower()
    sections: list[str] = []
    total = 0

    for key, label in _VOICE_CATEGORY_LABELS.items():
        entries = body.get(key) or []
        if not isinstance(entries, list) or not entries:
            continue
        rows: list[str] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            vid = str(e.get("voice_id") or "").strip()
            if not vid:
                continue
            name = str(e.get("voice_name") or "").strip()
            descs = e.get("description") or []
            desc = (
                "; ".join(str(d) for d in descs if d).strip()
                if isinstance(descs, list) else str(descs or "").strip()
            )
            created = str(e.get("created_time") or "").strip()

            if needle and key == "system_voice":
                haystack = f"{vid} {name}".lower()
                if needle not in haystack:
                    continue

            parts: list[str] = [f"`{vid}`"]
            if name:
                parts.append(f"— {name}")
            if desc:
                parts.append(f"({desc})")
            if created:
                parts.append(f"[created {created}]")
            rows.append("- " + " ".join(parts))
        if rows:
            sections.append(f"## {label}\n" + "\n".join(rows))
            total += len(rows)

    if total == 0:
        if language:
            return f"No voices matched '{language}'."
        return "No voices available for this account."
    header = (
        f"{total} voice(s) "
        + (f"matching '{language}' " if language else "")
        + "available:"
    )
    return header + "\n\n" + "\n\n".join(sections)


async def _list_voices(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    voice_type = str(args.get("voice_type") or "system").strip().lower()
    if voice_type not in ("system", "voice_cloning", "voice_generation", "all"):
        return (
            f"Error: voice_type must be one of system, voice_cloning, "
            f"voice_generation, all (got '{voice_type}').",
            [],
        )
    language = args.get("language")
    url = f"{ctx.api_base}/v1/get_voice"
    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                url,
                headers=_auth_headers(ctx.api_key, json_body=True),
                json={"voice_type": voice_type},
            )
    except Exception as exc:
        return (f"Error: get_voice request failed: {exc}", [])
    if r.status_code >= 400:
        return (f"Error: get_voice {r.status_code}: {r.text[:400]}", [])
    try:
        body = r.json()
    except Exception as exc:
        return (f"Error: unparseable get_voice response: {exc}", [])
    ok, msg = _base_resp_ok(body)
    if not ok:
        return (f"Error: get_voice rejected: {msg or body}", [])
    return (
        _format_voice_list(body, str(language) if language else None),
        [],
    )


async def _delete_voice(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    voice_id = str(args.get("voice_id", "")).strip()
    voice_type = str(args.get("voice_type", "")).strip().lower()
    if not voice_id:
        return ("Error: voice_id is required", [])
    if voice_type not in ("voice_cloning", "voice_generation"):
        return (
            "Error: voice_type must be 'voice_cloning' or 'voice_generation'. "
            "System voices cannot be deleted.",
            [],
        )
    url = f"{ctx.api_base}/v1/delete_voice"
    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                url,
                headers=_auth_headers(ctx.api_key, json_body=True),
                json={"voice_id": voice_id, "voice_type": voice_type},
            )
    except Exception as exc:
        return (f"Error: delete_voice request failed: {exc}", [])
    if r.status_code >= 400:
        return (f"Error: delete_voice {r.status_code}: {r.text[:400]}", [])
    try:
        body = r.json()
    except Exception as exc:
        return (f"Error: unparseable delete_voice response: {exc}", [])
    ok, msg = _base_resp_ok(body)
    if not ok:
        return (f"Error: delete_voice rejected: {msg or body}", [])
    return (f"Deleted voice_id='{voice_id}' ({voice_type}).", [])


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
            "Error: voice_id is required. Call list_voices first or pass "
            "a voice_id returned from clone_voice.",
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

    # Record the new voice_id so the agent can surface it in the final
    # summary (the API won't list it via /v1/get_voice until it's been
    # used at least once, so the caller's only record is what we return).
    if voice_id not in ctx.created_voice_ids:
        ctx.created_voice_ids.append(voice_id)

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


# ---------------------------------------------------------------------------
# generate_music
# ---------------------------------------------------------------------------


_MUSIC_REF_EXTS: set[str] = {".mp3", ".wav", ".flac", ".m4a"}
_MUSIC_REF_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB per MiniMax docs


def _resolve_music_reference(
    reference: str,
    ctx: ToolContext,
) -> tuple[str, str]:
    """Resolve ``reference_audio`` → the MiniMax request field name and value.

    Returns ``("audio_url", url)`` or ``("audio_base64", b64)``.  Raises
    ValueError on invalid / missing references.
    """
    ref = reference.strip()
    if not ref:
        raise ValueError("reference_audio is empty")
    if ref.startswith(("http://", "https://")):
        return ("audio_url", ref)
    if ref.startswith(("/", "\\")) or ":" in ref[:3]:
        raise ValueError(
            "reference_audio must be a filename from Reference Files, "
            "or an http(s) URL — absolute paths are not allowed"
        )

    local_path_str: Optional[str] = None
    entry = ctx.ref_map.get(ref)
    if entry:
        lp = entry.get("local_path", "")
        if lp and Path(lp).is_file():
            local_path_str = lp
    if local_path_str is None and ctx.inbox_resolver is not None:
        local_path_str = ctx.inbox_resolver(ref)
    if not local_path_str:
        raise ValueError(f"reference_audio '{ref}' not found in inbox")

    p = Path(local_path_str)
    if p.suffix.lower() not in _MUSIC_REF_EXTS:
        raise ValueError(
            f"unsupported reference_audio format '{p.suffix}'. "
            "MiniMax accepts mp3, wav, flac, m4a."
        )
    size = p.stat().st_size
    if size > _MUSIC_REF_MAX_BYTES:
        raise ValueError(
            f"reference_audio is {size} bytes; MiniMax limit is 50 MB"
        )
    return ("audio_base64", base64.b64encode(p.read_bytes()).decode("ascii"))


async def _generate_music(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    model = str(args.get("model") or ctx.music_model).strip() or ctx.music_model
    prompt = str(args.get("prompt") or "").strip()
    lyrics = str(args.get("lyrics") or "").strip()
    is_instrumental = bool(args.get("is_instrumental"))
    lyrics_optimizer = bool(args.get("lyrics_optimizer"))
    fmt = str(args.get("format") or "mp3").lower()
    if fmt not in ("mp3", "wav", "pcm"):
        fmt = "mp3"
    reference = args.get("reference_audio")

    is_cover = model.startswith("music-cover")

    # --- Pre-flight validation (catch the most common misconfigurations
    # before we send a billed request). ---
    if is_cover:
        if not prompt:
            return (
                "Error: `prompt` is required for music-cover models "
                "(10-300 chars describing the target cover style).",
                [],
            )
        if not reference:
            return (
                "Error: `reference_audio` is required for music-cover "
                "models.",
                [],
            )
        if is_instrumental or lyrics_optimizer:
            return (
                "Error: `is_instrumental` and `lyrics_optimizer` are only "
                "valid with music-2.6 / music-2.6-free, not music-cover.",
                [],
            )
    else:
        if is_instrumental and not prompt:
            return (
                "Error: `prompt` is required when `is_instrumental` is true.",
                [],
            )
        if (
            not is_instrumental
            and not lyrics
            and not lyrics_optimizer
        ):
            return (
                "Error: vocal songs need `lyrics`, or set "
                "`lyrics_optimizer: true` to auto-generate them from "
                "`prompt`, or set `is_instrumental: true` for a vocal-less "
                "track.",
                [],
            )

    payload: dict[str, Any] = {
        "model": model,
        "output_format": "hex",
        "audio_setting": {"format": fmt},
    }
    if prompt:
        payload["prompt"] = prompt
    if lyrics:
        payload["lyrics"] = lyrics
    if is_instrumental:
        payload["is_instrumental"] = True
    if lyrics_optimizer:
        payload["lyrics_optimizer"] = True
    if is_cover and reference:
        try:
            field_name, value = _resolve_music_reference(str(reference), ctx)
        except ValueError as exc:
            return (f"Error: {exc}", [])
        payload[field_name] = value

    url = f"{ctx.api_base}/v1/music_generation"
    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                url,
                headers=_auth_headers(ctx.api_key, json_body=True),
                json=payload,
            )
    except Exception as exc:
        return (f"Error: music_generation request failed: {exc}", [])
    if r.status_code >= 400:
        return (f"Error: music_generation {r.status_code}: {r.text[:400]}", [])
    try:
        body = r.json()
    except Exception as exc:
        return (f"Error: unparseable music_generation response: {exc}", [])
    ok, msg = _base_resp_ok(body)
    if not ok:
        return (f"Error: music_generation rejected: {msg or body}", [])

    data = body.get("data") or {}
    status = data.get("status")
    if status != 2:
        return (
            f"Error: music generation did not complete (status={status}).",
            [],
        )
    audio_hex = data.get("audio")
    if not audio_hex:
        return ("Error: music_generation returned no audio payload.", [])
    try:
        audio_bytes = bytes.fromhex(audio_hex)
    except Exception as exc:
        return (f"Error: could not decode audio hex: {exc}", [])

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"minimax_music_{uuid.uuid4().hex[:10]}.{fmt}"
    out_path = ctx.output_dir / fname
    out_path.write_bytes(audio_bytes)
    pf_dict = ctx.pfm.resolve(str(out_path.resolve()))
    if pf_dict is None:
        return ("Error: failed to register generated music as a ProxyFile.", [])

    extra = body.get("extra_info") or {}
    duration_ms = extra.get("music_duration")
    dur_str = ""
    if isinstance(duration_ms, (int, float)) and duration_ms > 0:
        dur_str = f", {round(duration_ms / 1000, 1)}s"

    kind = (
        "cover" if is_cover
        else "instrumental" if is_instrumental
        else "song"
    )
    summary = (
        f"Generated {kind} with {model} ({len(audio_bytes)} bytes"
        f"{dur_str}): {fname}"
    )
    return (summary, [ProxyFile(**pf_dict)])


# ---------------------------------------------------------------------------
# generate_video
# ---------------------------------------------------------------------------


# Mode-specific default models for the three modes that don't use the
# configured VIDEO_MODEL (which covers text-to-video + image-to-video).
_VIDEO_MODE_DEFAULT_MODELS: dict[str, str] = {
    "first_last": "MiniMax-Hailuo-02",
    "subject":    "S2V-01",
    "default":    "MiniMax-Hailuo-2.3",  # text + image-to-video
}


async def _generate_video(
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[str, list[ProxyFile]]:
    prompt = str(args.get("prompt", "")).strip()
    if not prompt:
        return ("Error: prompt is required", [])

    first_frame = args.get("first_frame_image")
    last_frame = args.get("last_frame_image")
    subject_ref = args.get("subject_reference")
    has_first = bool(first_frame)
    has_last = bool(last_frame)
    has_subject = bool(subject_ref)

    # --- Mode validation ---
    if has_subject and (has_first or has_last):
        return (
            "Error: subject_reference cannot be combined with "
            "first_frame_image / last_frame_image.",
            [],
        )
    if has_last and not has_first:
        return (
            "Error: last_frame_image requires first_frame_image (first-"
            "last-frame mode).",
            [],
        )

    # Resolve model: explicit override > per-mode default.
    model_override = str(args.get("model") or "").strip()
    if model_override:
        model = model_override
    else:
        if has_subject:
            model = _VIDEO_MODE_DEFAULT_MODELS["subject"]
        elif has_last:
            model = _VIDEO_MODE_DEFAULT_MODELS["first_last"]
        else:
            model = ctx.video_model or _VIDEO_MODE_DEFAULT_MODELS["default"]

    try:
        duration = int(args.get("duration") or 6)
    except (TypeError, ValueError):
        duration = 6
    resolution = str(args.get("resolution") or "1080P").strip() or "1080P"

    payload: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "duration": duration,
        "resolution": resolution,
    }

    if has_first:
        try:
            payload["first_frame_image"] = _ref_to_image_file(
                str(first_frame), ctx,
            )
        except Exception as exc:
            return (f"Error: invalid first_frame_image: {exc}", [])
    if has_last:
        try:
            payload["last_frame_image"] = _ref_to_image_file(
                str(last_frame), ctx,
            )
        except Exception as exc:
            return (f"Error: invalid last_frame_image: {exc}", [])
    if has_subject:
        try:
            subj_value = _ref_to_image_file(str(subject_ref), ctx)
        except Exception as exc:
            return (f"Error: invalid subject_reference: {exc}", [])
        payload["subject_reference"] = [
            {"type": "character", "image": [subj_value]},
        ]

    # --- 1. Create task ---
    create_url = f"{ctx.api_base}/v1/video_generation"
    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            r = await client.post(
                create_url,
                headers=_auth_headers(ctx.api_key, json_body=True),
                json=payload,
            )
    except Exception as exc:
        return (f"Error: video_generation request failed: {exc}", [])
    if r.status_code >= 400:
        return (f"Error: video_generation {r.status_code}: {r.text[:400]}", [])
    try:
        create_body = r.json()
    except Exception as exc:
        return (f"Error: unparseable video_generation response: {exc}", [])
    # video_generation's response does not always include base_resp, but
    # when it does we surface the error.  task_id is the authoritative
    # success signal.
    task_id = create_body.get("task_id")
    if not task_id:
        ok, msg = _base_resp_ok(create_body)
        return (
            f"Error: video_generation returned no task_id "
            f"({msg or create_body})",
            [],
        )

    # --- 2. Poll ---
    try:
        file_id = await _mm_poll_video_task(ctx, str(task_id))
    except Exception as exc:
        return (f"Error: video task failed: {exc}", [])

    # --- 3. Retrieve download URL + download the bytes ---
    try:
        download_url = await _mm_get_file_download_url(ctx, file_id)
    except Exception as exc:
        return (f"Error: could not retrieve video download URL: {exc}", [])

    try:
        async with httpx.AsyncClient(timeout=ctx.http_timeout) as client:
            vr = await client.get(download_url)
            vr.raise_for_status()
            video_bytes = vr.content
    except Exception as exc:
        return (f"Error: video download failed: {exc}", [])
    if not video_bytes:
        return ("Error: downloaded video was empty.", [])

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"minimax_video_{uuid.uuid4().hex[:10]}.mp4"
    out_path = ctx.output_dir / fname
    out_path.write_bytes(video_bytes)
    pf_dict = ctx.pfm.resolve(str(out_path.resolve()))
    if pf_dict is None:
        return ("Error: failed to register generated video as a ProxyFile.", [])

    # Mode label for the LLM-facing summary.
    if has_subject:
        kind = "subject-reference video"
    elif has_last:
        kind = "first-last-frame video"
    elif has_first:
        kind = "image-to-video"
    else:
        kind = "text-to-video"
    summary = (
        f"Generated {kind} with {model} at {resolution}, {duration}s "
        f"({len(video_bytes)} bytes): {fname}"
    )
    return (summary, [ProxyFile(**pf_dict)])
