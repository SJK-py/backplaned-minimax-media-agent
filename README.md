# minimax_media_agent

A drop-in embedded agent for [Backplaned](https://github.com/SJK-py/backplaned) that wraps MiniMax's full media-generation suite behind a single LLM-driven tool loop.

One agent, one tool-loop planner (MiniMax's Anthropic-compatible endpoint), seven tools covering image, speech, music, and video. Accepts the usual `llmdata` / `files` payload, returns generated media as `ProxyFile` attachments.

No new dependencies — only `httpx` and `fastapi`, both already required by Backplaned itself.

## Capabilities

| Tool | MiniMax endpoint | Modes |
|---|---|---|
| `generate_image` | `POST /v1/image_generation` | text-to-image, image-to-image (via `reference_image`) |
| `list_voices` | `POST /v1/get_voice` | `system` / `voice_cloning` / `voice_generation` / `all`, with optional language-substring filter |
| `delete_voice` | `POST /v1/delete_voice` | `voice_cloning` / `voice_generation` (system voices are not deletable) |
| `generate_speech` | `POST /v1/t2a_async_v2` (+ poll + retrieve) | async long-form T2A; up to 1M characters of input |
| `clone_voice` | `POST /v1/files/upload` + `POST /v1/voice_clone` | source-audio-only cloning; optional preview synthesis |
| `generate_music` | `POST /v1/music_generation` | vocal song, instrumental, cover (via `reference_audio`) |
| `generate_video` | `POST /v1/video_generation` (+ poll + retrieve) | text-to-video, image-to-video, first-last-frame, subject-reference |

## Installation

Drop the `minimax_media_agent/` directory into Backplaned's `agents/` folder:

```bash
cp -r minimax_media_agent /path/to/backplaned/agents/
```

The router auto-registers any subdirectory of `agents/` that contains an `agent.py` with an `app: FastAPI`. No further wiring needed.

### Bare-metal

```bash
./start.sh   # or just restart the router process
```

### Docker

Backplaned's `docker/docker-compose.yml` declares one bind mount per embedded agent so that each agent's `data/` directory (config + inboxes + output) survives container rebuilds. Add a matching line for `minimax_media_agent` alongside the existing `md_converter` / `web_agent` / etc. entries in the `router` service's `volumes:` block:

```yaml
services:
  router:
    volumes:
      # ... existing bind mounts for router, core_personal_agent, llm_agent, ...
      - "${DATA_ROOT:-./data}/minimax_media_agent:/app/agents/minimax_media_agent/data"
```

Then rebuild + restart:

```bash
cd /path/to/backplaned/docker
docker compose up -d --build router
```

On first boot the `data/` directory under `DATA_ROOT` is empty, so the agent will run with pure code defaults. Set `MINIMAX_API_KEY` (and optionally `MINIMAX_BACKUP_API_KEY`) + populate `ALLOWED_USER_IDS` either through the web-admin config editor or by editing `${DATA_ROOT}/minimax_media_agent/config.json` directly.

Without the bind mount, the config written through the web-admin UI (including your API key) is lost on container rebuild.

### Agent registration

The agent registers with:

- **Group:** `usertool` (inbound + outbound)
- **Input schema:** `llmdata: LLMData, files: Optional[List[ProxyFile]], user_id: str`
- **Output schema:** `content: str, files: Optional[List[ProxyFile]]`
- **Required:** `llmdata`, `user_id`

## Configuration

Edit `agents/minimax_media_agent/data/config.json` (or via the web-admin UI's config editor; the editor reads field descriptions from `config.example`).

| Key | Default | Description |
|---|---|---|
| `MINIMAX_API_KEY` | `""` | MiniMax Platform API key. **Required.** Get one from <https://platform.minimax.io/user-center/basic-information/interface-key>. |
| `MINIMAX_BACKUP_API_KEY` | `""` | Optional secondary API key used as fallback when the main key fails with a tier/permission/auth/balance error. See **Key fallback** below. |
| `MINIMAX_API_BASE` | `https://api.minimax.io` | Global endpoint (use `https://api.minimaxi.com` for Mainland China). |
| `LLM_MODEL` | `MiniMax-M2.7` | Model driving the tool-loop planner. Options: `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`, `MiniMax-M2.1`, `MiniMax-M2.1-highspeed`, `MiniMax-M2`. |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM turn. |
| `LLM_TEMPERATURE` | `1.0` | Sampling temperature, range (0, 1]. |
| `IMAGE_MODEL` | `image-01` | Image-generation model id. |
| `SPEECH_MODEL` | `speech-2.8-hd` | Default speech model. `*-hd` favours fidelity, `*-turbo` favours speed/cost. LLM may override per call. |
| `SPEECH_POLL_INTERVAL` | `3` | Seconds between async T2A status polls. |
| `SPEECH_MAX_WAIT` | `600` | Max seconds to wait for T2A completion. |
| `MUSIC_MODEL` | `music-2.6` | Default music model. Use `music-2.6-free` for free-tier API keys; `music-cover[-free]` for covers. |
| `MUSIC_MAX_WAIT` | `300` | Max seconds for a single (synchronous) music generation HTTP call. |
| `VIDEO_MODEL` | `MiniMax-Hailuo-2.3` | Default text/image-to-video model. First-last-frame auto-switches to `MiniMax-Hailuo-02`; subject-reference to `S2V-01`. |
| `VIDEO_POLL_INTERVAL` | `10` | Seconds between video-task status polls. |
| `VIDEO_MAX_WAIT` | `1800` | Max seconds to wait for video completion. Videos take minutes. |
| `OUTPUT_DIR` | `""` | Where to write generated media. Blank = `agents/minimax_media_agent/data/output`. |
| `AGENT_TIMEOUT` | `180` | Per-LLM-call timeout. Tool execution has its own budgets (`HTTP_TIMEOUT`, `SPEECH_MAX_WAIT`, `VIDEO_MAX_WAIT`). |
| `HTTP_TIMEOUT` | `120` | Per outbound MiniMax HTTP call. |
| `MAX_ITERATIONS` | `6` | Max LLM tool-loop iterations. |
| `MAX_TOOL_CALLS` | `8` | Max total tool calls across the loop. |
| `ALLOWED_USER_IDS` | `[]` | Access allowlist. See **Access control** below. |

Config is re-read on every request, so edits via the web-admin UI take effect without restart.

## Timeout coordination with Backplaned

Media generation — especially video — can take longer than Backplaned's default per-layer timeouts. The full cascade from user request to this agent's generation call is:

| Layer | Setting | Backplaned default | Scope |
|---|---|---|---|
| Router HTTP-to-agent | `EMBEDDED_AGENT_TIMEOUT` (env var) | 300 s | Per agent call |
| Router whole task | `GLOBAL_TIMEOUT_HOURS` (env var) | 1 h | Whole task tree |
| core_personal_agent overall loop | `CORE_AGENT_TIMEOUT` (config.json) | 290 s | LLM loop |
| core_personal_agent per-tool | `CORE_TOOL_TIMEOUT` (config.json) | 240 s | Each sub-agent call |
| minimax_media_agent music | `MUSIC_MAX_WAIT` | 300 s | Synchronous music HTTP |
| minimax_media_agent video | `VIDEO_MAX_WAIT` | 1800 s | Async video poll cap |

**This agent will exceed the stock defaults for video every time, and can exceed them for music.** The innermost timeout always wins, so if you don't raise the upstream ones, the router / core agent will kill the request before MiniMax finishes the job.

### Suggested values

For a deployment that uses **music + video** end-to-end from a `core_personal_agent` chat:

```jsonc
// agents/core_personal_agent/data/config.json
{
  "CORE_TOOL_TIMEOUT": 1900,   // ≥ VIDEO_MAX_WAIT + small buffer
  "CORE_AGENT_TIMEOUT": 2000   // ≥ CORE_TOOL_TIMEOUT
}
```

Router `EMBEDDED_AGENT_TIMEOUT`:

- **Bare-metal:** set `EMBEDDED_AGENT_TIMEOUT=1900` in the router's environment (e.g. in `start.config` or a systemd unit) before starting.
- **Docker:** uncomment the optional line in `docker/docker-compose.yml` and raise the default:

  ```yaml
  services:
    router:
      environment:
        - EMBEDDED_AGENT_TIMEOUT=${EMBEDDED_AGENT_TIMEOUT:-1900}
  ```

`GLOBAL_TIMEOUT_HOURS` default (1 h) comfortably accommodates a single 30-minute video, so usually no change needed there.

### If you only use music

Bump to ~400 s everywhere instead (`MUSIC_MAX_WAIT` is 300, so 400 leaves headroom):

```jsonc
// agents/core_personal_agent/data/config.json
{ "CORE_TOOL_TIMEOUT": 400, "CORE_AGENT_TIMEOUT": 500 }
```

```yaml
# docker-compose.yml
- EMBEDDED_AGENT_TIMEOUT=${EMBEDDED_AGENT_TIMEOUT:-400}
```

### If the upstream timeouts are fixed

If you can't change core / router settings, cap this agent instead — set `VIDEO_MAX_WAIT` (and/or `MUSIC_MAX_WAIT`) **below** the smallest upstream timeout. The tool will error with a clear "timed out after Ns — raise MUSIC_MAX_WAIT / VIDEO_MAX_WAIT" message instead of the router cutting the connection mid-flight.

## Access control

MiniMax's media APIs are billed per call, so this agent is **deny-by-default**. A request is accepted only if `user_id` is present in the payload **and** matches an entry in `ALLOWED_USER_IDS`.

```json
{
  "ALLOWED_USER_IDS": ["alice", "bob"]
}
```

- Empty list → everyone is denied.
- `["*"]` → any caller that supplies a non-empty `user_id` is allowed.
- Any other list → exact-match allowlist.

`core_personal_agent` injects the session's `user_id` authoritatively whenever a destination agent's `input_schema` declares it, so normal conversation flow Just Works once the user is on the allowlist. Unknown callers get a `403` and a helpful error message telling the operator which file to edit.

Denied calls never reach the MiniMax API — the check fires before any outbound work.

## Key fallback

Some MiniMax APIs are not available to every account tier — a Token Plan key may reject a model that the pay-as-you-go tier happily accepts, and vice versa. Set `MINIMAX_BACKUP_API_KEY` alongside `MINIMAX_API_KEY` and the agent will automatically retry the full tool call with the backup key when the main-key error looks tier-related:

```json
{
  "MINIMAX_API_KEY": "sk-primary-token-plan-...",
  "MINIMAX_BACKUP_API_KEY": "sk-backup-payg-..."
}
```

Retry fires only on these signals in the error text (case-insensitive):

- `"not support"` (matches MiniMax's `"token plan not support model"`)
- `"token plan"` / `"pay-as-you-go"`
- `"permission"`
- `"insufficient balance"` (status 1008)
- `" 401"` / `" 403"` / `"authentication failed"` / `"invalid api key"`

Validation errors (HTTP 2013 "invalid input parameters") are **not** retried — no key change would fix them. If the backup key also fails retryably, the main-key error is returned with a note that the backup was tried, so the operator can see the primary cause.

## Tool details

### Reference files

Anything you drop into the payload's `files` array (`ProxyFile`) is fetched to a per-task inbox at `data/inboxes/<task_id>/` and surfaced to the LLM **by filename only** — the system prompt never exposes absolute paths.

The LLM then passes a filename (exactly as shown) via whichever tool parameter accepts a reference:

| Parameter | Tool | Notes |
|---|---|---|
| `reference_image` | `generate_image` | subject/character consistency |
| `source_audio` | `clone_voice` | mp3 / m4a / wav, 10 s – 5 min, ≤ 20 MB |
| `reference_audio` | `generate_music` (cover modes) | mp3 / wav / flac / m4a, 6 s – 6 min, ≤ 50 MB |
| `first_frame_image` | `generate_video` | triggers image-to-video mode |
| `last_frame_image` | `generate_video` | requires `first_frame_image`; triggers first-last-frame mode |
| `subject_reference` | `generate_video` | subject-reference mode |

For any of these, an http(s) URL is also accepted. Absolute filesystem paths are rejected.

### `generate_speech`

Uses the asynchronous long-form TTS pipeline (`/v1/t2a_async_v2` → poll → `/v1/files/retrieve_content`), so input up to 1,000,000 characters is supported.

Requires a `voice_id`. Use `list_voices(voice_type="system", language="english")` first to pick one, or pass a `voice_id` returned by `clone_voice`.

### `clone_voice` + `list_voices` — known quirk

A freshly cloned voice does **not** appear in `/v1/get_voice` responses until it has been used for synthesis at least once. The agent addresses this in two ways:

1. `clone_voice` returns the new `voice_id` in its result text so the LLM can pass it straight to `generate_speech`.
2. The agent's final response **always** appends a footer listing any voice_ids created during the request, so the user has a durable record even if the LLM forgets to mention it:

   ```
   **New voice IDs created in this request (pass to generate_speech to use, or delete_voice to remove):**
   - `user_custom_alice_01`
   ```

Only the source audio is sent to MiniMax's `/v1/voice_clone`; the optional `clone_prompt.prompt_audio` field is intentionally not populated, per agent spec.

### `generate_music`

One tool covers three modes, selected by `model`:

- **Vocal song** (`music-2.6[-free]`): needs `lyrics`, or set `lyrics_optimizer: true` to have MiniMax auto-write them from `prompt`.
- **Instrumental** (`music-2.6[-free]` + `is_instrumental: true`): `prompt` required, `lyrics` ignored.
- **Cover** (`music-cover[-free]`): `prompt` (10–300 chars describing the target style) + `reference_audio`. `lyrics` is optional — MiniMax ASRs them from the reference otherwise.

Pre-flight validation catches the common misconfigurations (cover without `reference_audio`, instrumental without `prompt`, vocal song without lyrics/optimizer, music-2.6-only flags on a cover model) **before** a billed request is sent.

### `generate_video`

Mode is auto-detected from which inputs you pass — you don't pick the mode name, you pick the inputs:

| Inputs | Mode | Default model |
|---|---|---|
| `prompt` only | text-to-video | `MiniMax-Hailuo-2.3` (from config) |
| `prompt + first_frame_image` | image-to-video | `MiniMax-Hailuo-2.3` |
| `prompt + first_frame_image + last_frame_image` | first-last-frame | `MiniMax-Hailuo-02` |
| `prompt + subject_reference` | subject-reference | `S2V-01` |

Typical wall-clock latency is 1–5 minutes. The agent polls every `VIDEO_POLL_INTERVAL` seconds (default 10) up to `VIDEO_MAX_WAIT` (default 1800 = 30 min), then fetches the mp4 via `/v1/files/retrieve` → the returned `download_url`.

## File lifecycle

| Location | Purpose | Cleanup |
|---|---|---|
| `data/inboxes/<task_id>/` | Per-task sandbox for incoming reference files | Throttled sweep evicts subdirs whose newest file is older than 6 h |
| `data/output/` | Generated media files | Throttled sweep deletes files older than 6 h |

Both sweeps run at most once every 5 minutes, piggy-backed on the `/receive` handler. Using **newest-mtime-per-subdir** rather than the directory's own mtime means long-running tasks still writing into an existing inbox aren't evicted mid-run.

Generated media survives the agent's response because the router reads the `localfile` ProxyFile after the request completes — immediate deletion would race the router.

## Files in this repo

```
minimax_media_agent/
├── agent.py             # FastAPI app + LLM tool loop
├── tools.py             # 7 tool schemas + executors + MiniMax HTTP helpers
├── config.default.json  # Seed config
└── config.example       # Field-description map consumed by the web-admin UI
```

## Related

- [Backplaned repo](https://github.com/SJK-py/backplaned)
- [Backplaned agent reference](https://github.com/SJK-py/backplaned/blob/main/docs/agents.md)
- [Backplaned agent development guide](https://github.com/SJK-py/backplaned/blob/main/docs/agent-development.md)
- [MiniMax Platform docs](https://platform.minimax.io/docs/)

## License

MIT