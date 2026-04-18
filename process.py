"""Local TikTok to Obsidian markdown pipeline.

Transcribes videos with faster-whisper (CUDA) and summarizes with
Qwen2.5:14b via Ollama. One markdown file per video, flat output.
Idempotent, resumable, memory-aware.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

# Register NVIDIA DLL directories before importing faster_whisper (Windows).
# CTranslate2 needs cublas and cudnn DLLs on the DLL search path.
import os
import sys
if sys.platform == "win32":
    import site
    _dll_dirs = []
    for _site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        _nvidia = Path(_site_dir) / "nvidia"
        if not _nvidia.is_dir():
            continue
        for _sub in ("cublas", "cudnn", "cuda_nvrtc"):
            _bin = _nvidia / _sub / "bin"
            if _bin.is_dir():
                _dll_dirs.append(str(_bin))
                os.add_dll_directory(str(_bin))
    if _dll_dirs:
        os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

import yaml
from faster_whisper import WhisperModel
from ollama import Client as OllamaClient
from tqdm import tqdm

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# --- Constants ---------------------------------------------------------------

INPUT_DIR = Path(r"D:\videos\raw\tiktok")
OUTPUT_DIR = Path(r"D:\Dev\second-brain\raw\tiktok")
PROCESSED_DIR = Path(r"D:\videos\processed\tiktok")
FAILURE_LOG = Path(r"D:\Dev\video-vault\failures.log")
EMPTY_LOG = Path(r"D:\Dev\video-vault\empty_videos.log")
PROCESSED_LOG = Path(r"D:\Dev\video-vault\processed.log")
QWEN_MODEL = "qwen2.5:14b"
COMMIT_CHUNK = 500
_SLUG_INVALID = re.compile(r"[^a-z0-9\-]")
_SLUG_COLLAPSE = re.compile(r"-+")

_INVALID_PATH_CHARS = re.compile(r'[\\/:*?"<>|]')

QWEN_SYSTEM = (
    "You extract minimal metadata from short-video transcripts. "
    "You respond with a single JSON object matching the schema you are given. "
    "No prose, no markdown, no keys outside the schema."
)

QWEN_PROMPT_TEMPLATE = """\
A short video has been transcribed. Extract minimal metadata.

Poster title: {title}
Poster description: {description}

Transcript:
\"\"\"
{transcript}
\"\"\"

Return a single JSON object with EXACTLY these keys and no others:

{{
  "one_line_summary": "One sentence, under 25 words, describing what the video is about. Use the same language as the transcript.",
  "key_points": ["2 to 4 short bullet strings covering the substantive content. Empty array only if the transcript truly has no speech content."],
  "slug": "A short English kebab-case slug describing the video's topic. 3 to 6 words. Lowercase. ASCII only. Hyphens between words. No punctuation, no emoji, no accents. Example: 'face-recognition-privacy-risk' or 'alinti-plant-electricity-peru'. Even if the transcript is Spanish, the slug is English."
}}

Rules:
- Base answers on the transcript. Use title and description only to disambiguate unclear references.
- Do not invent facts not present in the transcript.
- If the transcript is empty or has no speech: one_line_summary="(no speech detected)", key_points=[], slug="no-speech-detected".
- The slug should describe the CONTENT, not the creator. Ignore clickbait prefixes in the title.
"""


# --- Logging -----------------------------------------------------------------

def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_failure(file: str, stage: str, error: str, dry_run: bool = False) -> None:
    payload = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "file": file,
        "stage": stage,
        "error": error,
    }
    if dry_run:
        print(f"[dry-run] would log failure: {payload}", file=sys.stderr)
        return
    _append_jsonl(FAILURE_LOG, payload)


def log_empty(file: str, whisper_lang: str, duration_seconds: int, dry_run: bool = False) -> None:
    payload = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "file": file,
        "whisper_lang": whisper_lang,
        "duration_seconds": duration_seconds,
    }
    if dry_run:
        print(f"[dry-run] would log empty: {payload}", file=sys.stderr)
        return
    _append_jsonl(EMPTY_LOG, payload)


def read_empty_set(path: Path) -> set:
    if not path.exists():
        return set()
    names = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                names.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                continue
    return names


def read_processed_set(path: Path) -> set:
    """Read processed.log as a set of TikTok video IDs."""
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def log_processed(video_id: str, slug: str, filename: str, dry_run: bool = False) -> None:
    payload = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "id": video_id,
        "slug": slug,
        "filename": filename,
    }
    if dry_run:
        print(f"[dry-run] would log processed: {payload}", file=sys.stderr)
        return
    _append_jsonl(PROCESSED_LOG, payload)


# --- Pure helpers ------------------------------------------------------------

def load_info_json(video_path: Path) -> dict:
    info_path = video_path.with_suffix(".info.json")
    with info_path.open(encoding="utf-8") as f:
        return json.load(f)


def sanitize_slug(s: str) -> str:
    """Normalize a Qwen-generated slug to safe kebab-case ASCII."""
    if not s:
        return "untitled"
    cleaned = s.strip().lower()
    cleaned = _SLUG_INVALID.sub("-", cleaned)
    cleaned = _SLUG_COLLAPSE.sub("-", cleaned).strip("-")
    cleaned = cleaned[:50].rstrip("-")
    return cleaned or "untitled"


def parse_upload_date(yyyymmdd: str) -> str:
    if yyyymmdd and len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return dt.date.today().isoformat()


def _read_source_url(md_path: Path):
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("source_url")
    return value if isinstance(value, str) else None


def compute_output_path(slug: str, output_dir: Path) -> Path:
    """Return the target output path for a given slug.

    Collision handling: if the slug is already in use by a different video,
    append a 2-char hash. Callers must ensure the video isn't already
    processed (via the processed log) before calling this.
    """
    safe = sanitize_slug(slug)
    return output_dir / f"{safe}.md"


# --- Rendering ---------------------------------------------------------------

def _pick_title(info: dict, transcript: str) -> str:
    title = (info.get("title") or "").strip()
    if title:
        return title
    if transcript.strip():
        return transcript.strip()[:80]
    uploader = info.get("uploader") or "unknown"
    short_id = str(info.get("id") or "00000000")[:8]
    return f"(untitled) {uploader}_{short_id}"


def render_frontmatter(info: dict, whisper_lang: str, title: str) -> str:
    data = {
        "title": title,
        "date": parse_upload_date(info.get("upload_date", "")),
        "source": "tiktok",
        "source_url": info.get("webpage_url", ""),
        "uploader": info.get("uploader", ""),
        "channel": info.get("channel", ""),
        "duration": int(info.get("duration") or 0),
        "language": whisper_lang or "en",
        "tags": ["video", "tiktok"],
    }
    dumped = yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return f"---\n{dumped}---\n"


def render_markdown(info: dict, qwen: dict, transcript: str, whisper_lang: str) -> str:
    title = _pick_title(info, transcript)
    frontmatter = render_frontmatter(info, whisper_lang, title)
    summary = (qwen.get("one_line_summary") or "").strip() or "(no summary)"
    key_points = qwen.get("key_points") or []
    key_points_md = "\n".join(f"- {p}" for p in key_points) if key_points else "- (none)"
    return (
        f"{frontmatter}\n"
        f"# {title}\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## Key Points\n\n{key_points_md}\n\n"
        f"## Transcript\n\n{transcript}\n"
    )


# --- Stages ------------------------------------------------------------------

def load_whisper(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cuda", compute_type="float16")


def _release_whisper(model) -> None:
    if model is not None:
        del model
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe(model: WhisperModel, video_path: Path):
    segments, info_obj = model.transcribe(
        str(video_path),
        language=None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )
    text = "\n".join(s.text.strip() for s in segments).strip()
    return text, info_obj.language or "en"


def _extract_content(resp) -> str:
    try:
        return resp["message"]["content"]
    except (TypeError, KeyError):
        pass
    message = getattr(resp, "message", None)
    if message is None:
        raise RuntimeError("ollama response missing 'message'")
    content = getattr(message, "content", None)
    if content is None:
        raise RuntimeError("ollama response message missing 'content'")
    return content


def summarize(client: OllamaClient, transcript: str, title: str, description: str) -> dict:
    user_msg = QWEN_PROMPT_TEMPLATE.format(
        title=title or "(empty)",
        description=description or "(empty)",
        transcript=transcript,
    )
    messages = [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    def _call(temperature: float) -> dict:
        resp = client.chat(
            model=QWEN_MODEL,
            format="json",
            messages=messages,
            options={"temperature": temperature, "num_ctx": 4096},
        )
        return json.loads(_extract_content(resp))

    try:
        return _call(0.0)
    except (json.JSONDecodeError, ValueError):
        return _call(0.0)


# --- Orchestrator ------------------------------------------------------------

def process_one(
    video_path: Path,
    output_dir: Path,
    whisper_model: WhisperModel,
    ollama_client: OllamaClient,
    dry_run: bool,
    sequential_vram: bool,
    model_size: str,
):
    """Process a single video. Returns (status, whisper_model).

    status in {"written", "skipped", "empty", "failed"}.
    """
    filename = video_path.name

    try:
        info = load_info_json(video_path)
    except Exception as e:
        log_failure(filename, "load_info", repr(e), dry_run=dry_run)
        return "failed", whisper_model

    try:
        transcript, whisper_lang = transcribe(whisper_model, video_path)
    except Exception as e:
        log_failure(filename, "transcribe", repr(e), dry_run=dry_run)
        return "failed", whisper_model

    if not transcript.strip():
        log_empty(filename, whisper_lang, int(info.get("duration") or 0), dry_run=dry_run)
        return "empty", whisper_model

    current_model = whisper_model
    if sequential_vram:
        _release_whisper(current_model)
        current_model = None

    try:
        qwen = summarize(
            ollama_client,
            transcript,
            info.get("title", ""),
            info.get("description", ""),
        )
    except Exception as e:
        log_failure(filename, "qwen_json", repr(e), dry_run=dry_run)
        if sequential_vram:
            current_model = load_whisper(model_size)
        return "failed", current_model

    if sequential_vram:
        current_model = load_whisper(model_size)

    # Qwen signals "no real content" via the slug or an empty summary.
    # Treat as empty: skip writing, log, move video.
    raw_slug = (qwen.get("slug") or "").strip().lower()
    summary_text = (qwen.get("one_line_summary") or "").strip().lower()
    if (
        raw_slug == "no-speech-detected"
        or "no speech detected" in summary_text
        or not qwen.get("one_line_summary")
    ):
        log_empty(filename, whisper_lang, int(info.get("duration") or 0), dry_run=dry_run)
        if not dry_run:
            try:
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                video_path.replace(PROCESSED_DIR / video_path.name)
                info_json = video_path.with_suffix(".info.json")
                if info_json.exists():
                    info_json.replace(PROCESSED_DIR / info_json.name)
            except Exception as e:
                log_failure(filename, "move", repr(e), dry_run=dry_run)
        return "empty", current_model

    slug = sanitize_slug(raw_slug)
    out_path = compute_output_path(slug, output_dir)

    # Collision handling: if slug already exists on disk from a DIFFERENT video,
    # append 2-char hash of this video's TikTok id.
    if out_path.exists():
        video_id = str(info.get("id", ""))
        if video_id:
            short_hash = hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:2]
            out_path = output_dir / f"{slug}-{short_hash}.md"

    markdown = render_markdown(info, qwen, transcript, whisper_lang)

    if dry_run:
        print(f"\n===== {out_path.name} =====\n{markdown}")
        return "written", current_model

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    except Exception as e:
        log_failure(filename, "write", repr(e), dry_run=dry_run)
        return "failed", current_model

    # Record success and move the source video out of the input folder.
    video_id = str(info.get("id", ""))
    log_processed(video_id, slug, out_path.name, dry_run=dry_run)

    if not dry_run:
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            dest = PROCESSED_DIR / video_path.name
            video_path.replace(dest)
            info_json = video_path.with_suffix(".info.json")
            if info_json.exists():
                info_json.replace(PROCESSED_DIR / info_json.name)
        except Exception as e:
            log_failure(filename, "move", repr(e), dry_run=dry_run)
            # Non-fatal: file was written successfully.

    return "written", current_model


def _resolve_videos(input_dir: Path, files_arg, limit, dry_run: bool):
    if files_arg:
        names = [n.strip() for n in files_arg.split(",") if n.strip()]
        return [input_dir / n for n in names]
    videos = sorted(input_dir.glob("*.mp4"))
    if dry_run and limit is None:
        limit = 10
    if limit is not None:
        videos = videos[:limit]
    return videos


def main() -> int:
    parser = argparse.ArgumentParser(description="TikTok to Obsidian markdown pipeline.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N videos (alphabetical order).")
    parser.add_argument("--files", type=str, default=None,
                        help="Comma-separated filenames under --input-dir. Overrides --limit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rendered markdown to stdout. No file or log writes.")
    parser.add_argument("--model-size", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--sequential-vram", action="store_true",
                        help="Unload whisper before each Qwen call (OOM fallback).")
    args = parser.parse_args()

    videos = _resolve_videos(args.input_dir, args.files, args.limit, args.dry_run)
    missing = [v for v in videos if not v.exists()]
    if missing:
        names = [v.name for v in missing[:5]]
        print(f"error: {len(missing)} video(s) not found: {names}", file=sys.stderr)
        return 2

    empty_set = read_empty_set(EMPTY_LOG)
    processed_ids = read_processed_set(PROCESSED_LOG)

    filtered = []
    skipped_empty = 0
    skipped_processed = 0
    for v in videos:
        if v.name in empty_set:
            skipped_empty += 1
            continue
        # Cheap ID extraction: filename is {uploader}_{id}.mp4
        stem = v.stem
        if "_" in stem:
            candidate_id = stem.rsplit("_", 1)[-1]
            if candidate_id in processed_ids:
                skipped_processed += 1
                continue
        filtered.append(v)

    if skipped_empty:
        print(f"info: {skipped_empty} video(s) in empty_videos.log skipped "
              f"(delete log to force re-process)")
    if skipped_processed:
        print(f"info: {skipped_processed} video(s) already in processed.log skipped "
              f"(delete log to force re-process)")
    videos = filtered

    if not videos:
        print("nothing to do")
        return 0

    print(
        f"processing {len(videos)} videos "
        f"(dry_run={args.dry_run}, model={args.model_size}, "
        f"sequential_vram={args.sequential_vram})"
    )

    ollama_client = OllamaClient()
    try:
        ollama_client.list()
    except Exception as e:
        print(f"error: cannot reach Ollama at localhost:11434: {e}", file=sys.stderr)
        return 1

    whisper_model = load_whisper(args.model_size)
    counters = {"written": 0, "skipped": 0, "empty": 0, "failed": 0}
    written = 0

    try:
        for video in tqdm(videos, desc="videos", unit="vid"):
            status, whisper_model = process_one(
                video,
                args.output_dir,
                whisper_model,
                ollama_client,
                args.dry_run,
                args.sequential_vram,
                args.model_size,
            )
            counters[status] += 1
            if status == "written" and not args.dry_run:
                written += 1
                if written % COMMIT_CHUNK == 0:
                    batch = written // COMMIT_CHUNK
                    tqdm.write(
                        f"\n[commit reminder] {written} new files written.\n"
                        f"  cd D:\\Dev\\second-brain && git add raw/tiktok && "
                        f'git commit -m "pipeline: batch {batch}"'
                    )
    finally:
        _release_whisper(whisper_model)

    print(f"done: {counters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())