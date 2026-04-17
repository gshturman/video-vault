# video-vault

Local pipeline to process saved TikTok videos into searchable Obsidian markdown.

## Stack

- faster-whisper (large-v3) on CUDA for transcription
- Qwen2.5:14b via Ollama for summarization
- Python 3.10, no cloud APIs

## Runs on

Windows 11, RTX 4090 Laptop (16GB VRAM). Sequential VRAM usage: unload whisper, then run Qwen.

## Input

`D:\videos\raw\tiktok\` (.mp4 + matching .info.json from yt-dlp)

## Output

`D:\Dev\second-brain\raw\tiktok\` (one markdown per video, YAML frontmatter + summary + full transcript)

## Setup

Python 3.10 venv at `.venv` with faster-whisper, ollama, python-frontmatter, pyyaml, nvidia-cudnn-cu12.
Ollama running locally at `localhost:11434` with `qwen2.5:14b` pulled.
