"""Locate external tools (ffmpeg, Chromium, git) with documented fallbacks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


def probe_video(path: Path) -> dict:
    """Duration (s), width, height, codec and audio presence of a video file.

    Uses ffprobe when present, otherwise parses `ffmpeg -i` output (the
    imageio-ffmpeg bundle ships without ffprobe).
    """
    probe = ffprobe_exe()
    if probe:
        out = subprocess.run(
            [probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        info = json.loads(out)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        return {
            "duration_s": float(info["format"]["duration"]),
            "width": int(v["width"]),
            "height": int(v["height"]),
            "codec": v["codec_name"],
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
            "size_mb": round(path.stat().st_size / 1e6, 2),
        }
    ff = ffmpeg_exe()
    if ff is None:
        raise RuntimeError("ffmpeg not found")
    res = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    text = res.stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text)
    vid = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", text)
    if not dur or not vid:
        raise RuntimeError(f"could not parse ffmpeg output for {path}")
    h, m, s = dur.groups()
    return {
        "duration_s": int(h) * 3600 + int(m) * 60 + float(s),
        "width": int(vid.group(2)),
        "height": int(vid.group(3)),
        "codec": vid.group(1),
        "has_audio": "Audio:" in text,
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }


def chromium_executable() -> str | None:
    """Playwright's Chromium, or an existing system Chromium as a fallback."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            exe = p.chromium.executable_path
            if exe and Path(exe).exists():
                return None  # Playwright-managed browser available; no override needed
    except Exception:
        pass
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        exe = shutil.which(name)
        if exe:
            return exe
    pw_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if pw_path:
        for cand in Path(pw_path).glob("chromium-*/chrome-*/chrome*"):
            if cand.is_file():
                return str(cand)
    return None


def playwright_ready() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            exe = p.chromium.executable_path
            return bool(exe) and Path(exe).exists()
    except Exception:
        return False


def launch_kwargs() -> dict:
    exe = chromium_executable()
    return {"executable_path": exe} if exe else {}


def git_origin() -> str | None:
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def github_visibility(url: str) -> str:
    """'public', 'private-or-missing' (unauthenticated 404) or 'unknown'.

    An unauthenticated GitHub API request returns 404 for a private repo, so a
    404 plus a successful authenticated push means the repo is private.
    """
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        return "unknown"
    api = f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}"
    try:
        with urllib.request.urlopen(api, timeout=10) as r:
            data = json.load(r)
            return "private" if data.get("private") else "public"
    except urllib.error.HTTPError as e:
        return "private-or-missing" if e.code == 404 else "unknown"
    except Exception:
        return "unknown"
