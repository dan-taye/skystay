"""Placeholder voiceover with offline Windows text-to-speech (SAPI, System.Speech).

Writes video/voiceover.m4a: one line per scene (the same lines as
video/narration_script.md), each placed at its scene's start in the final cut
and sped up only if it would overrun its slot. A teammate can replace the file
with their own recording under the same name; `skystay video --no-record` then
remixes it. No text leaves this machine.
"""

from __future__ import annotations

import json
import platform
import subprocess
import wave
from pathlib import Path

import numpy as np
import yaml

from skystay import paths
from skystay.video import build

VOICE = "Microsoft Zira Desktop"
BASE_RATE = 0  # SAPI rate -10..10
MAX_RATE = 4
GAP = 0.35  # seconds of silence kept before the next scene


def synthesize(text: str, out: Path, rate: int, voice: str = VOICE) -> float:
    """Speak `text` to a WAV file with SAPI; returns its duration in seconds."""
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); $s.Rate = {rate}; "
        f"$s.SetOutputToWaveFile('{out}'); "
        "$t = [Console]::In.ReadToEnd(); $s.Speak($t); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], input=text, text=True,
                   capture_output=True, check=True)
    with wave.open(str(out), "rb") as w:
        return w.getnframes() / w.getframerate()


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return data, sr


def run() -> Path:
    if platform.system() != "Windows":
        raise RuntimeError("the placeholder voiceover uses Windows SAPI; record video/voiceover.m4a by hand instead")
    rep = json.loads((paths.VIDEO_OUT / "video_report.json").read_text(encoding="utf-8"))
    durs = rep["scene_durations"]
    spec = yaml.safe_load((paths.VIDEO / "demo_script.yaml").read_text(encoding="utf-8"))
    scenes = spec["scenes"]
    lines = build.narration_lines(scenes, build.caption_values())
    work = paths.VIDEO_OUT / "tts"
    work.mkdir(parents=True, exist_ok=True)
    clips, starts = [], []
    t = 0.0
    for sc, line in zip(scenes, lines, strict=True):
        slot = durs[sc["id"]]
        f = work / f"{sc['id']}.wav"
        rate = BASE_RATE
        d = synthesize(line, f, rate)
        while d > slot - GAP and rate < MAX_RATE:
            rate += 1
            d = synthesize(line, f, rate)
        status = "fits" if d <= slot - GAP else "OVERRUNS"
        print(f"  {sc['id']:14s} slot {slot:5.2f}s  speech {d:5.2f}s  rate {rate:+d}  {status}")
        clips.append(f)
        starts.append(t + 0.15)
        t += slot
    total = t
    data0, sr = _read(clips[0])
    track = np.zeros(int(total * sr) + sr, dtype=np.int32)
    for f, st in zip(clips, starts, strict=True):
        x, _ = _read(f)
        i = int(st * sr)
        j = min(i + len(x), len(track))
        track[i:j] += x[: j - i]
    track = np.clip(track, -32768, 32767).astype(np.int16)
    wav = work / "voiceover.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(track.tobytes())
    out = paths.VIDEO / "voiceover.m4a"
    build.ff("-i", str(wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
             str(out))
    print(f"wrote {out.relative_to(paths.ROOT)} ({total:.1f}s, voice: {VOICE}, offline SAPI)")
    return out
