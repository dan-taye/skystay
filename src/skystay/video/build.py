"""`skystay video`: record the app, render cards and captions, assemble with ffmpeg, verify, copy to demo/.

ffmpeg: libx264, yuv420p, 30 fps, CRF 20, +faststart. Captions are Pillow-rendered
overlays (the bundled ffmpeg may lack libass). Narration: video/narration_script.md
is written for a teammate to record; if video/voiceover.(wav|mp3|m4a) exists it is
mixed in. No music.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from skystay import paths, settings, tools
from skystay.reconcile import regions

W, H = 1920, 1080
FPS = 30
CRF = 20
LEAD_TRIM = 0.25  # seconds trimmed from each recorded scene start (cursor settle)


# ------------------------------------------------------------------ fonts
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    cands = [r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
             r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    try:
        import matplotlib

        mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        cands.append(str(mpl / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")))
    except Exception:
        pass
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default(size)


# ------------------------------------------------------------------ captions from reports
def caption_values() -> dict:
    from skystay.app.views.simulator import preset_scenario, top_origin
    from skystay.scenario.engine import Engine
    from skystay.scenario.levers import NewRoute, Scenario
    from skystay.train import load

    met = json.loads((paths.REPORTS / "metrics.json").read_text(encoding="utf-8"))
    sens = json.loads((paths.REPORTS / "sensitivity.json").read_text(encoding="utf-8"))
    e = Engine(load())
    a = e.baseline.arrays
    o = top_origin(e)
    i = a["origins"].index(o)
    chain_guests = float((a["flow"][i] * a["factor"]).sum())
    nr = e.run(Scenario(levers=[NewRoute(origin="CA", aircraft="B787-9", weekly_freq=2.0, start_month=1)]))
    wc = e.run(preset_scenario("winter_cut", e))
    dg = nr.totals["delta_guests_interval"]
    lv = met["selection"]["level"]
    short = {"capacity": "seat capacity", "load_factor": "load factor", "transfer_share": "transfer share",
             "capture": "capture rate", "alos": "length of stay", "events": "events", "season_shift": "season shift",
             "market_mix": "market mix"}
    sob = [short[f["factor"]] for f in sens["sobol"]["factors"][:3]]
    return {
        "chain_origin": regions.label(o), "chain_seats": f"{a['S'][i].sum():,.0f}", "chain_guests": f"{chain_guests:,.0f}",
        "new_route_origin": regions.label("CA"), "new_route_delta": f"{nr.totals['delta_guests']:+,.0f}",
        "new_route_lo": f"{dg['lo80']:+,.0f}", "new_route_hi": f"{dg['hi80']:+,.0f}",
        "winter_market": regions.label(top_origin(e)), "winter_delta": f"{wc.totals['delta_guests']:+,.0f}",
        "sobol_1": sob[0], "sobol_2": sob[1], "sobol_3": sob[2],
        "wmape": f"{met['chosen']['holdout']['guests']['total']['wmape'] * 100:.1f}%",
        "naive": f"{met['table'][lv]['seasonal_naive']['holdout']['guests']['wmape'] * 100:.1f}%",
        "new_route_nights": f"{nr.totals['delta_nights']:+,.0f}",
        "winter_nights": f"{wc.totals['delta_nights']:+,.0f}",
    }


# ------------------------------------------------------------------ images
def _badge(d: ImageDraw.ImageDraw, x_right: int, y: int, scale: float = 1.0) -> None:
    f = font(int(22 * scale), bold=True)
    text = "SYNTHETIC DEMO DATA"
    tw = d.textlength(text, font=f)
    pad = int(12 * scale)
    box = (x_right - tw - 2 * pad, y, x_right, y + int(40 * scale))
    d.rounded_rectangle(box, radius=int(8 * scale), fill="#fff3d6", outline="#eda100", width=2)
    d.text((box[0] + pad, y + int(7 * scale)), text, font=f, fill="#7a4b00")


def card(kind: str, out: Path) -> Path:
    img = Image.new("RGB", (W, H), "#fcfcfb")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 14), fill="#2a78d6")
    product = settings.product_name()
    team = settings.team_name()
    if kind == "title":
        d.text((160, 330), product, font=font(140, bold=True), fill="#0b0b0b")
        d.text((166, 520), "Flight-to-hotel demand scenario simulator", font=font(56), fill="#52514e")
        d.text((166, 620), "For Abu Dhabi tourism planning", font=font(46, bold=True), fill="#2a78d6")
        d.text((166, 700), f"Team: {team}", font=font(40), fill="#52514e")
    else:
        d.text((160, 300), "Working prototype · reproducible · one command", font=font(70, bold=True),
               fill="#0b0b0b")
        d.rounded_rectangle((160, 430, 760, 520), radius=12, fill="#0b0b0b")
        d.text((196, 446), "skystay all", font=font(52, bold=True), fill="#ffffff")
        lines = ["Every number on screen comes from the pipeline's reports.",
                 "Back-tested on held-out months, with calibrated uncertainty."]
        if settings.is_synthetic():
            lines.append("Shown on SYNTHETIC DEMO DATA with known ground truth: not real Abu Dhabi figures.")
            lines.append("Swap in real data and re-run to get real results.")
        for k, ln in enumerate(lines):
            d.text((166, 590 + k * 64), ln, font=font(40), fill="#52514e")
        d.text((166, 900), f"{product}  ·  Team: {team}", font=font(34),
               fill="#898781")
    if settings.is_synthetic():
        _badge(d, W - 40, 40, 1.3)
    img.save(out)
    return out


def caption_overlay(text: str, out: Path) -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(46, bold=True)
    tw = d.textlength(text, font=f)
    while tw > W - 260 and f.size > 30:
        f = font(f.size - 2, bold=True)
        tw = d.textlength(text, font=f)
    x0 = (W - tw) / 2 - 36
    y0 = H - 150
    d.rounded_rectangle((x0, y0, x0 + tw + 72, y0 + f.size + 40), radius=16, fill=(11, 11, 11, 215))
    d.text((x0 + 36, y0 + 16), text, font=f, fill=(255, 255, 255, 255))
    img.save(out)
    return out


# ------------------------------------------------------------------ ffmpeg
def ff(*args: str) -> None:
    exe = tools.ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
    res = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[-2000:]}")


ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-crf", str(CRF), "-preset", "medium"]


def seg_card(png: Path, dur: float, out: Path) -> Path:
    ff("-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(png),
       "-vf", f"scale={W}:{H},format=yuv420p,fade=t=in:st=0:d=0.4", *ENC, str(out))
    return out


def seg_scene(raw: Path, start: float, dur: float, cap_png: Path, out: Path) -> Path:
    ff("-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(raw), "-loop", "1", "-i", str(cap_png),
       "-filter_complex",
       f"[0:v]fps={FPS},scale={W}:{H}:flags=lanczos,setsar=1[v];"
       f"[1:v]format=rgba,fade=t=in:st=0.3:d=0.4:alpha=1[c];"
       f"[v][c]overlay=0:0:shortest=1,format=yuv420p[o]",
       "-map", "[o]", "-an", *ENC, str(out))
    return out


def find_voiceover() -> Path | None:
    for ext in ("m4a", "mp3", "wav"):
        f = paths.VIDEO / f"voiceover.{ext}"
        if f.exists():
            return f
    return None


def assemble(segments: list[Path], out: Path, voice: Path | None) -> Path:
    lst = out.parent / "concat.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="utf-8")
    joined = out.parent / "joined.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined))
    if voice:
        ff("-i", str(joined), "-i", str(voice), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
           "-b:a", "160k", "-af", "apad", "-shortest", "-movflags", "+faststart", str(out))
    else:
        ff("-i", str(joined), "-c", "copy", "-movflags", "+faststart", str(out))
    return out


def extract_frames(video: Path, times: list[tuple[str, float]], folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k, (sid, t) in enumerate(times, 1):
        p = folder / f"scene_{k:02d}_{sid}.png"
        ff("-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(p))
        out.append(p)
    return out


# ------------------------------------------------------------------ narration
NARRATION = {
    "title": "{product}: a flight-to-hotel demand simulator for the Department of Culture and Tourism, Abu Dhabi.",
    "problem": "Flights are counted by departure country, hotels by guest nationality, and they are planned by "
               "different teams. So nobody can say what a new route does to hotel demand.",
    "system": "We turn the raw flight, traffic and hotel data into one store, fit a transparent conversion chain with "
              "a machine-learning correction, and drive everything from a scenario engine.",
    "chain": "Every step is visible: {chain_seats} seats from {chain_origin} become {chain_guests} hotel guests, "
             "and each assumption has a source and an interval.",
    "whatif_route": "Add a twice-weekly route from {new_route_origin}: about {new_route_delta} hotel guests a year, "
                    "with an eighty-percent range of {new_route_lo} to {new_route_hi}, by month and by market.",
    "whatif_winter": "Cut winter frequency on {winter_market} by thirty percent and we lose {winter_delta} hotel "
                     "guests, concentrated in the peak months.",
    "sensitivity": "Sensitivity analysis shows what matters most: {sobol_1}, {sobol_2} and {sobol_3}.",
    "validation": "We back-test on held-out months: {wmape} error against {naive} for a seasonal-naive baseline, "
                  "with calibrated uncertainty.",
    "brief": "The planner brief turns this into decisions: rank routes by hotel guests per thousand seats, protect "
             "the most exposed season, and plan non-air demand separately.",
    "end": "A working prototype, reproducible with one command.{synthetic_note}",
}


def write_narration(scenes: list[dict], durations: list[float], vals: dict) -> Path:
    vals = {**vals, "product": settings.product_name(),
            "synthetic_note": " Shown here on synthetic demo data." if settings.is_synthetic() else ""}
    L = ["# Narration script", "",
         "Read in your own voice. Aim for about 2.3 words per second and stay within each time window. Record one "
         "take as `video/voiceover.m4a` (or .mp3/.wav), upload it to the `video` folder on GitHub, then re-run "
         "`skystay video`. No music.", ""]
    if settings.is_synthetic():
        L += ["> Synthetic mode: please say once that the figures are synthetic demo data (the end-card line does).", ""]
    L += ["| Time | Scene | Line |", "| :- | :- | :- |"]
    t = 0.0
    for sc, d in zip(scenes, durations, strict=True):
        line = NARRATION.get(sc["id"], "").format(**vals)
        L.append(f"| {int(t // 60)}:{t % 60:04.1f} to {int((t + d) // 60)}:{(t + d) % 60:04.1f} | {sc['id']} | {line} |")
        t += d
    out = paths.VIDEO / "narration_script.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------ main
def run(record_again: bool = True) -> int:
    from skystay.video import record as rec

    paths.ensure_dirs()
    spec = yaml.safe_load((paths.VIDEO / "demo_script.yaml").read_text(encoding="utf-8"))
    scenes = spec["scenes"]
    vals = caption_values()
    caps = {}
    for sc in scenes:
        if sc.get("caption"):
            text = sc["caption"].format(**vals)
            if len(text.split()) > 12:
                raise ValueError(f"caption for {sc['id']} has {len(text.split())} words (> 12): {text}")
            caps[sc["id"]] = text
    tl_file = paths.VIDEO_OUT / "timeline.json"
    if record_again or not tl_file.exists():
        print("recording the app...")
        tl = rec.record(scenes)
    else:
        tl = json.loads(tl_file.read_text(encoding="utf-8"))
    times = {s["id"]: s for s in tl["scenes"]}
    work = paths.VIDEO_OUT / "work"
    work.mkdir(parents=True, exist_ok=True)
    segments, durations = [], []
    for k, sc in enumerate(scenes):
        out = work / f"seg_{k:02d}_{sc['id']}.mp4"
        if sc.get("type") == "card":
            png = card("title" if sc["id"] == "title" else "end", work / f"card_{sc['id']}.png")
            seg_card(png, float(sc["hold_seconds"]), out)
            durations.append(float(sc["hold_seconds"]))
        else:
            t = times[sc["id"]]
            raw = Path(t["raw"])
            start = t["start"] + LEAD_TRIM
            dur = min(t["end"] - start, tools.probe_video(raw)["duration_s"] - start - 0.05)
            cap = caption_overlay(caps[sc["id"]], work / f"cap_{sc['id']}.png")
            seg_scene(raw, start, dur, cap, out)
            durations.append(dur)
        segments.append(out)
    product = settings.product_name()
    final = paths.VIDEO_OUT / f"{product}_demo.mp4"
    voice = find_voiceover()
    assemble(segments, final, voice)
    info = tools.probe_video(final)
    mids, t = [], 0.0
    for sc, d in zip(scenes, durations, strict=True):
        mids.append((sc["id"], t + d * 0.6))
        t += d
    frames = extract_frames(final, mids, paths.VIDEO_OUT / "frames")
    write_narration(scenes, durations, vals)
    checks = {
        "duration_60_120": 60 <= info["duration_s"] <= 120,
        "resolution_1080p": (info["width"], info["height"]) == (W, H),
        "h264": info["codec"] == "h264",
        "under_50mb": info["size_mb"] < 50,
        "audio_iff_voiceover": info["has_audio"] == bool(voice),
    }
    report = {"path": str(final.relative_to(paths.ROOT)), **info, "voiceover": str(voice) if voice else None,
              "checks": checks, "captions": caps, "scene_durations": dict(zip([s["id"] for s in scenes],
                                                                              np.round(durations, 2).tolist(),
                                                                              strict=True)),
              "frames": [str(f.relative_to(paths.ROOT)) for f in frames]}
    (paths.VIDEO_OUT / "video_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("path", "duration_s", "width", "height", "codec", "size_mb", "checks")},
                     indent=1))
    if all(checks.values()):
        shutil.copy2(final, paths.DEMO / final.name)
        print(f"copied to demo/{final.name}")
        return 0
    print("video checks failed: NOT copied to demo/")
    return 1
