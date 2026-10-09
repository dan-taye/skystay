"""Record a scripted walkthrough of the running app with Playwright (1920x1080).

Each scene: navigate, wait until Streamlit's rerun has finished (DOM attribute
data-test-script-state == "notRunning" and Plotly charts drawn), then perform the
scene's actions with a visible cursor and highlight, and hold. Scene start/end
times relative to the raw recording are logged to video/out/timeline.json.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from skystay import paths, tools

W, H = 1920, 1080

CURSOR_JS = """
(() => {
  const add = () => {
    if (document.getElementById('__cursor')) return;
    const c = document.createElement('div');
    c.id = '__cursor';
    Object.assign(c.style, {position: 'fixed', width: '22px', height: '22px', borderRadius: '50%',
      background: 'rgba(11,11,11,0.55)', border: '3px solid #ffffff', boxShadow: '0 0 0 1px rgba(0,0,0,0.35)',
      pointerEvents: 'none', zIndex: 2147483647, left: '-40px', top: '-40px', transform: 'translate(-50%,-50%)',
      transition: 'width 120ms, height 120ms'});
    document.body.appendChild(c);
    window.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; },
                            true);
    window.addEventListener('mousedown', () => { c.style.width = '30px'; c.style.height = '30px'; }, true);
    window.addEventListener('mouseup', () => { c.style.width = '22px'; c.style.height = '22px'; }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', add); else add();
  new MutationObserver(add).observe(document.documentElement, {childList: true, subtree: false});
})();
"""

HIGHLIGHT_JS = """
(el) => { const o = el.style.outline, oo = el.style.outlineOffset;
  el.style.outline = '3px solid #eda100'; el.style.outlineOffset = '3px';
  setTimeout(() => { el.style.outline = o; el.style.outlineOffset = oo; }, 900); }
"""


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_app(port: int) -> subprocess.Popen:
    app = paths.ROOT / "src" / "skystay" / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port), "--server.headless", "true"]
    proc = subprocess.Popen(cmd, cwd=paths.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "SKYSTAY_ROOT": str(paths.ROOT)})
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("app did not become ready")


def wait_idle(page, timeout: float = 90.0) -> None:
    """Wait for the Streamlit rerun to finish and charts to be drawn, then let rendering settle."""
    page.wait_for_function(
        """() => { const a = document.querySelector('[data-testid="stApp"]');
                   return a && a.getAttribute('data-test-script-state') === 'notRunning'; }""",
        timeout=timeout * 1000,
    )
    page.wait_for_function(
        """() => Array.from(document.querySelectorAll('.js-plotly-plot')).every(p => p.querySelector('.main-svg'))""",
        timeout=timeout * 1000,
    )
    page.wait_for_timeout(700)


class Cursor:
    def __init__(self, page):
        self.page = page
        self.x, self.y = W / 2, H / 2

    def move_to(self, x: float, y: float, steps: int = 28) -> None:
        self.page.mouse.move(x, y, steps=steps)
        self.x, self.y = x, y

    def to_locator(self, loc) -> None:
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        if box:
            self.move_to(box["x"] + min(box["width"] / 2, 60), box["y"] + box["height"] / 2)


def do_action(page, cur: Cursor, act: dict) -> None:
    if "wait" in act:
        page.wait_for_timeout(int(act["wait"] * 1000))
        return
    if "scroll" in act:
        cur.move_to(W * 0.075, H * 0.55, steps=12)
        total, step = int(act["scroll"]), 40
        for _ in range(max(total // step, 1)):
            page.mouse.wheel(0, step)
            page.wait_for_timeout(22)
        page.wait_for_timeout(400)
        return
    if "hover_text" in act:
        loc = page.get_by_text(act["hover_text"], exact=False).first
        cur.to_locator(loc)
        return
    if "click_text" in act:
        loc = page.get_by_text(act["click_text"], exact=True).first
    elif "click_tab" in act:
        loc = page.get_by_role("tab", name=act["click_tab"]).first
    elif "click_role" in act:
        role, name = act["click_role"]
        loc = page.get_by_role(role, name=name).first
    else:
        raise ValueError(f"unknown action {act}")
    cur.to_locator(loc)
    page.wait_for_timeout(250)
    loc.evaluate(HIGHLIGHT_JS)
    page.wait_for_timeout(350)
    page.mouse.down()
    page.wait_for_timeout(90)
    page.mouse.up()
    page.wait_for_timeout(300)
    wait_idle(page)


def record(scenes: list[dict], out_dir: Path | None = None, port: int | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    out_dir = out_dir or paths.VIDEO_OUT
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in raw_dir.glob("*.webm"):
        f.unlink()
    port = port or free_port()
    proc = start_app(port)
    base = f"http://127.0.0.1:{port}/"
    timeline = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**tools.launch_kwargs())
            ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(raw_dir),
                                      record_video_size={"width": W, "height": H}, device_scale_factor=1)
            ctx.add_init_script(CURSOR_JS)
            warm = ctx.new_page()  # loads the model once; its video is discarded
            warm.goto(base)
            wait_idle(warm, timeout=180)
            warm.close()
            for sc in scenes:
                if sc.get("type") == "card":
                    continue
                page = ctx.new_page()  # one page per scene = one video file per scene (no clock drift across scenes)
                t0 = time.monotonic()
                page.goto(base + sc["page"])
                wait_idle(page)
                cur = Cursor(page)
                cur.move_to(W * 0.5, H * 0.45, steps=5)
                start = time.monotonic() - t0
                for act in sc.get("actions", []):
                    do_action(page, cur, act)
                elapsed = time.monotonic() - t0 - start
                rest = sc["hold_seconds"] - elapsed
                if rest > 0:
                    page.wait_for_timeout(int(rest * 1000))
                end = time.monotonic() - t0
                path = page.video.path()
                page.close()
                timeline.append({"id": sc["id"], "raw": str(Path(path)), "start": round(start, 3),
                                 "end": round(end, 3), "actions_seconds": round(elapsed, 2)})
                print(f"  recorded {sc['id']}: {end - start:.1f}s (actions {elapsed:.1f}s)")
            ctx.close()
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    out = {"scenes": timeline, "viewport": [W, H]}
    (out_dir / "timeline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
