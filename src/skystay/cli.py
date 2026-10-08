"""`skystay` command-line interface (the Makefile is a thin wrapper around this)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

from skystay import paths


def _banner(name: str) -> None:
    print(f"\n=== skystay {name} ===", flush=True)


def cmd_setup(_: argparse.Namespace) -> int:
    from skystay import settings, tools

    _banner("setup")
    ok = True
    py = sys.version_info
    print(f"python      {py.major}.{py.minor}.{py.micro}  {'OK' if py >= (3, 11) else 'TOO OLD (need >= 3.11)'}")
    ok &= py >= (3, 11)
    git = shutil.which("git")
    print(f"git         {git or 'MISSING'}")
    origin = tools.git_origin()
    if origin:
        print(f"origin      {origin}  visibility: {tools.github_visibility(origin)}")
    else:
        print("origin      not configured (local commits only)")
    ff = tools.ffmpeg_exe()
    print(f"ffmpeg      {ff or 'MISSING (pip install imageio-ffmpeg)'}")
    print(f"ffprobe     {tools.ffprobe_exe() or 'missing (falls back to parsing ffmpeg -i)'}")
    pw = tools.playwright_ready()
    alt = tools.chromium_executable()
    print(f"chromium    {'playwright OK' if pw else (alt or 'MISSING (run: playwright install chromium)')}")
    print(f"docker      {shutil.which('docker') or 'not installed (optional)'}")
    s = settings.write_settings()
    print(f"data mode   {s['data_mode']}  (raw files: {len(settings.raw_files())})")
    paths.ensure_dirs()
    return 0 if ok else 1


def cmd_data(_: argparse.Namespace) -> int:
    from skystay.io import pipeline

    _banner("data")
    pipeline.run()
    return 0


def cmd_train(_: argparse.Namespace) -> int:
    from skystay import train

    _banner("train")
    train.run()
    return 0


def cmd_backtest(_: argparse.Namespace) -> int:
    from skystay.backtest import runner

    _banner("backtest")
    runner.run()
    return 0


def cmd_sensitivity(_: argparse.Namespace) -> int:
    from skystay.sensitivity import run as sens

    _banner("sensitivity")
    sens.run()
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    _banner("app")
    app = paths.ROOT / "src" / "skystay" / "app" / "main.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]
    return subprocess.call(cmd, cwd=paths.ROOT)


def cmd_test(args: argparse.Namespace) -> int:
    _banner("test")
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *args.pytest_args], cwd=paths.ROOT)


def cmd_video(_: argparse.Namespace) -> int:
    from skystay.video import build

    _banner("video")
    return build.run()


def cmd_deck(_: argparse.Namespace) -> int:
    from skystay.deck import build

    _banner("deck")
    return build.run()


def cmd_clean(_: argparse.Namespace) -> int:
    _banner("clean")
    for d in (paths.PROCESSED, paths.SYNTH_GEN, paths.VIDEO_OUT, paths.DECK_OUT):
        if d.exists():
            shutil.rmtree(d)
            print(f"removed {d.relative_to(paths.ROOT)}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
        ("setup", cmd_setup),
        ("data", cmd_data),
        ("backtest", cmd_backtest),
        ("train", cmd_train),
        ("sensitivity", cmd_sensitivity),
        ("test", cmd_test),
    ]
    if not args.skip_media:
        steps += [("video", cmd_video), ("deck", cmd_deck)]
    args.pytest_args = []
    for name, fn in steps:
        t0 = time.time()
        rc = fn(args)
        print(f"[{name}] exit {rc} in {time.time() - t0:.1f}s", flush=True)
        if rc != 0:
            print(f"skystay all: stopped at '{name}'")
            return rc
    print("\nskystay all: done")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skystay", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="environment checks; writes config/settings.yaml").set_defaults(fn=cmd_setup)
    sub.add_parser("data", help="ingest raw data or generate synthetic data").set_defaults(fn=cmd_data)
    sub.add_parser("train", help="fit the production models").set_defaults(fn=cmd_train)
    sub.add_parser("backtest", help="rolling-origin back-tests and model selection").set_defaults(fn=cmd_backtest)
    sub.add_parser("sensitivity", help="tornado, Sobol and planner signals").set_defaults(fn=cmd_sensitivity)
    p_app = sub.add_parser("app", help="launch the simulator")
    p_app.add_argument("--port", type=int, default=8501)
    p_app.add_argument("--headless", action="store_true")
    p_app.set_defaults(fn=cmd_app)
    p_test = sub.add_parser("test", help="run the test suite")
    p_test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p_test.set_defaults(fn=cmd_test)
    sub.add_parser("video", help="record and assemble the demo video").set_defaults(fn=cmd_video)
    sub.add_parser("deck", help="build the 10-page PDF deck").set_defaults(fn=cmd_deck)
    p_all = sub.add_parser("all", help="everything end to end")
    p_all.add_argument("--skip-media", action="store_true", help="skip video and deck")
    p_all.set_defaults(fn=cmd_all)
    sub.add_parser("clean", help="remove generated outputs").set_defaults(fn=cmd_clean)
    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
