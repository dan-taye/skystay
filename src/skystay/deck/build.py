"""`skystay deck`: the 10-page PDF deck (Jinja2 HTML -> Playwright page.pdf), every number from reports/*.json.

Also captures app screenshots with Playwright, renders every page to PNG for
inspection (deck/out/pages/), checks page count and overflow, and copies the
PDF to demo/<product>_deck.pdf.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader

from skystay import paths, settings, tools
from skystay.reconcile import regions

TPL = Path(__file__).parent / "templates"
W, H = 1920, 1080


def _j(name: str) -> dict:
    f = paths.REPORTS / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _p(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _s(x: float) -> str:
    return f"{x:+,.0f}"


def count_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=paths.ROOT, capture_output=True,
                         text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    return int(m.group(1)) if m else 0


def context() -> dict:
    from skystay.app.views.how import diagram
    from skystay.backtest.runner import LEVEL_LABELS, MODEL_LABELS, MODELS
    from skystay.scenario.engine import Engine
    from skystay.scenario.levers import Frequency, NewRoute, Scenario
    from skystay.train import load
    from skystay.video.build import caption_values

    met, mod, sens, rec, dq = _j("metrics"), _j("model"), _j("sensitivity"), _j("recovery"), _j("data_quality")
    asm = yaml.safe_load(paths.ASSUMPTIONS_FILE.read_text(encoding="utf-8"))
    syn = settings.is_synthetic()
    sel = met["selection"]
    lv = sel["level"]
    t = met["table"][lv]
    models = [{"label": MODEL_LABELS[m], "cv": _p(t[m]["cv_guests"]["wmape_mean"]),
               "ho": _p(t[m]["holdout"]["guests"]["wmape"]), "chosen": m == sel["model"]} for m in MODELS]
    recon = [{"label": LEVEL_LABELS[k], "par_cv": _p(v["parametric"]["cv_wmape"]), "hyb_cv": _p(v["hybrid"]["cv_wmape"]),
              "hyb_ho": _p(v["hybrid"]["holdout_wmape"]), "chosen": k == lv}
             for k, v in met["reconciliation"]["by_level"].items()]
    ho = met["chosen"]["holdout"]
    cov = met["chosen"]["coverage_holdout"]
    vals = caption_values()

    eng = Engine(load())
    a = eng.baseline.arrays
    i = int(np.argmax(a["flow"].sum(axis=(1, 2))))
    o = a["origins"][i]
    nat = o if o in eng.nats else "OTHER"

    def iv(e, k=3):
        lo, hi = e["interval_80"]
        return f"{lo:.{k}g} to {hi:.{k}g}"

    chain = {
        "origin": regions.label(o), "nat": regions.label(nat),
        "lf": _p(asm["load_factor"][o]["value"]), "lf_iv": iv(asm["load_factor"][o]),
        "tr": _p(asm["transfer_share"][o]["value"]), "tr_iv": iv(asm["transfer_share"][o]),
        "cap": f"{asm['capture_rate'][nat]['value']:.2f}", "cap_iv": iv(asm["capture_rate"][nat], 2),
        "alos": f"{asm['length_of_stay'][nat]['value']:.2f}", "alos_iv": iv(asm["length_of_stay"][nat]),
        "nonair": _p(asm["non_air_baseline"][nat]["non_air_share_last12"], 0),
    }
    # timing of a typical scenario (measured, not assumed)
    sc = Scenario(levers=[Frequency(market=o, pct=-30, season=list(settings.seasons())[0])])
    eng.run(sc)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        eng.run(sc)
        ts.append(time.perf_counter() - t0)
    speed_ms = int(round(np.median(ts) * 1000, -1))
    nr = eng.run(Scenario(levers=[NewRoute(origin="CA", aircraft="B787-9", weekly_freq=2.0)]))

    p1000 = sens["per_1000_seats"]["rows"]
    served = [r for r in p1000 if r["status"] == "served"]
    cold = [r for r in p1000 if r["status"] != "served"]

    def row(r):
        return {"name": r["name"], "v": _n(r["guests_per_1000_seats"]), "iv": f"{_n(r['lo80'])} to {_n(r['hi80'])}"}

    vul = sens["vulnerability"]
    worst = next(r for r in vul["rows"] if r["season"] == vul["most_vulnerable"])
    sob = [{"label": f["label"], "ST": f"{f['ST']:.2f}"} for f in sens["sobol"]["factors"]]
    folds = [f for f in met["folds"] if not f["holdout"]]
    hold = next(f for f in met["folds"] if f["holdout"])
    from skystay.app.views.simulator import preset_scenario
    wc = eng.run(preset_scenario("winter_cut", eng))
    checks = rec.get("checks", {})
    return {
        "syn": syn, "product": settings.product_name(), "team_name": settings.team_name(),
        "byline": settings.byline(),
        "members": settings.team_members() or [{"name": settings.TBC, "role": settings.TBC}],
        "university": settings.load_team().get("university", ""), "team_role": settings.TBC,
        "n_levers": 11, "n_regions": len([r for r in regions.region_names() if r not in ("UAE", "OTH")]),
        "dq": {"first": dq["overlap_window"]["first"], "last": dq["overlap_window"]["last"],
               "n_months": dq["overlap_window"]["n_months"], "origin_countries": dq["counts"]["origin_countries"],
               "origin_airports": dq["counts"]["origin_airports"], "nationalities": dq["counts"]["nationalities"],
               "events": dq["counts"]["events"],
               "flags": len(dq["missing_hotel_months"]) + len(dq["missing_pax_months"]) + dq["load_factor"]["above_1"]
               + sum(1 for a_ in dq["adapters"] for n_ in a_["notes"] if "duplicate" in n_)},
        "sel": sel, "models": models, "recon": recon,
        "ho": {"wmape": _p(ho["guests"]["total"]["wmape"]), "nights": _p(ho["nights"]["total"]["wmape"]),
               "bias": f"{ho['guests']['total']['bias']:+.1%}"},
        "naive_ho": _p(t["seasonal_naive"]["holdout"]["guests"]["wmape"]),
        "cov80": _p(cov["guests"]["0.8"], 0), "cov95": _p(cov["guests"]["0.95"], 0),
        "rec": {"passed": sum(c["pass"] for c in checks.values()), "total": len(checks),
                "nonair": f"{checks.get('non_air_share', {}).get('value', float('nan')):.2f}"},
        "folds": ", ".join(f["test_first"][:4] for f in folds), "holdout": hold["test_first"][:4],
        "model": {"nonair_total": _p(mod["last12"]["non_air_share_total"], 0)},
        "chain": chain, "speed_ms": speed_ms, "draws": eng.draws, "n_tests": count_tests(),
        "sob": sob, "sobol_runs": f"{sens['sobol']['evaluations']:,}",
        "per1000_top": [row(r) for r in served[:5]], "per1000_low": [row(r) for r in served[-2:]],
        "per1000_cold": [row(r) for r in cold[:3]],
        "vul": {"season": worst["season"], "nights": _n(-worst["delta_nights"]), "pct": _p(-worst["pct_of_season"])},
        "bullets": sens["planner_brief"]["sections"]["do_differently"]["bullets"],
        "nr": {"origin": regions.label("CA"), "delta": _s(nr.totals["delta_guests"]),
               "lo": _s(nr.totals["delta_guests_interval"]["lo80"]),
               "hi": _s(nr.totals["delta_guests_interval"]["hi80"])},
        "wc": {"market": regions.label(o), "delta": _s(wc.totals["delta_guests"])},
        "svg": diagram(sel["level_label"], sel["model_label"], syn),
        "caption_values": vals,
    }


# ------------------------------------------------------------------ screenshots
def screenshots(out: Path) -> dict[str, Path]:
    from playwright.sync_api import sync_playwright

    from skystay.video import record as rec

    out.mkdir(parents=True, exist_ok=True)
    port = rec.free_port()
    proc = rec.start_app(port)
    base = f"http://127.0.0.1:{port}/"
    shots = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**tools.launch_kwargs())
            ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
            page = ctx.new_page()
            page.goto(base)
            rec.wait_idle(page, timeout=180)

            def settle():
                page.wait_for_timeout(700)  # let a click start its rerun before checking the script state
                rec.wait_idle(page)

            def content_clip(top: float = 0, height: float = 1000):
                loc = page.locator(".block-container").first
                bb = loc.bounding_box() if loc.count() else None
                x, w = (bb["x"], bb["width"]) if bb else (0, 1600)
                return {"x": max(x, 0), "y": top, "width": min(w, 1600 - max(x, 0)), "height": height}

            def results_clip(anchor: str, above: float = 70, height: float = 760):
                """Crop to the simulator's right-hand results column, starting just above `anchor`."""
                kb = page.get_by_text("Change in hotel guests", exact=True).first.bounding_box()
                ab_ = page.get_by_text(anchor, exact=False).first.bounding_box()
                blk = page.locator(".block-container").first.bounding_box()
                x0 = kb["x"] - 18
                y0 = max(ab_["y"] - above, 0)
                return {"x": x0, "y": y0, "width": blk["x"] + blk["width"] - x0, "height": min(height, 1000 - y0)}

            def open_page(url):
                page.goto(base + url)
                rec.wait_idle(page)
                page.mouse.move(5, 995)

            # Sankey and bridge heatmap: element screenshots of the Plotly charts
            open_page("chain")
            page.locator(".js-plotly-plot").first.screenshot(path=str(out / "sankey.png"))
            shots["sankey"] = out / "sankey.png"
            open_page("data")
            heat = page.locator(".js-plotly-plot").first
            heat.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            heat.screenshot(path=str(out / "bridge.png"))
            shots["bridge"] = out / "bridge.png"
            # simulator with the new-route preset
            open_page("simulator?preset=new_route")
            page.wait_for_timeout(400)
            page.screenshot(path=str(out / "sim.png"), clip=results_clip("new 2x-weekly", above=12))
            shots["sim"] = out / "sim.png"
            # scenario A (2 flights/week) vs B (7 flights/week)
            for _attempt in range(3):
                open_page("simulator?preset=new_route")
                page.get_by_role("tab", name="Scenario A vs B").first.click()
                settle()
                page.get_by_text("Save current as Scenario A", exact=True).first.click()
                settle()
                slider = page.locator('input[type="range"][aria-label="Weekly flights"]').first
                slider.focus()
                for _ in range(5):
                    page.keyboard.press("ArrowRight")
                    page.wait_for_timeout(150)
                settle()
                page.get_by_role("tab", name="Scenario A vs B").first.click()
                settle()
                page.get_by_text("Save current as Scenario B", exact=True).first.click()
                settle()
                if page.get_by_text("Scenario A:", exact=False).count() and page.get_by_text(
                        "Scenario B:", exact=False).count():
                    break
            anchor = page.get_by_text("Save current as Scenario A", exact=True).first
            anchor.scroll_into_view_if_needed()
            page.mouse.move(5, 995)
            page.mouse.wheel(0, anchor.bounding_box()["y"] - 40)  # bring the comparison to the top of the viewport
            page.wait_for_timeout(700)
            page.screenshot(path=str(out / "ab.png"), clip=results_clip("Save current as Scenario A", above=20))
            shots["ab"] = out / "ab.png"
            # planner brief
            open_page("brief")
            page.wait_for_timeout(400)
            page.screenshot(path=str(out / "brief.png"), clip=content_clip(top=40, height=820))
            shots["brief"] = out / "brief.png"
            ctx.close()
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return shots


# ------------------------------------------------------------------ render
def render(ctx: dict, out_dir: Path) -> Path:
    env = Environment(loader=FileSystemLoader(str(TPL)), autoescape=True)
    css = (TPL / "deck.css").read_text(encoding="utf-8")
    html = env.get_template("deck.html.j2").render(css=css, **ctx)
    f = out_dir / "deck.html"
    f.write_text(html, encoding="utf-8")
    return f


def to_pdf_and_pngs(html: Path, pdf: Path, pages_dir: Path) -> dict:
    from playwright.sync_api import sync_playwright

    pages_dir.mkdir(parents=True, exist_ok=True)
    for old in pages_dir.glob("page_*.png"):
        old.unlink()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**tools.launch_kwargs())
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(html.as_uri())
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(500)
        overflow = page.evaluate("""() => Array.from(document.querySelectorAll('.slide')).map((s, i) => {
            const r = s.getBoundingClientRect(); let worst = 0;
            s.querySelectorAll('*').forEach(el => { const b = el.getBoundingClientRect();
              if (b.width && b.height && !el.closest('.footer') && !el.classList.contains('bar')) {
                worst = Math.max(worst, b.bottom - (r.bottom - 60), b.right - r.right); } });
            return {page: i + 1, overflow_px: Math.round(worst)}; })""")
        slides = page.locator(".slide")
        n = slides.count()
        for k in range(n):
            slides.nth(k).screenshot(path=str(pages_dir / f"page_{k + 1:02d}.png"))
        page.pdf(path=str(pdf), width=f"{W}px", height=f"{H}px", print_background=True, prefer_css_page_size=True)
        browser.close()
    pdf_pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    return {"slides": n, "pdf_pages": pdf_pages, "overflow": overflow}


def run(capture: bool = True) -> int:
    paths.ensure_dirs()
    out = paths.DECK_OUT
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    shots_dir = out / "shots"
    if capture or not (shots_dir / "sim.png").exists():
        # separate process: the app server + Chromium must not share this process's memory with the engine below
        print("capturing app screenshots...")
        code = f"from pathlib import Path; from skystay.deck.build import screenshots; screenshots(Path(r'{shots_dir}'))"
        rc = subprocess.run([sys.executable, "-c", code], cwd=paths.ROOT).returncode
        if rc != 0:
            print("screenshot capture failed")
            return rc
    print("building context from reports...")
    ctx = context()
    for name in ("backtest_top10", "sobol", "guests_per_1000_seats", "tornado"):
        shutil.copy2(paths.FIGURES / f"{name}.png", img / f"{name}.png")
    ctx["imgs"] = {"sankey": "shots/sankey.png", "sim": "shots/sim.png", "ab": "shots/ab.png", "bridge": "shots/bridge.png",
                   "brief": "shots/brief.png", "top10": "img/backtest_top10.png", "sobol": "img/sobol.png"}
    html = render(ctx, out)
    pdf = out / f"{settings.product_name()}_deck.pdf"
    res = to_pdf_and_pngs(html, pdf, out / "pages")
    bad = [o for o in res["overflow"] if o["overflow_px"] > 2]
    report = {"pdf": str(pdf.relative_to(paths.ROOT)), "size_mb": round(pdf.stat().st_size / 1e6, 2), **res,
              "pages_with_overflow": bad}
    (out / "deck_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pdf", "size_mb", "slides", "pdf_pages", "pages_with_overflow")}, indent=1))
    from skystay import docs

    docs.write_all()
    if res["slides"] == 10 and res["pdf_pages"] == 10 and not bad:
        shutil.copy2(pdf, paths.DEMO / pdf.name)
        print(f"copied to demo/{pdf.name}")
        return 0
    print("deck checks failed: NOT copied to demo/")
    return 1
