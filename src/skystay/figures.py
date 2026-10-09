"""Shared Plotly styling, market colours and PNG export (via Playwright, no kaleido).

Colour rules (dataviz reference palette, validated):
- categorical hues in fixed order; only the top-7 markets by guest volume get a
  hue (fixed once from the data), every other market is neutral grey
- sequential = one hue (blue) light -> dark; diverging = blue <-> red with grey midpoint
- text never wears a series colour
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from skystay import paths, settings

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b5b3ab"
BAND = "rgba(42,120,214,0.18)"
BAND_95 = "rgba(42,120,214,0.09)"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
BADGE_TEXT = "SYNTHETIC DEMO DATA"


def layout(fig: go.Figure, title: str | None = None, height: int | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="none",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "color": INK_2, "size": 13},
        title={"text": title, "font": {"color": INK, "size": 16}, "x": 0.01, "xanchor": "left",
               "yref": "container", "y": 0.985, "yanchor": "top"} if title else None,
        margin={"l": 60, "r": 24, "t": (88 if legend else 60) if title else 40, "b": 48},
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0, "font": {"color": INK_2},
                "yref": "paper"},
        showlegend=legend,
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK, "family": FONT}, "bordercolor": GRID},
        separators=".,",
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", zeroline=False, automargin=True,
                     title_font={"color": INK_2}, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=False, tickformat=",", automargin=True,
                     title_font={"color": INK_2}, tickfont={"color": MUTED})
    if height:
        fig.update_layout(height=height)
    return badge(fig)


def badge(fig: go.Figure) -> go.Figure:
    """Stamp the synthetic-data badge on the figure itself (so exports carry it too)."""
    if not settings.is_synthetic():
        return fig
    existing = [a for a in (fig.layout.annotations or []) if getattr(a, "name", None) == "synthetic-badge"]
    if existing:
        return fig
    fig.add_annotation(
        name="synthetic-badge", text=f"<b>{BADGE_TEXT}</b>", xref="paper", yref="paper", x=1, y=1,
        xanchor="right", yanchor="top", showarrow=False, font={"size": 11, "color": "#7a4b00"},
        bgcolor="#fff3d6", bordercolor="#eda100", borderwidth=1, borderpad=3,
        yshift=max(int(fig.layout.margin.t or 40) - 8, 20),
    )
    return fig


def top_markets(guests_by_market: dict[str, float], k: int = 7) -> list[str]:
    return [m for m, _ in sorted(guests_by_market.items(), key=lambda kv: -kv[1])[:k]]


def market_colors(top: list[str], all_markets: list[str]) -> dict[str, str]:
    """Fixed entity -> colour map: top markets take slots 1..7 in rank order, the rest are grey."""
    cmap = {m: SERIES[i] for i, m in enumerate(top[:7])}
    return {m: cmap.get(m, OTHER) for m in all_markets}


def save_png(figs: dict[str, go.Figure], folder: Path | None = None, width: int = 1100, height: int = 560,
             scale: float = 2.0) -> list[Path]:
    """Render Plotly figures to PNG with headless Chromium (one browser session for the batch)."""
    from playwright.sync_api import sync_playwright

    from skystay import tools

    folder = folder or paths.FIGURES
    folder.mkdir(parents=True, exist_ok=True)
    import plotly

    js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    out = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**tools.launch_kwargs())
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
        page.set_content(f"<html><body style='margin:0;background:{SURFACE}'><div id='c'></div></body></html>")
        page.add_script_tag(path=str(js))
        for name, fig in figs.items():
            h = int(fig.layout.height or height)
            w = int(fig.layout.width or width)
            page.set_viewport_size({"width": w, "height": h})
            spec = json.loads(fig.to_json())
            page.evaluate(
                "([data, layout, w, h]) => { layout.width = w; layout.height = h; "
                "return Plotly.newPlot('c', data, layout, {staticPlot: true}); }",
                [spec.get("data", []), spec.get("layout", {}), w, h],
            )
            page.wait_for_timeout(150)
            path = folder / f"{name}.png"
            page.locator("#c").screenshot(path=str(path))
            out.append(path)
        browser.close()
    return out
