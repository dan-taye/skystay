"""Deterministic preset scenarios (used by the app's ?preset= URLs, the video and the deck)."""

from __future__ import annotations

from skystay.scenario.levers import Frequency, NewRoute, Scenario

WINTER = "Winter peak (Dec-Feb)"


def presets(top_market: str = "GB", new_route_origin: str = "CA") -> dict[str, Scenario]:
    return {
        "new_route": Scenario(
            name=f"New 2x-weekly route from {new_route_origin}",
            levers=(NewRoute(origin=new_route_origin, airline="New service", aircraft="B787-9", weekly_freq=2.0,
                             start_month=1),),
        ),
        "winter_cut": Scenario(
            name=f"Winter frequency cut on {top_market} (-30%)",
            levers=(Frequency(market=top_market, pct=-30.0, season=WINTER),),
        ),
    }
