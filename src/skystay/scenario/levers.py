"""Scenario and lever definitions (pydantic: JSON-serialisable, hashable for caching).

Every lever has a default (no change), an allowed range, a unit and help text.
Market keys are ISO 3166 alpha-2 codes; "ALL" applies to every market.
Seasons are the names in config/settings.yaml, or "ALL".
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Lever(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NewRoute(_Lever):
    kind: Literal["new_route"] = "new_route"
    origin: str = Field(description="Departure country (ISO2); may have no current service (cold start)")
    airline: str = Field("New service", description="Operating carrier label")
    aircraft: str = Field("B787-9", description="Aircraft type (sets seats per flight)")
    seats_per_flight: int | None = Field(None, ge=50, le=600, description="Override seats per flight")
    weekly_freq: float = Field(2.0, gt=0, le=56, description="Weekly frequency (flights per week)")
    start_month: int = Field(1, ge=1, le=12, description="First month of operation (1 = January)")


class DiscontinueRoute(_Lever):
    kind: Literal["discontinue_route"] = "discontinue_route"
    route_id: str = Field(description="Route id 'AIRPORT · CARRIER' as listed in the route table")
    from_month: int = Field(1, ge=1, le=12)


class Frequency(_Lever):
    kind: Literal["frequency"] = "frequency"
    market: str = "ALL"
    pct: float = Field(0.0, ge=-100, le=200, description="Change in weekly frequency, %")
    season: str = "ALL"


class Aircraft(_Lever):
    kind: Literal["aircraft"] = "aircraft"
    market: str
    aircraft: str | None = Field(None, description="Swap every route of the market to this aircraft type")
    seats_pct: float = Field(0.0, ge=-50, le=100, description="Or change seats per flight by this %")


class LoadFactor(_Lever):
    kind: Literal["load_factor"] = "load_factor"
    market: str = "ALL"
    delta_pp: float = Field(0.0, ge=-40, le=30, description="Change in load factor, percentage points")
    value: float | None = Field(None, gt=0, le=1.0, description="Or set an absolute load factor (0-1]")


class TransferShare(_Lever):
    kind: Literal["transfer_share"] = "transfer_share"
    market: str = "ALL"
    delta_pp: float = Field(0.0, ge=-40, le=40, description="Change in transfer/transit share, percentage points")


class Capture(_Lever):
    kind: Literal["capture"] = "capture"
    market: str = "ALL"  # nationality
    pct: float = Field(0.0, ge=-90, le=200, description="Change in hotel capture rate, %")


class Alos(_Lever):
    kind: Literal["alos"] = "alos"
    market: str = "ALL"  # nationality
    pct: float = Field(0.0, ge=-50, le=100, description="Change in average length of stay, %")


class MixShift(_Lever):
    kind: Literal["mix_shift"] = "mix_shift"
    from_market: str
    to_market: str
    pct: float = Field(10.0, ge=0, le=100, description="% of from_market seats moved to to_market")
    season: str = "ALL"


class SeasonShift(_Lever):
    kind: Literal["season_shift"] = "season_shift"
    market: str = "ALL"
    from_season: str
    to_season: str
    pct: float = Field(10.0, ge=0, le=100, description="% of the market's from_season seats moved to to_season")


class Event(_Lever):
    kind: Literal["event"] = "event"
    month: int = Field(ge=1, le=12)
    days: int = Field(3, ge=-31, le=31, description="Event days added (negative removes)")


Lever = Annotated[
    NewRoute | DiscontinueRoute | Frequency | Aircraft | LoadFactor | TransferShare | Capture | Alos | MixShift
    | SeasonShift | Event,
    Field(discriminator="kind"),
]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = "Scenario"
    levers: tuple[Lever, ...] = ()
    draws: int = Field(500, ge=50, le=5000)
    seed: int = 42

    @field_validator("levers", mode="before")
    @classmethod
    def _tuple(cls, v):
        return tuple(v) if isinstance(v, list) else v

    def key(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


LEVER_HELP = {
    "new_route": ("Routes", "Add a route from any country, including one with no service today (cold-start estimate)."),
    "discontinue_route": ("Routes", "Stop an existing route from a given month."),
    "frequency": ("Capacity", "More or fewer weekly flights, in %."),
    "aircraft": ("Capacity", "Bigger or smaller aircraft: changes seats per flight."),
    "load_factor": ("Conversion", "Share of seats filled. Bounded so it never exceeds 100%."),
    "transfer_share": ("Conversion", "Share of arriving passengers who connect onward and never enter Abu Dhabi."),
    "capture": ("Conversion", "Share of visitors who stay in a hotel (returning residents are folded in)."),
    "alos": ("Conversion", "Nights per hotel guest."),
    "mix_shift": ("Season & events", "Move seats from one market to another."),
    "season_shift": ("Season & events", "Move seats from one season to another."),
    "event": ("Season & events", "Add or remove Abu Dhabi event days in a month."),
}
