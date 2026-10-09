"""Layer 1: the transparent parametric conversion chain.

    seats --LF--> passengers --(1-TR)--> terminating passengers
          --conversion C[o,n,moy]--> air-attributable hotel guests (by nationality)
             x calendar uplift (source-market holidays, Abu Dhabi events)
          + non-air baseline BASE[n,moy] x event uplift      = hotel guests
          x ALOS[n,moy]                                      = guest nights

Returning residents cannot be observed in aggregated traffic, so they are folded
into the conversion (capture) step, which is therefore "hotel guests per
terminating passenger". The conversion tensor C depends on the country
reconciliation level:

- L0 identity:  origin country o feeds only nationality n == o
- L1 regional:  origins feed their region; region guests are split to
                nationalities by historical shares
- L2 bridge:    a learned non-negative origin->nationality matrix K (see
                skystay.reconcile.bridge) times a seasonal capture multiplier

Per hotel unit u (nationality or region) the fitted structure is

    G_u(t) = beta_u * m_u(moy) * A_u(t) * (1 + h*hol_u(t) + e_a*evt(t))
             + s(moy) * (alpha_u + gamma_u * tau_t) * (1 + e_b*evt(t))

where A_u is terminating passengers mapped to the unit, m_u is an EB-shrunk
seasonal capture shape, s is the DCT seasonality profile, tau is years
before the training cutoff (so alpha_u is the baseline level at the cutoff),
hol/evt are the holiday / event share of days in the month, and h, e_a, e_b
are pooled calendar uplifts. All levels share one representation (LF, TR, C,
BASE, ALOS, uplifts), which is what the scenario engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from skystay import settings
from skystay.chain.panel import Panels
from skystay.chain.shrink import eb_moy
from skystay.reconcile import regions

LEVELS = ("L0", "L1", "L2")


@dataclass
class ParametricModel:
    level: str
    origins: list[str]  # origins with training history (known)
    nationalities: list[str]
    lf: np.ndarray  # (O, 12)
    lf_se: np.ndarray
    tr: np.ndarray
    tr_se: np.ndarray
    origin_n: np.ndarray  # (O,) months of history
    conv: np.ndarray  # (O, N, 12) guests of nationality n per terminating pax from o (before calendar uplift)
    base: np.ndarray  # (N, 12) non-air guests per month at the cutoff level (before event uplift)
    alos: np.ndarray  # (N, 12)
    alos_se: np.ndarray
    region_lf: dict[str, np.ndarray]
    region_tr: dict[str, np.ndarray]
    region_conv: dict[str, np.ndarray]  # region -> (12,) total conversion per terminating pax
    global_lf: np.ndarray
    global_tr: np.ndarray
    global_conv: np.ndarray
    cutoff: str
    hol_uplift: float = 0.0  # relative capture uplift per (holiday days / days in month)
    event_uplift_air: float = 0.0
    event_uplift_base: float = 0.0
    units: list[str] = field(default_factory=list)
    unit_cap: np.ndarray | None = None  # (U, 12) capture by hotel unit (L0/L1) or seasonal multiplier (L2)
    unit_base: np.ndarray | None = None  # (U, 12)
    unit_cap_se_rel: np.ndarray | None = None  # (U,) relative SE of the annual capture level
    unit_base_se_rel: np.ndarray | None = None
    bridge: dict | None = None  # L1 split shares or L2 details (K, mix, lambda)
    hist_seats: dict[str, tuple[float, float]] = field(default_factory=dict)  # origin -> monthly seats range
    hist_lf: dict[str, tuple[float, float]] = field(default_factory=dict)
    calendar_detail: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def origin_index(self) -> dict[str, int]:
        return {o: i for i, o in enumerate(self.origins)}

    def nat_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.nationalities)}

    def is_cold(self, origin: str) -> bool:
        return origin not in self.origin_index()

    def target_nationality(self, origin: str) -> str:
        return origin if origin in self.nat_index() else "OTHER"

    def origin_params(self, origin: str) -> dict:
        """LF, TR and conversion arrays for any origin (cold-start origins borrow region values)."""
        oi = self.origin_index()
        if origin in oi:
            i = oi[origin]
            return {"lf": self.lf[i], "lf_se": self.lf_se[i], "tr": self.tr[i], "tr_se": self.tr_se[i],
                    "conv": self.conv[i], "cold": False}
        reg = regions.region_of().get(origin, "OTH")
        lf = self.region_lf.get(reg, self.global_lf)
        tr = self.region_tr.get(reg, self.global_tr)
        ec = self.region_conv.get(reg, self.global_conv)
        conv = np.zeros((len(self.nationalities), 12))
        conv[self.nat_index()[self.target_nationality(origin)]] = ec
        return {"lf": lf, "lf_se": np.full(12, 0.05), "tr": tr, "tr_se": np.full(12, 0.08), "conv": conv,
                "cold": True}

    def mix(self) -> np.ndarray:
        """(O, N) annual nationality mix of air guests per origin (rows sum to 1)."""
        c = self.conv.mean(axis=2)
        s = c.sum(axis=1, keepdims=True)
        return np.divide(c, s, out=np.zeros_like(c), where=s > 0)


# ---------------------------------------------------------------------------
# estimation helpers
# ---------------------------------------------------------------------------

def time_tau(n_t: int) -> np.ndarray:
    """Years relative to the last training month (<= 0), so alpha = baseline level at the cutoff."""
    return (np.arange(n_t) - (n_t - 1)) / 12.0


def _region_moy_means(om: pd.DataFrame, col: str, weight: str) -> dict[str, np.ndarray]:
    d = om[om[weight] > 0].dropna(subset=[col])
    out = {}
    for reg, g in d.groupby("region"):
        w = g.groupby("moy").apply(lambda x: np.average(x[col], weights=x[weight]), include_groups=False)
        arr = w.reindex(range(1, 13)).to_numpy(dtype=float)
        if np.isnan(arr).any():
            arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
        out[reg] = arr
    return out


def _global_moy(om: pd.DataFrame, col: str, weight: str) -> np.ndarray:
    d = om[om[weight] > 0].dropna(subset=[col])
    w = d.groupby("moy").apply(lambda x: np.average(x[col], weights=x[weight]), include_groups=False)
    arr = w.reindex(range(1, 13)).to_numpy(dtype=float)
    return np.where(np.isnan(arr), np.nanmean(arr), arr)


TREND_RIDGE = 0.3  # ridge on the baseline trend (last column), relative to its column scale


def _trend_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Append one penalty row shrinking the last coefficient (baseline trend) toward 0."""
    if lam <= 0 or len(y) == 0:
        return X, y
    c = np.sqrt(np.mean(X[:, -1] ** 2))
    row = np.zeros((1, X.shape[1]))
    row[0, -1] = np.sqrt(lam * len(y)) * c
    return np.vstack([X, row]), np.append(y, 0.0)


def fit_capture_base(
    G: np.ndarray, A: np.ndarray, moy: np.ndarray, s: np.ndarray,
    cm: np.ndarray | None = None, cb: np.ndarray | None = None,
    row_w: np.ndarray | None = None, n_iter: int = 6,
) -> tuple[np.ndarray, float, float, float]:
    """Fit G = beta*m(moy)*A*cm + s(moy)*cb*(alpha + gamma*tau) by bounded weighted least squares.

    cm / cb are known calendar multipliers on the air and non-air parts.
    Weights are 1/G (relative errors). Returns (beta*m (12,), beta, alpha, gamma).
    """
    n_t = len(G)
    tau = time_tau(n_t)
    cm = np.ones(n_t) if cm is None else cm
    cb = np.ones(n_t) if cb is None else cb
    rw = np.ones(n_t) if row_w is None else np.sqrt(np.asarray(row_w, dtype=float))
    ok = ~np.isnan(G) & (rw > 0)
    if ok.sum() == 0:
        return np.zeros(12), 0.0, 0.0, 0.0
    A = np.nan_to_num(A) * cm
    sb = s[moy - 1] * cb
    w = rw / np.maximum(np.nan_to_num(G, nan=1.0), 1.0)
    has_air = A[ok].max() > 0
    m_hat = np.ones(12)
    beta = alpha = gamma = 0.0
    for _ in range(n_iter if has_air else 1):
        cols = ([A * m_hat[moy - 1]] if has_air else []) + [sb, sb * tau]
        X = np.column_stack(cols)[ok] * w[ok, None]
        y = G[ok] * w[ok]
        X, y = _trend_ridge(X, y, TREND_RIDGE)
        lb = ([0.0] if has_air else []) + [0.0, -np.inf]
        res = lsq_linear(X, y, bounds=(lb, [np.inf] * len(lb)))
        coef = res.x
        if has_air:
            beta, alpha, gamma = (float(c) for c in coef)
        else:
            alpha, gamma = float(coef[0]), float(coef[1])
            break
        if beta <= 1e-12:
            m_hat = np.ones(12)
            break
        pos = ok & (A > 0)
        ratio = (G[pos] - sb[pos] * (alpha + gamma * tau[pos])) / A[pos]
        est = eb_moy(ratio, moy[pos]).value
        new = np.clip(est / beta, 0.25, 4.0)
        new = new / new.mean()
        done = np.allclose(new, m_hat, atol=1e-4)
        m_hat = new
        if done:
            break
    return beta * m_hat, beta, alpha, gamma


def estimate_calendar(
    G: np.ndarray, A: np.ndarray, moy: np.ndarray, s: np.ndarray, hf: np.ndarray, ef: np.ndarray,
) -> tuple[float, float, float] | None:
    """Unit-level calendar uplifts (holiday on air, event on air, event on base), each >= 0.

    Linearised: G = b*A + b_h*A*hol + b_e*A*evt + a*s + g*s*tau + a_e*s*evt;
    uplifts are b_h/b, b_e/b and a_e/a. Returns None when the unit has no air supply.
    """
    n_t = len(G)
    tau = time_tau(n_t)
    ok = ~np.isnan(G)
    A = np.nan_to_num(A)
    if ok.sum() < 24 or A[ok].max() <= 0:
        return None
    # seasonal capture shape first, so calendar terms do not absorb ordinary seasonality
    cap, beta, _, _ = fit_capture_base(G, A, moy, s)
    if beta <= 1e-9:
        return None
    A = A * (cap / beta)[moy - 1]
    sm = s[moy - 1]
    w = 1.0 / np.maximum(np.nan_to_num(G, nan=1.0), 1.0)
    X = np.column_stack([A, A * hf, A * ef, sm, sm * tau, sm * ef])[ok] * w[ok, None]
    lb = [0, 0, 0, 0, -np.inf, 0]
    res = lsq_linear(X, G[ok] * w[ok], bounds=(lb, [np.inf] * 6))
    b, bh, be, a, _, ae = res.x
    if b <= 1e-9:
        return None
    return float(bh / b), float(be / b), float(ae / a) if a > 1e-9 else 0.0


def _calendar_arrays(p: Panels) -> tuple[np.ndarray, np.ndarray]:
    """Holiday share of days (N, T) and event share of days (T,)."""
    hf = p.nat_matrix("holiday_days") / p.nat_matrix("dim")
    ef_m = p.nat_matrix("event_days") / p.nat_matrix("dim")
    return np.nan_to_num(hf), np.nan_to_num(ef_m[0])


def _l0_mapping(known: list[str], nats: list[str]) -> np.ndarray:
    M = np.zeros((len(known), len(nats)))
    for i, o in enumerate(known):
        if o in nats:
            M[i, nats.index(o)] = 1.0
    return M


def _l1_mapping(known: list[str], units: list[str]) -> np.ndarray:
    reg = regions.region_of()
    ui = {u: k for k, u in enumerate(units)}
    M = np.zeros((len(known), len(units)))
    for i, o in enumerate(known):
        M[i, ui[reg[o]]] = 1.0
    return M


def _region_guests(G: np.ndarray, nats: list[str], units: list[str]) -> np.ndarray:
    """Region totals; months where any member nationality is missing are left NaN (not fitted)."""
    reg = regions.region_of()
    Gr = np.zeros((len(units), G.shape[1]))
    for k, u in enumerate(units):
        members = [j for j, n in enumerate(nats) if reg[n] == u]
        block = G[members]
        Gr[k] = np.where(np.isnan(block).any(axis=0), np.nan, np.nansum(block, axis=0))
    return Gr


def _region_holiday(G: np.ndarray, hf: np.ndarray, nats: list[str], units: list[str]) -> np.ndarray:
    """Guest-weighted holiday share per region (U, T)."""
    reg = regions.region_of()
    wts = np.nan_to_num(np.nanmean(G, axis=1))
    out = np.zeros((len(units), hf.shape[1]))
    for k, u in enumerate(units):
        members = [j for j, n in enumerate(nats) if reg[n] == u]
        w = wts[members]
        out[k] = (hf[members] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9)
    return out


def pooled_calendar(G, A, moy, s, hf_u, ef) -> tuple[float, float, float, dict]:
    """Guest-weighted mean of unit-level calendar uplifts (units with air supply)."""
    rows = []
    for u in range(G.shape[0]):
        est = estimate_calendar(G[u], A[u], moy, s, hf_u[u], ef)
        if est is not None:
            rows.append((np.nanmean(G[u]), *est))
    if not rows:
        return 0.0, 0.0, 0.0, {"units": 0}
    r = np.array(rows)
    w = r[:, 0] / r[:, 0].sum()
    h, ea, eb = (float(np.clip((r[:, k] * w).sum(), 0.0, 3.0)) for k in (1, 2, 3))
    return h, ea, eb, {"units": len(rows), "unit_estimates": r[:, 1:].round(3).tolist()}


def _fit_units(G, M, T, moy_t, s, cm_u, cb, row_w=None):
    """Capture + base per hotel unit, where unit u's air supply is A_u = sum_o M[o,u] T_o."""
    U = M.shape[1]
    A = M.T @ T
    ucap = np.zeros((U, 12))
    ubase = np.zeros((U, 12))
    betas = np.zeros(U)
    alphas = np.zeros(U)
    for u in range(U):
        cap, beta, alpha, _ = fit_capture_base(G[u], A[u], moy_t, s, cm=cm_u[u], cb=cb, row_w=row_w)
        ucap[u] = cap
        ubase[u] = alpha * s
        betas[u], alphas[u] = beta, alpha
    return ucap, ubase, betas, alphas


def _fill_terminating(p: Panels, known: list[str], lf: np.ndarray, tr: np.ndarray, glf, gtr) -> np.ndarray:
    """Actual terminating passengers (O_known, T); gaps filled with seats x LF x (1-TR)."""
    T = p.origin_matrix("terminating")
    S = p.origin_matrix("seats")
    moy = p.months.month.to_numpy()
    idx = {o: i for i, o in enumerate(known)}
    lf_all = np.array([lf[idx[o]] if o in idx else glf for o in p.origins])
    tr_all = np.array([tr[idx[o]] if o in idx else gtr for o in p.origins])
    est = S * lf_all[:, moy - 1] * (1 - tr_all[:, moy - 1])
    T = np.where(np.isnan(T), est, T)
    oi = {o: i for i, o in enumerate(p.origins)}
    return np.nan_to_num(T[[oi[o] for o in known]])


# ---------------------------------------------------------------------------
# main fit
# ---------------------------------------------------------------------------

def fit(p: Panels, level: str = "L1", lam: float | None = None, bootstrap: int = 0, seed: int = 42) -> ParametricModel:
    """Fit Layer 1 on the panel `p` (already restricted to the training window)."""
    if level not in LEVELS:
        raise ValueError(level)
    reg = regions.region_of()
    om = p.origin_month
    nm = p.nat_month
    moy_t = p.months.month.to_numpy()
    s = p.seasonality
    nats = p.nationalities
    N = len(nats)

    hist = om[om["seats"] > 0].groupby("origin")["month"].nunique()
    known = [o for o in p.origins if hist.get(o, 0) > 0]

    region_lf = _region_moy_means(om, "lf", "seats")
    region_tr = _region_moy_means(om, "tr", "pax")
    global_lf = _global_moy(om, "lf", "seats")
    global_tr = _global_moy(om, "tr", "pax")

    O = len(known)
    lf = np.zeros((O, 12))
    lf_se = np.zeros((O, 12))
    tr = np.zeros((O, 12))
    tr_se = np.zeros((O, 12))
    origin_n = np.zeros(O)
    hist_seats, hist_lf = {}, {}
    for i, o in enumerate(known):
        g = om[(om["origin"] == o) & (om["seats"] > 0)]
        r = reg[o]
        e_lf = eb_moy(g["lf"].to_numpy(), g["moy"].to_numpy(), prior=region_lf.get(r, global_lf))
        e_tr = eb_moy(g["tr"].to_numpy(), g["moy"].to_numpy(), prior=region_tr.get(r, global_tr))
        lf[i], lf_se[i] = np.clip(e_lf.value, 0.05, 1.05), e_lf.se
        tr[i], tr_se[i] = np.clip(e_tr.value, 0.0, 0.95), e_tr.se
        origin_n[i] = e_lf.n_total
        hist_seats[o] = (float(g["seats"].min()), float(g["seats"].max()))
        lfv = g["lf"].dropna()
        hist_lf[o] = (float(lfv.min()), float(lfv.max())) if len(lfv) else (np.nan, np.nan)

    alos = np.zeros((N, 12))
    alos_se = np.zeros((N, 12))
    for j, n in enumerate(nats):
        g = nm[(nm["nationality"] == n) & (nm["guests"] > 0)]
        e = eb_moy((g["nights"] / g["guests"]).to_numpy(), g["moy"].to_numpy())
        if e.n_total == 0:
            e = eb_moy((nm["nights"] / nm["guests"]).to_numpy(), nm["moy"].to_numpy())
        alos[j], alos_se[j] = e.value, e.se

    T = _fill_terminating(p, known, lf, tr, global_lf, global_tr)
    G = p.nat_matrix("guests")
    hf, ef = _calendar_arrays(p)

    # pooled calendar uplifts, estimated on regional units (more stable than single nationalities)
    units_r = sorted({reg[n] for n in nats})
    Mr = _l1_mapping(known, units_r)
    Gr = _region_guests(G, nats, units_r)
    hf_r = _region_holiday(G, hf, nats, units_r)
    h, ea, eb, cal_detail = pooled_calendar(Gr, Mr.T @ T, moy_t, s, hf_r, ef)
    cb = 1 + eb * ef

    kw = dict(
        origins=known, nationalities=nats, lf=lf, lf_se=lf_se, tr=tr, tr_se=tr_se, origin_n=origin_n,
        alos=alos, alos_se=alos_se, region_lf=region_lf, region_tr=region_tr, region_conv={},
        global_lf=global_lf, global_tr=global_tr, global_conv=np.zeros(12),
        cutoff=p.meta.get("cutoff", p.meta.get("last", "")), hol_uplift=h, event_uplift_air=ea,
        event_uplift_base=eb, hist_seats=hist_seats, hist_lf=hist_lf, calendar_detail=cal_detail,
    )
    ctx = {"G": G, "T": T, "moy": moy_t, "s": s, "hf": hf, "ef": ef, "cb": cb, "known": known, "p": p}

    if level == "L0":
        M = _l0_mapping(known, nats)
        cm = 1 + h * hf + ea * ef[None, :]
        ucap, ubase, _, _ = _fit_units(G, M, T, moy_t, s, cm, cb)
        conv = M[:, :, None] * ucap[None, :, :]
        out = ParametricModel(level="L0", conv=conv, base=ubase.copy(), units=nats, unit_cap=ucap,
                              unit_base=ubase, **kw)
    elif level == "L1":
        out = _fit_l1(ctx, kw)
    else:
        from skystay.reconcile import bridge

        months_avail = int((~np.isnan(G).all(axis=0)).sum())
        min_m = settings.load_settings()["reconciliation"]["min_months_for_bridge"]
        l1 = _fit_l1(ctx, kw)
        if months_avail < min_m:
            out = l1
            out.level = "L2"
            out.notes.append(f"L2 fell back to L1: only {months_avail} training months (< {min_m})")
        else:
            out = bridge.fit_l2(ctx, l1, kw, lam=lam)

    _set_region_conv(out, T, known)
    if bootstrap:
        _bootstrap(out, ctx, bootstrap, seed)
    return out


def _fit_l1(ctx: dict, kw: dict) -> ParametricModel:
    reg = regions.region_of()
    G, T, moy_t, s, hf, ef, cb, known = (ctx[k] for k in ("G", "T", "moy", "s", "hf", "ef", "cb", "known"))
    nats = kw["nationalities"]
    units = sorted({reg[n] for n in nats})
    ui = {u: k for k, u in enumerate(units)}
    M = _l1_mapping(known, units)
    Gr = _region_guests(G, nats, units)
    hf_r = _region_holiday(G, hf, nats, units)
    cm_r = 1 + kw["hol_uplift"] * hf_r + kw["event_uplift_air"] * ef[None, :]
    ucap, ubase, _, _ = _fit_units(Gr, M, T, moy_t, s, cm_r, cb)
    split = np.zeros((len(nats), 12))
    for j, n in enumerate(nats):
        share = G[j] / Gr[ui[reg[n]]]
        split[j] = eb_moy(share, moy_t).value if np.isfinite(share).any() else 0.0
    for u in units:
        members = [j for j, n in enumerate(nats) if reg[n] == u]
        tot = split[members].sum(axis=0)
        split[members] = split[members] / np.where(tot > 0, tot, 1.0)
    conv = np.zeros((len(known), len(nats), 12))
    base = np.zeros((len(nats), 12))
    for j, n in enumerate(nats):
        u = ui[reg[n]]
        conv[:, j, :] = M[:, u][:, None] * ucap[u][None, :] * split[j][None, :]
        base[j] = ubase[u] * split[j]
    m = ParametricModel(level="L1", conv=conv, base=base, units=units, unit_cap=ucap, unit_base=ubase, **kw)
    m.bridge = {"split": split, "units": units}
    return m


def _set_region_conv(m: ParametricModel, T: np.ndarray, known: list[str]) -> None:
    """Region-average total conversion per terminating passenger (for cold-start origins)."""
    reg = regions.region_of()
    tot = m.conv.sum(axis=1)  # (O, 12)
    weights = np.nansum(T, axis=1)
    m.region_conv = {}
    for r in sorted({reg[o] for o in known}):
        idx = [i for i, o in enumerate(known) if reg[o] == r]
        w = weights[idx]
        if w.sum() > 0:
            m.region_conv[r] = (tot[idx] * w[:, None]).sum(axis=0) / w.sum()
    m.global_conv = (tot * weights[:, None]).sum(axis=0) / max(weights.sum(), 1.0)


def year_block_weights(years: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Block bootstrap by calendar year expressed as row weights (a year drawn twice counts twice)."""
    uy = np.unique(years)
    counts = rng.multinomial(len(uy), np.full(len(uy), 1.0 / len(uy)))
    lookup = dict(zip(uy, counts, strict=True))
    return np.array([lookup[y] for y in years], dtype=float)


def _bootstrap(m: ParametricModel, ctx: dict, reps: int, seed: int) -> None:
    """Block bootstrap by calendar year: relative SE of annual capture level and baseline per unit."""
    rng = np.random.default_rng(seed)
    p = ctx["p"]
    years = p.months.year.to_numpy()
    G, T, moy_t, s, hf, ef, cb = (ctx[k] for k in ("G", "T", "moy", "s", "hf", "ef", "cb"))
    if m.level == "L2":
        from skystay.reconcile import bridge

        m.bridge["bootstrap"] = bridge.bootstrap(m, ctx, reps, seed)
        K = m.bridge["K"]
        Kb = m.bridge["bootstrap"]["K_samples"]
        lvl = K.sum(axis=0)
        lvl_b = Kb.sum(axis=1)
        m.unit_cap_se_rel = np.divide(lvl_b.std(axis=0), lvl, out=np.zeros_like(lvl), where=lvl > 0)
        a = m.bridge["alpha"]
        ab = m.bridge["bootstrap"]["alpha_samples"]
        m.unit_base_se_rel = np.divide(ab.std(axis=0), a, out=np.zeros_like(a), where=a > 0)
        return
    if m.level == "L0":
        M, Gu = _l0_mapping(m.origins, m.nationalities), G
        cm = 1 + m.hol_uplift * hf + m.event_uplift_air * ef[None, :]
    else:
        M, Gu = _l1_mapping(m.origins, m.units), _region_guests(G, m.nationalities, m.units)
        cm = 1 + m.hol_uplift * _region_holiday(G, hf, m.nationalities, m.units) + m.event_uplift_air * ef[None, :]
    caps = np.zeros((reps, M.shape[1]))
    bases = np.zeros((reps, M.shape[1]))
    for r in range(reps):
        w = year_block_weights(years, rng)
        _, _, caps[r], bases[r] = _fit_units(Gu, M, T, moy_t, s, cm, cb, row_w=w)
    _, _, b0, a0 = _fit_units(Gu, M, T, moy_t, s, cm, cb)
    m.unit_cap_se_rel = np.divide(caps.std(axis=0), b0, out=np.zeros_like(b0), where=b0 > 0)
    m.unit_base_se_rel = np.divide(bases.std(axis=0), a0, out=np.zeros_like(a0), where=a0 > 0)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def supply_arrays(model: ParametricModel, origins: list[str]) -> dict[str, np.ndarray]:
    """Stack LF, TR (O',12) and conversion (O',N,12) for a list of origins (cold-start aware)."""
    ps = [model.origin_params(o) for o in origins]
    no, nn = len(origins), len(model.nationalities)
    return {
        "lf": np.array([q["lf"] for q in ps]).reshape(no, 12),
        "lf_se": np.array([q["lf_se"] for q in ps]).reshape(no, 12),
        "tr": np.array([q["tr"] for q in ps]).reshape(no, 12),
        "tr_se": np.array([q["tr_se"] for q in ps]).reshape(no, 12),
        "conv": np.array([q["conv"] for q in ps]).reshape(no, nn, 12),
        "cold": np.array([q["cold"] for q in ps], dtype=bool),
    }


def calendar_matrices(model: ParametricModel, calendar: pd.DataFrame | None, months: pd.DatetimeIndex):
    """Air and base calendar multipliers (N, T) from a nationality x month calendar frame."""
    n_t = len(months)
    if calendar is None:
        return np.ones((len(model.nationalities), n_t)), np.ones(n_t)
    c = calendar.copy()
    c["hf"] = c["holiday_days"] / c["dim"]
    c["ef"] = c["event_days"] / c["dim"]
    hf = c.pivot_table(index="nationality", columns="month", values="hf").reindex(
        index=model.nationalities, columns=months).fillna(0.0).to_numpy()
    ef = c.groupby("month")["ef"].max().reindex(months).fillna(0.0).to_numpy()
    cm = 1 + model.hol_uplift * hf + model.event_uplift_air * ef[None, :]
    cb = 1 + model.event_uplift_base * ef
    return cm, cb


def predict(
    model: ParametricModel,
    seats: pd.DataFrame,
    terminating: pd.DataFrame | None = None,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict guests and nights by nationality x month.

    `seats`: long frame (month, origin, seats). If `terminating` (month, origin,
    terminating) is given, observed terminating passengers are used where
    available (in-sample fits); otherwise seats x LF x (1-TR). `calendar`:
    nationality x month frame with holiday_days, event_days, dim.
    """
    months = pd.DatetimeIndex(sorted(seats["month"].unique()))
    origins = sorted(seats["origin"].unique())
    S = seats.pivot_table(index="origin", columns="month", values="seats", aggfunc="sum").reindex(
        index=origins, columns=months).fillna(0.0).to_numpy()
    arr = supply_arrays(model, origins)
    moy = months.month.to_numpy()
    T = S * arr["lf"][:, moy - 1] * (1 - arr["tr"][:, moy - 1])
    if terminating is not None:
        Ta = terminating.pivot_table(index="origin", columns="month", values="terminating", aggfunc="sum").reindex(
            index=origins, columns=months).to_numpy()
        T = np.where(np.isnan(Ta), T, Ta)
    cm, cb = calendar_matrices(model, calendar, months)
    conv_t = arr["conv"][:, :, moy - 1]  # (O', N, T)
    air = np.einsum("ot,ont->nt", T, conv_t) * cm
    tot = conv_t.sum(axis=1, keepdims=True)
    mix = np.divide(conv_t, tot, out=np.zeros_like(conv_t), where=tot > 0)
    visitors = np.einsum("ot,ont->nt", T, mix)
    base = model.base[:, moy - 1] * cb[None, :]
    guests = air + base
    nights = guests * model.alos[:, moy - 1]
    idx = pd.MultiIndex.from_product([model.nationalities, months], names=["nationality", "month"])
    return pd.DataFrame(
        {
            "guests_pred": guests.ravel(),
            "nights_pred": nights.ravel(),
            "air_guests": air.ravel(),
            "base_guests": base.ravel(),
            "air_visitors": visitors.ravel(),
        },
        index=idx,
    ).reset_index()


def seats_frame(p: Panels, months: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    om = p.origin_month if months is None else p.origin_month[p.origin_month["month"].isin(months)]
    return om[["month", "origin", "seats"]]


def terminating_frame(p: Panels, months: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    om = p.origin_month if months is None else p.origin_month[p.origin_month["month"].isin(months)]
    return om[["month", "origin", "terminating"]]


def calendar_frame(p: Panels, months: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    nm = p.nat_month if months is None else p.nat_month[p.nat_month["month"].isin(months)]
    return nm[["nationality", "month", "holiday_days", "event_days", "dim"]]


def non_air_share(model: ParametricModel, p: Panels) -> dict[str, float]:
    """Share of each nationality's predicted guests that is non-air baseline, over the last 12 panel months.

    (The baseline level is the level at the end of the training window, so the
    share is measured on the most recent year.)
    """
    last = p.months[-12:]
    pred = predict(model, seats_frame(p, last), terminating_frame(p, last), calendar_frame(p, last))
    g = pred.groupby("nationality")[["base_guests", "guests_pred"]].sum()
    return (g["base_guests"] / g["guests_pred"].replace(0, np.nan)).fillna(1.0).to_dict()
