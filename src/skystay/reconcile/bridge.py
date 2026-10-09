"""L2: a learned origin -> nationality bridge.

Model (per nationality n, month t):

    G_n(t) = m_n(moy) * cal_n(t) * sum_o K[o,n] * T_o(t)
             + s(moy) * (alpha_n + gamma_n * tau_t) * (1 + e_b * evt(t))

K[o,n] = CAP_n * B[o,n] * (1 - RES_o) is "hotel guests of nationality n per
terminating passenger from origin o". CAP_n, B[o,n] and RES_o are not
separately identified from aggregated data (returning residents are folded
into capture, so rows of K do not sum to a known constant). We therefore
report K, its row-normalised mix (the nationality mix of hotel guests each
origin generates) and its row sums (guests per terminating passenger), and
say so.

- bounded least squares (K >= 0, alpha >= 0), one nationality at a time
- only plausible pairs: same region, plus listed hub origins (all origins for
  the "Others" bucket; none for UAE nationals, who are non-air by definition)
- ridge penalty pulling K toward the L1 regional prior
- lambda chosen on the last 12 training months (time-ordered validation)
- uncertainty from a block bootstrap by calendar year (final model only)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from skystay import settings
from skystay.chain.shrink import eb_moy
from skystay.reconcile import regions

LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def plausible_matrix(origins: list[str], nats: list[str], hubs: list[str] | None = None) -> np.ndarray:
    hubs = hubs if hubs is not None else settings.load_settings()["reconciliation"]["hub_origins"]
    reg = regions.region_of()
    P = np.zeros((len(origins), len(nats)), dtype=bool)
    for j, n in enumerate(nats):
        for i, o in enumerate(origins):
            if n == "AE":
                continue
            if n == "OTHER" or reg[o] == reg[n] or o in hubs:
                P[i, j] = True
    return P


def fit_K(ctx: dict, m_hat: np.ndarray, cm: np.ndarray, K0: np.ndarray, P: np.ndarray, lam: float,
          row_w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge-to-prior bounded least squares for K (O,N) and baseline (alpha, gamma) per nationality."""
    from skystay.chain.parametric import TREND_RIDGE, _trend_ridge, time_tau

    G, T, moy_t, s, cb = ctx["G"], ctx["T"], ctx["moy"], ctx["s"], ctx["cb"]
    n_o, N = K0.shape
    n_t = G.shape[1]
    tau = time_tau(n_t)
    rw = np.ones(n_t) if row_w is None else np.sqrt(np.asarray(row_w, dtype=float))
    sb = s[moy_t - 1] * cb
    K = np.zeros((n_o, N))
    alpha = np.zeros(N)
    gamma = np.zeros(N)
    for j in range(N):
        y = G[j]
        ok = ~np.isnan(y) & (rw > 0)
        if ok.sum() < 12:
            continue
        idx = np.where(P[:, j] & (T[:, ok].sum(axis=1) > 0))[0]
        Xa = (T[idx] * (m_hat[j, moy_t - 1] * cm[j])[None, :]).T
        X = np.column_stack([Xa, sb, sb * tau])
        w = rw / np.maximum(np.nan_to_num(y, nan=1.0), 1.0)
        Xw = X[ok] * w[ok, None]
        yw = y[ok] * w[ok]
        k = len(idx)
        if k:
            c = np.sqrt(np.mean(Xw[:, :k] ** 2, axis=0))
            scale = np.sqrt(lam * ok.sum())
            R = np.zeros((k, X.shape[1]))
            R[np.arange(k), np.arange(k)] = scale * c
            Xw = np.vstack([Xw, R])
            yw = np.concatenate([yw, scale * c * K0[idx, j]])
        Xw, yw = _trend_ridge(Xw, yw, TREND_RIDGE)
        lb = [0.0] * k + [0.0, -np.inf]
        res = lsq_linear(Xw, yw, bounds=(lb, [np.inf] * len(lb)))
        K[idx, j] = res.x[:k]
        alpha[j], gamma[j] = res.x[k], res.x[k + 1]
    return K, alpha, gamma


def _predict_cols(K, alpha, m_hat, cm, ctx, cols):
    T, moy_t, s, cb = ctx["T"], ctx["moy"], ctx["s"], ctx["cb"]
    mt = moy_t[cols]
    air = m_hat[:, mt - 1] * cm[:, cols] * (K.T @ T[:, cols])
    return air + alpha[:, None] * (s[mt - 1] * cb[cols])[None, :]


def _wmape(G: np.ndarray, pred: np.ndarray) -> float:
    ok = ~np.isnan(G)
    return float(np.abs(G[ok] - pred[ok]).sum() / max(np.abs(G[ok]).sum(), 1e-9))


def select_lambda(ctx, m_hat, cm, K0, P, horizon: int = 12) -> tuple[float, dict]:
    """Pick the ridge strength on the last `horizon` training months (time-ordered validation)."""
    n_t = ctx["G"].shape[1]
    w = np.ones(n_t)
    w[n_t - horizon:] = 0.0
    va = np.arange(n_t - horizon, n_t)
    scores = {}
    for lam in LAMBDAS:
        K, a, g = fit_K(ctx, m_hat, cm, K0, P, lam, row_w=w)
        # baseline level at the validation start = alpha + gamma * (12 months earlier than cutoff offset)
        a_va = np.maximum(a + g * (horizon / 12.0), 0.0)
        scores[lam] = _wmape(ctx["G"][:, va], _predict_cols(K, a_va, m_hat, cm, ctx, va))
    best = min(scores, key=scores.get)
    return best, scores


def prior_matrix(ctx: dict, l1, kw: dict) -> np.ndarray:
    """Diagonal-dominant prior: the L0 identity capture where a nationality has its own origin,
    the L1 regional prior otherwise (e.g. nationalities only reached through hubs or neighbours)."""
    from skystay.chain.parametric import _fit_units, _l0_mapping

    known, nats = kw["origins"], kw["nationalities"]
    M0 = _l0_mapping(known, nats)
    cm = 1 + kw["hol_uplift"] * ctx["hf"] + kw["event_uplift_air"] * ctx["ef"][None, :]
    ucap, _, _, _ = _fit_units(ctx["G"], M0, ctx["T"], ctx["moy"], ctx["s"], cm, ctx["cb"])
    K_l0 = M0 * ucap.mean(axis=1)[None, :]
    K_l1 = l1.conv.mean(axis=2)
    own = M0.sum(axis=0) > 0
    return np.where(own[None, :], K_l0, K_l1)


def fit_l2(ctx: dict, l1, kw: dict, lam: float | None = None):
    """Build the L2 ParametricModel from the L1 fit (prior + seasonal capture shape)."""
    from skystay.chain.parametric import ParametricModel

    G, T, moy_t, s, hf, ef = (ctx[k] for k in ("G", "T", "moy", "s", "hf", "ef"))
    nats = kw["nationalities"]
    known = kw["origins"]
    reg = regions.region_of()
    ui = {u: k for k, u in enumerate(l1.units)}
    m_hat = np.ones((len(nats), 12))
    for j, n in enumerate(nats):
        c = l1.unit_cap[ui[reg[n]]]
        if c.mean() > 0:
            m_hat[j] = c / c.mean()
    cm = 1 + kw["hol_uplift"] * hf + kw["event_uplift_air"] * ef[None, :]
    K0 = prior_matrix(ctx, l1, kw)
    P = plausible_matrix(known, nats)
    scores = {}
    if lam is None:
        lam, scores = select_lambda(ctx, m_hat, cm, K0, P)
    K, alpha, gamma = fit_K(ctx, m_hat, cm, K0, P, lam)
    # refine the seasonal capture shape per nationality given K (keeps mean 1; level stays in K)
    from skystay.chain.parametric import time_tau

    tau = time_tau(G.shape[1])
    sb = s[moy_t - 1] * ctx["cb"]
    for j in range(len(nats)):
        A = (K[:, j] @ T) * cm[j]
        ok = (A > 0) & ~np.isnan(G[j])
        if ok.sum() >= 12:
            ratio = (G[j, ok] - sb[ok] * (alpha[j] + gamma[j] * tau[ok])) / A[ok]
            est = eb_moy(ratio, moy_t[ok]).value
            if est.mean() > 0:
                shape = np.clip(est / est.mean(), 0.25, 4.0)
                m_hat[j] = shape / shape.mean()
    conv = K[:, :, None] * m_hat[None, :, :]
    base = alpha[:, None] * s[None, :]
    m = ParametricModel(level="L2", conv=conv, base=base, units=list(nats), unit_cap=m_hat,
                        unit_base=base.copy(), **kw)
    m.bridge = {"K": K, "K0": K0, "plausible": P, "lambda": lam, "lambda_scores": scores, "m_hat": m_hat,
                "alpha": alpha, "gamma": gamma, "cm": cm}
    return m


def bootstrap(model, ctx: dict, reps: int = 100, seed: int = 42) -> dict:
    """Block bootstrap by calendar year: samples and 10th/90th percentiles of K and of its row-normalised mix."""
    from skystay.chain.parametric import year_block_weights

    br = model.bridge
    years = ctx["p"].months.year.to_numpy()
    rng = np.random.default_rng(seed)
    Ks, As = [], []
    for _ in range(reps):
        w = year_block_weights(years, rng)
        K, a, _ = fit_K(ctx, br["m_hat"], br["cm"], br["K0"], br["plausible"], br["lambda"], row_w=w)
        Ks.append(K)
        As.append(a)
    Ks = np.array(Ks)
    rows = Ks.sum(axis=2, keepdims=True)
    mixes = np.divide(Ks, rows, out=np.zeros_like(Ks), where=rows > 0)
    return {
        "K_samples": Ks, "alpha_samples": np.array(As),
        "K_lo": np.percentile(Ks, 10, axis=0), "K_hi": np.percentile(Ks, 90, axis=0),
        "mix_lo": np.percentile(mixes, 10, axis=0), "mix_hi": np.percentile(mixes, 90, axis=0),
        "reps": reps,
    }


def mix_matrix(K: np.ndarray) -> np.ndarray:
    rows = K.sum(axis=1, keepdims=True)
    return np.divide(K, rows, out=np.zeros_like(K), where=rows > 0)
