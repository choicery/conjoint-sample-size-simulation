"""Willingness-to-pay simulation (21.11.2026): how exact is a willingness to pay from a CBC with Hierarchical Bayes?

Same chain as the sample-size simulation (2026-11-09_sample_size_sim/simlib.py and
2026-11-18_heterogenitaet_echte_studien/emplib.py), with one change: the last attribute is a price with a linear
effect. Design generation, estimation, market simulation and willingness to pay call the Choicery code paths
(DesignGenerator, JuliaHBAnalysisService, MarketSimulator.calculate_simulation_wtp, WTPCalculator) with app
defaults. Only the simulated people, their answers and the true values are computed here.

Simulated people
  categorical utilities   the 'tuned' generator of the formula runs (spec["gen"] = tuned_gen.json)
  price                   utility = coef_i * (price - mean price) / (half the tested price range); coef_i < 0.
                          spec["price"] = {"values": [...], "span_mean": utility difference between the lowest and
                          the highest tested price (mean over people), "cv": SD between people / mean,
                          "dist": "lognormal" (nobody prefers a higher price) or "normal"}

True values (never from app code)
  market willingness to pay   price change of the focal product at which its true share of preference is back at
                              the share before the improvement; for the population (N_TRUTH people) and for the
                              respondents of the run (their true utilities)
  utility-ratio value         utility difference / price slope per person; mean and median over people

Per run the estimation result of the app (hb_results/<run_id>.json.gz) and the estimated and true individual
utilities (utilities/<run_id>.npz) are stored, so that further questions can be answered later without estimating
again.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / "_scratch" / "2026-11-09_sample_size_sim"
HET = ROOT / "_scratch" / "2026-11-18_heterogenitaet_echte_studien"
CHOICERY = ROOT / "Choicery"
for _p in (str(SIM), str(SIM / "grid"), str(HET)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PRICE = "Price"
N_TRUTH = 200_000
N_TRUTH_ROBUST = 40_000        # people for the true value of every focal profile of the robust variant
N_SHARE_MARKETS = 6
SEARCH_RANGES = 10.0           # the app searches up to 10 tested price ranges away (sim_wtp_max_price_range_multiples)


def price_label(value: float) -> str:
    return f"{value:g}"


# ----------------------------------------------------------------------------------------
# Simulated people
# ----------------------------------------------------------------------------------------
def make_population(spec: dict) -> dict:
    import emplib

    pop = emplib.tuned_population(spec["levels"], spec, int(spec["pop_seed"]))
    p = spec["price"]
    values = np.asarray(p["values"], float)
    mean = float(p["span_mean"]) / 2.0     # mean size of the coefficient on the scaled price (-1 to 1)
    pop["price"] = dict(values=values, center=float(values.mean()), half=float(values.max() - values.min()) / 2.0,
                        mean=mean, sd=mean * float(p["cv"]), dist=str(p.get("dist", "lognormal")))
    return pop


def sample_people(pop: dict, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """(effects-coded categorical parameters n x K, price coefficient on the scaled price n)."""
    params = pop["means"] + rng.standard_normal((n, len(pop["means"]))) * np.sqrt(pop["var"])
    pr = pop["price"]
    z = rng.standard_normal(n)
    if pr["dist"] == "lognormal":
        s2 = math.log(1.0 + (pr["sd"] / pr["mean"]) ** 2)
        coef = -np.exp(math.log(pr["mean"]) - s2 / 2.0 + math.sqrt(s2) * z)
    elif pr["dist"] == "normal":
        coef = -(pr["mean"] + pr["sd"] * z)
    else:
        raise ValueError(pr["dist"])
    return params, coef


def partworths(params: np.ndarray, levels) -> list[np.ndarray]:
    out, pos = [], 0
    for l in levels:
        p = params[:, pos:pos + l - 1]
        out.append(np.concatenate([p, -p.sum(axis=1, keepdims=True)], axis=1))
        pos += l - 1
    return out


def softmax(u: np.ndarray) -> np.ndarray:
    e = np.exp(u - u.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def market_utilities(pw, coef, pop, cat: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """cat: m x A level codes, prices: m price values -> n x m true utilities."""
    u = np.zeros((coef.shape[0], cat.shape[0]))
    for a, mat in enumerate(pw):
        u += mat[:, cat[:, a]]
    pr = pop["price"]
    return u + coef[:, None] * ((np.asarray(prices, float) - pr["center"]) / pr["half"])[None, :]


# ----------------------------------------------------------------------------------------
# Design, answers, input frame
# ----------------------------------------------------------------------------------------
def generate_design(levels, price_values, tasks, options, versions, seed, workdir: Path) -> np.ndarray:
    """Choicery DesignGenerator (balanced overlap = app default) -> versions x tasks x options x (A + 1) codes."""
    import app.services.design_generator as _dg
    from app.services.design_config import DesignConfigurator
    from app.services.design_generator import DesignGenerator

    # Harness only: the wall-clock guard of the app is lifted, algorithm and result are unchanged.
    _dg.MAX_DESIGN_GENERATION_SECONDS = max(_dg.MAX_DESIGN_GENERATION_SECONDS, 3600)
    cfg = workdir / "design_cfg"
    if cfg.exists():
        shutil.rmtree(cfg)
    conf = DesignConfigurator(str(cfg))
    for i, n in enumerate(levels, 1):
        conf.add_attribute(f"A{i}", [f"L{k}" for k in range(1, n + 1)])
    labels = [price_label(v) for v in price_values]
    conf.add_attribute(PRICE, labels)
    conf.save_configuration()
    gen = DesignGenerator(str(cfg))
    gen.set_parameters(num_participants=versions, num_choice_tasks=tasks, alternatives_per_task=options,
                       method="balanced", design_seed=int(seed))
    df = gen.generate_design().sort_values(["participant_id", "task_id", "alternative_id"]).reset_index(drop=True)
    cols = [df[f"A{i}"].str[1:].astype(int).to_numpy() - 1 for i in range(1, len(levels) + 1)]
    cols.append(df[PRICE].astype(str).map({lab: k for k, lab in enumerate(labels)}).to_numpy())
    assert not pd.isna(cols[-1]).any(), sorted(df[PRICE].astype(str).unique())
    codes = np.stack(cols, axis=1).astype(int)
    return codes.reshape(df["participant_id"].nunique(), df["task_id"].nunique(), df["alternative_id"].nunique(),
                         len(levels) + 1)


def simulate_choices(pop, params, coef, codes, rng) -> np.ndarray:
    n, t, o, a = codes.shape
    v = np.zeros((n, t, o))
    for ai, mat in enumerate(partworths(params, pop["levels"])):
        v += np.take_along_axis(mat[:, None, :], codes[:, :, :, ai].reshape(n, 1, t * o), axis=2).reshape(n, t, o)
    pr = pop["price"]
    v += coef[:, None, None] * ((pr["values"] - pr["center"]) / pr["half"])[codes[:, :, :, a - 1]]
    return np.argmax(v + rng.gumbel(size=(n, t, o)), axis=2)


def cbc_frame(codes, choice, attr_names) -> pd.DataFrame:
    """ChoiceModelR format as ChoiceModelRConverter.convert_to_choicemodel_format emits it."""
    n, t, o, a = codes.shape
    alt = np.tile(np.arange(1, o + 1), n * t)
    data = {"resp.id": np.repeat(np.arange(1, n + 1), t * o), "ques": np.tile(np.repeat(np.arange(1, t + 1), o), n),
            "alt": alt, "choice": (alt.reshape(n, t, o) - 1 == choice[:, :, None]).astype(np.int64).ravel()}
    for ai, name in enumerate(attr_names):
        data[name] = (codes[:, :, :, ai] + 1).ravel().astype(np.int64)
    return pd.DataFrame(data)


def estimate(frame, levels, price_values, project_dir: Path, run_dir: Path, hb: dict, price_constraint: str | None) -> dict:
    from app.services.hb_analysis_julia import JuliaHBAnalysisService
    from simlib import _consume

    names = [f"A{i}" for i in range(1, len(levels) + 1)] + [PRICE]
    labels = {f"A{i}": [f"L{k}" for k in range(1, l + 1)] for i, l in enumerate(levels, 1)}
    labels[PRICE] = [price_label(v) for v in price_values]
    types = {a: "categorical" for a in names}
    types[PRICE] = "linear"
    svc = JuliaHBAnalysisService(project_dir, run_dir=run_dir)
    gen = svc.run_hb_analysis_streaming(
        choicemodel_data=frame, attr_names=names, n_levels=list(levels) + [len(price_values)], attr_labels=labels,
        attr_types=types, linear_values={PRICE: [float(v) for v in price_values]}, linear_scales={PRICE: 1.0},
        n_tasks=int(frame["ques"].nunique()), n_concepts=int(frame.groupby("ques")["alt"].nunique().iloc[0]),
        include_none=False, constraints=({PRICE: price_constraint} if price_constraint else {}),
        estimation_method="full_hb_mcmc", task_id="sim", task_type="CBC", random_holdout_enabled=False,
        random_holdout_task=None, n_chains=hb["n_chains"], n_burn=hb["n_burn"], n_draws=hb["n_draws"],
        prior_variance=hb["prior_variance"], prior_mean=hb["prior_mean"], prior_variance_mean=hb["prior_variance_mean"],
        seed=hb["seed"], target_accept=hb["target_accept"], thinning=hb["thinning"], df_offset=hb["df_offset"],
        parallel_chains=False, individual_rhat=False, respondent_holdout_enabled=False, respondent_holdout_fraction=0.2)
    return _consume(gen)


# ----------------------------------------------------------------------------------------
# True willingness to pay
# ----------------------------------------------------------------------------------------
def true_market_wtp(pw, coef, pop, cat, prices, focal: int, attr: int, to_level: int):
    """Price change of the focal product that brings its true share back to the share before the improvement.

    Returns (wtp or None, base share, improved share). The search follows the app: it walks away from the current
    price in growing steps (at most SEARCH_RANGES tested ranges) until the share crosses the base share; the root is
    then found inside that bracket. Share of the focal product per person = 1 / (1 + exp(-(u_focal - log sum exp of
    the other products))), so a price change of the focal product only shifts one number per person.
    """
    from scipy.optimize import brentq
    from scipy.special import expit, logsumexp

    pr = pop["price"]
    u = market_utilities(pw, coef, pop, cat, prices)
    others = [j for j in range(u.shape[1]) if j != focal]
    d0 = u[:, focal] - logsumexp(u[:, others], axis=1)
    base = float(expit(d0).mean())
    d1 = d0 + pw[attr][:, to_level] - pw[attr][:, cat[focal, attr]]
    slope = coef / pr["half"]

    def gap(dp: float) -> float:
        return float(expit(d1 + slope * dp).mean()) - base

    g0 = gap(0.0)
    improved = base + g0
    if abs(g0) < 1e-9:
        return 0.0, base, improved
    rng_ = 2.0 * pr["half"]
    direction = 1.0 if g0 > 0 else -1.0
    step, lo, hi = 0.25 * rng_, 0.0, None
    while abs(lo) < SEARCH_RANGES * rng_:
        cand = lo + direction * step
        if gap(cand) * g0 <= 0:
            hi = cand
            break
        lo, step = cand, step * 1.8
    if hi is None:
        return None, base, improved
    return float(brentq(gap, min(lo, hi), max(lo, hi), xtol=1e-5)), base, improved


def true_ratio_wtp(pw, coef, pop, attr: int, from_level: int, to_level: int) -> dict:
    """Utility difference divided by the price slope, per person (people with a slope of zero or above are left out)."""
    slope = coef / pop["price"]["half"]
    ok = slope < 0
    w = (pw[attr][ok, to_level] - pw[attr][ok, from_level]) / (-slope[ok])
    return {"mean": float(w.mean()), "median": float(np.median(w)), "n": int(ok.sum())}


# ----------------------------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------------------------
def make_wtp_questions(pop: dict, seed: int, count: int) -> list[dict]:
    """Markets of 3 or 4 products; product 0 is improved in one attribute (from the level with the lower to the
    level with the higher mean utility in the population). Product 0 starts at the second tested price, so that
    most break-even prices stay inside the tested prices."""
    rng = np.random.default_rng(seed)
    levels = pop["levels"]
    means = partworths(pop["means"][None, :], levels)
    n_price = len(pop["price"]["values"])
    out = []
    while len(out) < count:
        m = int(rng.integers(3, 5))
        cat = np.stack([rng.integers(0, l, size=m) for l in levels], axis=1)
        price_idx = rng.integers(0, n_price, size=m)
        price_idx[0] = 1
        attr = int(rng.integers(0, len(levels)))
        a, b = (int(x) for x in rng.choice(levels[attr], size=2, replace=False))
        lo, hi = (a, b) if means[attr][0, a] <= means[attr][0, b] else (b, a)
        cat[0, attr] = lo
        if len({tuple(r) + (int(p),) for r, p in zip(cat, price_idx)}) < m:
            continue
        out.append({"cat": cat, "price_idx": price_idx, "attr": attr, "from": lo, "to": hi,
                    "mean_utility_gain": float(means[attr][0, hi] - means[attr][0, lo])})
    return out


def make_share_markets(pop: dict, seed: int, count: int = N_SHARE_MARKETS) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    levels, n_price, out = pop["levels"], len(pop["price"]["values"]), []
    while len(out) < count:
        m = int(rng.integers(3, 5))
        cat = np.stack([rng.integers(0, l, size=m) for l in levels], axis=1)
        price_idx = rng.integers(0, n_price, size=m)
        if len({tuple(r) + (int(p),) for r, p in zip(cat, price_idx)}) < m:
            continue
        out.append((cat, price_idx))
    return out


def products_of(cat, price_idx, price_values) -> list[dict]:
    return [{**{f"A{a + 1}": f"L{int(c) + 1}" for a, c in enumerate(row)}, PRICE: price_label(price_values[int(p)])}
            for row, p in zip(cat, price_idx)]


def _pick(res: dict) -> dict:
    keys = ("wtp", "wtp_status", "base_share", "improved_share", "break_even_price", "is_boundary_estimate",
            "is_extrapolated", "boundary_wtp", "iterations", "n_simulation_respondents",
            "n_positive_price_beta_excluded", "n_near_zero_price_beta_excluded", "effective_method",
            "negative_price_bound_applied", "robust_focal_profiles_n_profiles_used",
            "robust_focal_profiles_n_profiles_identified", "robust_focal_profiles_n_profiles_boundary",
            "robust_focal_profiles_median_wtp")
    return {k: res.get(k) for k in keys if k in res}


# ----------------------------------------------------------------------------------------
# Evaluation of one estimated run
# ----------------------------------------------------------------------------------------
def evaluate(spec: dict, results: dict, project_dir: Path, pop: dict, params, coef, rec: dict) -> None:
    from app.services.market_simulator import MarketSimulator
    from app.services.wtp_calculator import WTPCalculator

    levels = pop["levels"]
    price_values = pop["price"]["values"]
    n = int(spec["n"])
    pw = partworths(params, levels)
    rng_pop = np.random.default_rng(12345)
    pop_params, pop_coef = sample_people(pop, N_TRUTH, rng_pop)
    pop_pw = partworths(pop_params, levels)
    rob_pw = [m[:N_TRUTH_ROBUST] for m in pop_pw]
    rob_coef = pop_coef[:N_TRUTH_ROBUST]

    # Estimated individual utilities in a fixed column order; price as slope per price unit.
    beta = results.get("beta_individual", {})
    cat_cols = [f"A{a + 1}_L{k + 1}" for a, l in enumerate(levels) for k in range(l)]
    price_cols = [f"{PRICE}_{price_label(v)}" for v in price_values]
    missing = [c for c in cat_cols + price_cols if c not in beta]
    rec["missing_beta_cols"] = missing
    rec["beta_columns"] = sorted(beta.keys())[:60]
    if missing:
        rec["status"] = "missing_columns"
        return
    est_cat = np.column_stack([np.asarray(beta[c], float) for c in cat_cols])
    est_price = np.column_stack([np.asarray(beta[c], float) for c in price_cols])
    rec["n_est_rows"] = int(est_cat.shape[0])
    if est_cat.shape[0] != n:
        rec["status"] = "row_mismatch"
        return
    half = pop["price"]["half"]
    est_slope = (est_price[:, -1] - est_price[:, 0]) / (price_values[-1] - price_values[0])
    true_slope = coef / half
    tru_cat = np.concatenate(pw, axis=1)
    rec["price_recovery"] = {
        "true_slope_mean": float(true_slope.mean()), "est_slope_mean": float(est_slope.mean()),
        "true_slope_median": float(np.median(true_slope)), "est_slope_median": float(np.median(est_slope)),
        "true_slope_sd": float(true_slope.std(ddof=1)), "est_slope_sd": float(est_slope.std(ddof=1)),
        "corr": float(np.corrcoef(true_slope, est_slope)[0, 1]),
        "est_share_positive": float((est_slope >= 0).mean()),
        "true_share_positive": float((true_slope >= 0).mean()),
        "pop_slope_mean": float((pop_coef / half).mean()), "pop_slope_median": float(np.median(pop_coef / half))}
    zs = [np.arctanh(np.clip(np.corrcoef(est_cat[i], tru_cat[i])[0, 1], -0.999999, 0.999999)) for i in range(n)]
    rec["indiv_corr"] = float(np.tanh(np.mean(zs)))
    util_dir = Path(rec["_results_dir"]) / "utilities"
    util_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(util_dir / f"{spec['run_id']}.npz", est_cat=est_cat.astype(np.float32),
                        est_price=est_price.astype(np.float32), true_params=params.astype(np.float32),
                        true_coef=coef.astype(np.float32))

    sim = MarketSimulator(results, project_dir, task_id="sim")
    calc = WTPCalculator(project_dir, task_id="sim", results_payload=results)
    timing = {"sop": 0.0, "all": 0.0, "fc": 0.0, "rfc": 0.0, "robust": 0.0, "ratio": 0.0, "truth": 0.0}
    rows = []
    questions = make_wtp_questions(pop, seed=int(spec["pop_seed"]) + 21, count=int(spec.get("wtp_questions", 10)))
    for qi, q in enumerate(questions):
        cat, price_idx, attr = q["cat"], q["price_idx"], q["attr"]
        prices = price_values[price_idx]
        products = products_of(cat, price_idx, price_values)
        names = [f"P{i + 1}" for i in range(len(products))]
        row = {"question": qi, "n_products": len(products), "attribute": attr, "from": q["from"], "to": q["to"],
               "mean_utility_gain": q["mean_utility_gain"], "focal_price": float(prices[0])}
        t0 = time.time()
        for label, people in (("pop", (pop_pw, pop_coef)), ("ins", (pw, coef))):
            w, base, improved = true_market_wtp(people[0], people[1], pop, cat, prices, 0, attr, q["to"])
            row[f"true_{label}"] = w
            row[f"true_{label}_base_share"] = base
            row[f"true_{label}_improved_share"] = improved
            row[f"true_ratio_{label}"] = true_ratio_wtp(people[0], people[1], pop, attr, q["from"], q["to"])
        timing["truth"] += time.time() - t0

        def ask(key, **kw):
            t1 = time.time()
            try:
                res = sim.calculate_simulation_wtp(products=products, product_names=names, focal_product_idx=0,
                                                   attribute=f"A{attr + 1}", to_level=f"L{q['to'] + 1}",
                                                   price_attribute=PRICE, include_none=False, **kw)
                row[key] = _pick(res)
            except Exception as exc:  # noqa: BLE001
                row[key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            timing[key] += time.time() - t1

        ask("sop", method="share_of_preference")                                   # app default
        ask("all", method="share_of_preference", include_invalid_price_betas=True)  # nobody left out
        ask("fc", method="first_choice")
        if qi < int(spec.get("wtp_questions_rfc", 2)):   # the slow rule; the stored estimation result allows more later
            ask("rfc", method="randomized_first_choice")
        if qi < int(spec.get("wtp_questions_robust", 5)):
            ask("robust", method="share_of_preference", robust_focal_profiles=True)
            t1 = time.time()
            try:
                meta = sim._generate_valid_focal_profiles(focal_product=products[0], attribute=f"A{attr + 1}",
                                                          to_level=f"L{q['to'] + 1}", price_attribute=PRICE,
                                                          include_price=False, vary_levels=True)
                vals, missing_truth = [], 0
                for prof in meta.get("profiles") or []:
                    c2 = cat.copy()
                    c2[0] = [int(str(prof[f"A{a + 1}"])[1:]) - 1 for a in range(len(levels))]
                    w, _, _ = true_market_wtp(rob_pw, rob_coef, pop, c2, prices, 0, attr, q["to"])
                    if w is None:
                        missing_truth += 1
                    else:
                        vals.append(w)
                row["true_pop_robust"] = float(np.mean(vals)) if vals else None
                row["true_pop_robust_profiles"] = len(vals)
                row["true_pop_robust_not_found"] = missing_truth
            except Exception as exc:  # noqa: BLE001
                row["true_pop_robust_error"] = f"{type(exc).__name__}: {exc}"[:300]
            timing["truth"] += time.time() - t1
        t1 = time.time()
        try:
            r = calc.calculate_level_upgrade_wtp(f"A{attr + 1}", f"L{q['from'] + 1}", f"L{q['to'] + 1}",
                                                 price_attribute=PRICE)
            if r.get("error"):
                row["ratio"] = {"error": str(r["error"])[:300]}
            else:
                st = r.get("statistics") or r.get("stats") or {}
                row["ratio"] = {k: st.get(k) for k in ("mean", "median", "std", "percentile_5", "percentile_95",
                                                         "n_respondents")}
                row["ratio"]["mode"] = r.get("mode")
                if not st:
                    row["ratio"]["keys"] = sorted(r.keys())[:40]
        except Exception as exc:  # noqa: BLE001
            row["ratio"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        timing["ratio"] += time.time() - t1
        rows.append(row)
    rec["wtp"] = rows

    # Shares of preference, first choice and randomized first choice in markets with a price, as in the formula runs.
    t0 = time.time()
    share_rows = []
    for si, (cat, price_idx) in enumerate(make_share_markets(pop, seed=int(spec["pop_seed"]) + 7)):
        prices = price_values[price_idx]
        products = products_of(cat, price_idx, price_values)
        names = [f"P{i + 1}" for i in range(len(products))]
        pop_sh = softmax(market_utilities(pop_pw, pop_coef, pop, cat, prices)).mean(axis=0)
        ins_sh = softmax(market_utilities(pw, coef, pop, cat, prices)).mean(axis=0)
        for rule, fn in (("first_choice", sim.simulate_first_choice), ("share_of_preference", sim.simulate_share_of_preference),
                         ("rfc", sim.simulate_randomized_first_choice)):
            res = fn(products=products, product_names=names, include_none=False, return_details=False)
            for j, name in enumerate(names):
                share_rows.append({"scenario": si, "n_products": len(names), "alt": j, "rule": rule,
                                   "est": float(res["shares"][name]) / 100.0, "pop": float(pop_sh[j]),
                                   "insample": float(ins_sh[j])})
    rec["shares"] = share_rows
    timing["shares"] = time.time() - t0
    rec["eval_seconds"] = {k: round(v, 1) for k, v in timing.items()}


# ----------------------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------------------
def run_one(spec: dict) -> dict:
    """spec: run_id, levels (categorical attributes), tasks, options, n, price, gen, pop_seed, data_seed, _results."""
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    logging.disable(logging.WARNING)
    if str(CHOICERY) not in sys.path:
        sys.path.insert(0, str(CHOICERY))
    import simlib

    results_dir = HERE / spec["_results"]
    work = HERE / ("work_" + spec["_results"])
    run_id = spec["run_id"]
    base = work / run_id
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    project_dir = base / "project"
    run_dir = project_dir / "hb_runs" / "run"
    tmp_root = base / "tmp"
    for d in (run_dir, tmp_root):
        d.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp_root)   # service temp dirs stay under _scratch

    from flask import Flask

    app = Flask("wtp_sim")
    app.config["JULIA_ANALYSIS_TIMEOUT_SECONDS"] = 4 * 3600
    rec = dict(spec)
    rec["_results_dir"] = str(results_dir)
    t_start = time.time()
    try:
        levels, tasks, options, n = list(spec["levels"]), int(spec["tasks"]), int(spec["options"]), int(spec["n"])
        pop = make_population(spec)
        price_values = pop["price"]["values"]
        rng = np.random.default_rng(int(spec["data_seed"]))
        t0 = time.time()
        codes = generate_design(levels, price_values, tasks, options, n, int(spec["data_seed"]), base)
        rec["design_seconds"] = round(time.time() - t0, 1)
        params, coef = sample_people(pop, n, rng)
        choice = simulate_choices(pop, params, coef, codes, rng)
        names = [f"A{i}" for i in range(1, len(levels) + 1)] + [PRICE]
        frame = cbc_frame(codes, choice, names)
        frame.to_csv(base / "input_frame.csv", index=False)
        hb = dict(simlib.APP_HB)
        hb["seed"] = int(spec.get("hb_seed", 42))
        if spec.get("prior_variance"):   # arm with another prior variance than the app default
            hb["prior_variance"] = float(spec["prior_variance"])
        rec["hb_prior_variance"] = hb["prior_variance"]
        t0 = time.time()
        with app.app_context():
            outcome = estimate(frame, levels, price_values, project_dir, run_dir, hb, spec.get("price_constraint"))
        rec["hb_seconds"] = round(time.time() - t0, 1)
        rec["hb_warnings"] = outcome["warnings"]
        if outcome["error"] or not (run_dir / "hb_results.json").exists():
            rec["status"] = "hb_failed"
            rec["error"] = outcome["error"]
            return rec
        with open(run_dir / "hb_results.json", encoding="utf-8-sig") as f:
            results = json.load(f)
        results.setdefault("task_id", "sim")
        (results_dir / "hb_results").mkdir(parents=True, exist_ok=True)
        with gzip.open(results_dir / "hb_results" / f"{run_id}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(results, f)
        rec["status"] = "ok"
        rec["diagnostics"] = {k: results.get(k) for k in (
            "rhat_max", "rhat_mean", "convergence", "convergence_status", "ess_min", "acceptance_rate", "avg_rlh",
            "rlh", "mean_rlh", "n_parameters", "n_respondents") if k in results}
        t0 = time.time()
        evaluate(spec, results, project_dir, pop, params, coef, rec)
        rec["evaluate_seconds"] = round(time.time() - t0, 1)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "exception"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    finally:
        rec["total_seconds"] = round(time.time() - t_start, 1)
        rec.pop("_results_dir", None)
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        shutil.rmtree(tmp_root, ignore_errors=True)
        if rec.get("status") == "ok" and not spec.get("keep_work"):
            shutil.rmtree(base, ignore_errors=True)
    return rec


def cost(s: dict) -> float:
    return s["n"] * s["tasks"] * (sum(l - 1 for l in s["levels"]) + 1)
