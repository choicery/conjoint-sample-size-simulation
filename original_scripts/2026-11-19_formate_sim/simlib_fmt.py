"""Sample-size simulation: synthetic CBC / DR-CBC / BW-CBC data estimated with the in-app
Choicery Julia HB services and simulated with the in-app MarketSimulator.

Only data generation and truth computation live here. Design generation, estimation and
market simulation call the real Choicery code paths (DesignGenerator, JuliaHBAnalysisService,
DRCBCHBService, JuliaBWCBCHBService, MarketSimulator) with app default settings.
"""
from __future__ import annotations

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
CHOICERY = HERE.parents[1] / "Choicery"
WORK = HERE / "work"
RESULTS = HERE / "results"

# App defaults (hb_orchestration CBC/BW/DR routes, app/config/limits.py, hb_config.py).
APP_HB = dict(n_chains=2, n_burn=10000, n_draws=10000, prior_variance=4.0, prior_mean=0.0,
              prior_variance_mean=100.0, seed=42, target_accept=0.30, thinning=5, df_offset=5)

DESIGNS = {
    "simple": dict(levels=[3] * 6, tasks=12, options=4),
    "medium": dict(levels=[3, 4, 4, 3, 5], tasks=10, options=3),
    "thin": dict(levels=[5] * 8, tasks=6, options=3),
}

N_TRUTH = 200_000
N_SCENARIOS = 6


# ----------------------------------------------------------------------------------------
# Population (Hein et al. 2020 / Wirth 2010 scheme)
# ----------------------------------------------------------------------------------------
def _draw_variance(rng, hetero: bool) -> float:
    if hetero:
        r1 = rng.gamma(0.7, 4.5)
        r2 = rng.uniform(0.2, 2.0)
        r3 = rng.uniform(13.0, 18.0)
    else:
        r1 = rng.gamma(0.7, 1.5)
        r2 = rng.uniform(0.08, 0.4)
        r3 = rng.uniform(9.0, 11.0)
    return float(min(r1 + r2, r3))


def _draw_mean(rng) -> float:
    u = rng.random()
    if u < 0.8:
        return float(rng.uniform(-2, 2))
    if u < 0.9:
        return float(rng.uniform(-5, -2))
    return float(rng.uniform(2, 5))


def make_population(levels, hetero: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k = sum(l - 1 for l in levels)
    means = np.array([_draw_mean(rng) for _ in range(k)])
    var = np.array([_draw_variance(rng, hetero) for _ in range(k)])
    return {"levels": list(levels), "means": means, "var": var,
            "asc_mean": None, "asc_var": float(_draw_variance(rng, hetero))}


def sample_params(pop: dict, n: int, rng) -> np.ndarray:
    """Effects-coded parameters (n x K)."""
    z = rng.standard_normal((n, len(pop["means"])))
    return pop["means"] + z * np.sqrt(pop["var"])


def sample_asc(pop: dict, n: int, rng) -> np.ndarray:
    return pop["asc_mean"] + rng.standard_normal(n) * math.sqrt(pop["asc_var"])


def full_partworths(params: np.ndarray, levels) -> list[np.ndarray]:
    """Per attribute an (n x L) matrix of part-worths, last level = -sum of the others."""
    out, pos = [], 0
    for l in levels:
        p = params[:, pos:pos + l - 1]
        out.append(np.concatenate([p, -p.sum(axis=1, keepdims=True)], axis=1))
        pos += l - 1
    return out


def profile_utilities(pw: list[np.ndarray], profiles: np.ndarray) -> np.ndarray:
    """pw: per attribute (n x L); profiles: (m x A) 0-based level codes -> (n x m)."""
    u = np.zeros((pw[0].shape[0], profiles.shape[0]))
    for a, mat in enumerate(pw):
        u += mat[:, profiles[:, a]]
    return u


def softmax(u: np.ndarray) -> np.ndarray:
    e = np.exp(u - u.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def calibrate_asc(pop: dict, options: int, target: float, seed: int) -> None:
    """Set the population mean ASC so that about `target` of random tasks end in None."""
    rng = np.random.default_rng(seed)
    m = 40_000
    params = sample_params(pop, m, rng)
    pw = full_partworths(params, pop["levels"])
    prof = np.stack([rng.integers(0, l, size=(m, options)) for l in pop["levels"]], axis=2)  # m x opt x A
    v = np.zeros((m, options))
    for a, mat in enumerate(pw):
        v += np.take_along_axis(mat, prof[:, :, a], axis=1)
    zasc = rng.standard_normal(m) * math.sqrt(pop["asc_var"])

    def share(mu):
        u = np.concatenate([v, (mu + zasc)[:, None]], axis=1)
        return softmax(u)[:, -1].mean()

    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if share(mid) < target:
            lo = mid
        else:
            hi = mid
    pop["asc_mean"] = 0.5 * (lo + hi)
    pop["asc_calibrated_share"] = float(share(pop["asc_mean"]))


# ----------------------------------------------------------------------------------------
# Design (Choicery DesignGenerator, balanced overlap = app default method)
# ----------------------------------------------------------------------------------------
def generate_design(levels, tasks, options, versions, seed, workdir: Path) -> pd.DataFrame:
    import app.services.design_generator as _dg
    from app.services.design_config import DesignConfigurator
    from app.services.design_generator import DesignGenerator

    # Harness only (16.11.2026): 1,000 versions on a busy machine can take longer than the app's
    # 300 s wall-clock guard. The algorithm and its result are unchanged, only the guard is lifted.
    _dg.MAX_DESIGN_GENERATION_SECONDS = max(_dg.MAX_DESIGN_GENERATION_SECONDS, 3600)

    cfg = workdir / "design_cfg"
    if cfg.exists():
        shutil.rmtree(cfg)
    conf = DesignConfigurator(str(cfg))
    for i, n in enumerate(levels, 1):
        conf.add_attribute(f"A{i}", [f"L{k}" for k in range(1, n + 1)])
    conf.save_configuration()
    gen = DesignGenerator(str(cfg))
    gen.set_parameters(num_participants=versions, num_choice_tasks=tasks, alternatives_per_task=options,
                       method="balanced", design_seed=int(seed))
    df = gen.generate_design()
    df = df.sort_values(["participant_id", "task_id", "alternative_id"]).reset_index(drop=True)
    return df


def design_codes(df: pd.DataFrame, n_attr: int) -> np.ndarray:
    """(versions x tasks x options x A) 0-based level codes."""
    cols = [f"A{i}" for i in range(1, n_attr + 1)]
    codes = np.stack([df[c].str[1:].astype(int).to_numpy() - 1 for c in cols], axis=1)
    v = df["participant_id"].nunique()
    t = df["task_id"].nunique()
    o = df["alternative_id"].nunique()
    return codes.reshape(v, t, o, n_attr)


# ----------------------------------------------------------------------------------------
# Response generation
# ----------------------------------------------------------------------------------------
def simulate_responses(method: str, pop: dict, params, asc, codes, rng) -> dict:
    """Returns per respondent/task answers. codes: N x T x O x A."""
    n, t, o, a = codes.shape
    pw = full_partworths(params, pop["levels"])
    v = np.zeros((n, t, o))
    for ai, mat in enumerate(pw):
        v += np.take_along_axis(mat[:, None, :], codes[:, :, :, ai].reshape(n, 1, t * o), axis=2).reshape(n, t, o)
    out = {"v": v}
    gumbel = rng.gumbel(size=(n, t, o))
    if method == "cbc":
        out["choice"] = np.argmax(v + gumbel, axis=2)
    elif method == "cbc_none":
        g_none = rng.gumbel(size=(n, t))
        u = np.concatenate([v + gumbel, (asc[:, None] + g_none)[:, :, None]], axis=2)
        out["choice"] = np.argmax(u, axis=2)  # o == None
    elif method == "drcbc":
        g_none = rng.gumbel(size=(n, t))
        u = v + gumbel
        out["choice"] = np.argmax(u, axis=2)
        best_u = np.take_along_axis(u, out["choice"][:, :, None], axis=2)[:, :, 0]
        out["buy"] = best_u > (asc[:, None] + g_none)
    elif method == "bwcbc":
        p_best = softmax(v)
        p_worst = softmax(-v)
        best = np.zeros((n, t), dtype=int)
        worst = np.zeros((n, t), dtype=int)
        redraws = 0
        for i in range(n):
            for k in range(t):
                b = rng.choice(o, p=p_best[i, k])
                w = rng.choice(o, p=p_worst[i, k])
                while w == b:
                    redraws += 1
                    w = rng.choice(o, p=p_worst[i, k])
                best[i, k], worst[i, k] = b, w
        out["best"], out["worst"], out["worst_redraws"] = best, worst, redraws
    elif method == "bwcbc_none":
        g_none = rng.gumbel(size=(n, t))
        u = v + gumbel
        best = np.argmax(u, axis=2)
        best_u = np.take_along_axis(u, best[:, :, None], axis=2)[:, :, 0]
        out["buy"] = best_u > (asc[:, None] + g_none)
        p_worst = softmax(-v)
        worst = np.zeros((n, t), dtype=int)
        redraws = 0
        for i in range(n):
            for k in range(t):
                w = rng.choice(o, p=p_worst[i, k])
                while w == best[i, k]:
                    redraws += 1
                    w = rng.choice(o, p=p_worst[i, k])
                worst[i, k] = w
        out["best"], out["worst"], out["worst_redraws"] = best, worst, redraws
    else:
        raise ValueError(method)
    return out


# ----------------------------------------------------------------------------------------
# Input frames in the exact converter formats
# ----------------------------------------------------------------------------------------
def cbc_frame(codes, resp, with_none: bool) -> pd.DataFrame:
    """ChoiceModelR format as ChoiceModelRConverter.convert_to_choicemodel_format emits it."""
    n, t, o, a = codes.shape
    rows_per_task = o + (1 if with_none else 0)
    rid = np.repeat(np.arange(1, n + 1), t * rows_per_task)
    ques = np.tile(np.repeat(np.arange(1, t + 1), rows_per_task), n)
    alt = np.tile(np.arange(1, rows_per_task + 1), n * t)
    choice = (alt.reshape(n, t, rows_per_task) - 1 == resp["choice"][:, :, None]).astype(np.int64).ravel()
    data = {"resp.id": rid, "ques": ques, "alt": alt, "choice": choice}
    for ai in range(a):
        c = codes[:, :, :, ai] + 1
        if with_none:
            c = np.concatenate([c, np.zeros((n, t, 1), dtype=c.dtype)], axis=2)
        data[f"A{ai + 1}"] = c.ravel().astype(np.int64)
    if with_none:
        data["NoneOpt"] = (alt == rows_per_task).astype(np.int64)
    return pd.DataFrame(data)


def drcbc_frame(codes, resp) -> pd.DataFrame:
    """Long format as DRCBCDataConverter._phase_build_long_format emits it."""
    n, t, o, a = codes.shape
    blk = o + 1
    rid = np.repeat(np.arange(1, n + 1), t * blk)
    task = np.tile(np.repeat(np.arange(1, t + 1), blk), n)
    alt = np.tile(np.arange(1, blk + 1), n * t)
    data = {"ID": rid, "Task": task, "Alt": alt}
    for ai in range(a):
        c = np.concatenate([codes[:, :, :, ai] + 1, np.zeros((n, t, 1), dtype=int)], axis=2)
        data[f"A{ai + 1}"] = c.ravel().astype(int)
    data["NoneOpt"] = (alt == blk).astype(int)
    data["SelectedAlt"] = np.repeat((resp["choice"] + 1).ravel(), blk).astype(int)
    data["NoneResponse"] = np.repeat(np.where(resp["buy"], 1, 2).ravel(), blk).astype(int)
    return pd.DataFrame(data)


def bwcbc_frame(codes, resp) -> pd.DataFrame:
    """Product-only single-block format as BWCBCConverter emits it (Response 1/-1/0)."""
    n, t, o, a = codes.shape
    rid = np.repeat(np.arange(1, n + 1), t * o)
    task = np.tile(np.repeat(np.arange(1, t + 1), o), n)
    concept = np.tile(np.arange(1, o + 1), n * t)
    data = {"sys_RespNum": rid, "Task": task, "Concept": concept}
    for ai in range(a):
        data[f"A{ai + 1}"] = (codes[:, :, :, ai] + 1).ravel().astype(int)
    pos = np.arange(o)[None, None, :]
    response = np.where(pos == resp["best"][:, :, None], 1, np.where(pos == resp["worst"][:, :, None], -1, 0))
    data["Response"] = response.ravel().astype(int)
    return pd.DataFrame(data)


# ----------------------------------------------------------------------------------------
# Estimation through the in-app services
# ----------------------------------------------------------------------------------------
def _consume(gen) -> dict:
    events, last_error, completed, warnings = [], None, None, []
    for ev in gen:
        if not isinstance(ev, str):
            continue
        for line in ev.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[5:].strip())
            except Exception:
                continue
            status = str(payload.get("status", ""))
            if status == "error" or "error" in payload and status in ("", "error"):
                last_error = payload
            elif status == "completed":
                completed = payload
            elif status == "warning" or payload.get("message") and "warning" in str(payload.get("status", "")):
                warnings.append(payload.get("message"))
    return {"error": last_error, "completed": completed, "warnings": warnings}


class _StubDRCBCConverter:
    """Stands in for DRCBCDataConverter: returns the synthetic long frame in its exact format."""

    long_df: pd.DataFrame = None
    attr_names: list = []
    attr_labels: dict = {}

    def __init__(self, project_dir, drcbc_id):
        self.excluded_participant_ids = set()
        self.run_excluded_participants = ()
        n = int(self.long_df["ID"].nunique())
        self.respondent_mapping = {i: f"sim{i}" for i in range(1, n + 1)}

    def convert(self, output_file=None, excluded_ids=None, *, none_required=True, allow_partial_profiles=False,
                **_):
        df = self.long_df.copy()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False, encoding="utf-8")
        n_products = int(df.loc[df["NoneOpt"] == 0].groupby(["ID", "Task"]).size().iloc[0])
        meta = {"n_respondents": int(df["ID"].nunique()), "n_tasks": int(df["Task"].nunique()),
                "n_alternatives": n_products + 1, "n_products": n_products,
                "attributes": {a: {"levels": self.attr_labels[a], "n_levels": len(self.attr_labels[a])}
                               for a in self.attr_names},
                "none_yes_count": 0, "none_no_count": 0, "warnings": []}
        return df, meta

    def get_attribute_info(self):
        attributes = {a: {"type": "categorical", "levels": self.attr_labels[a], "level_names": self.attr_labels[a],
                          "n_levels": len(self.attr_labels[a])} for a in self.attr_names}
        return {"attr_names": list(self.attr_names), "n_levels": [len(self.attr_labels[a]) for a in self.attr_names],
                "attr_labels": dict(self.attr_labels), "attr_types": {a: "categorical" for a in self.attr_names},
                "attributes": attributes}


def estimate(method: str, frame: pd.DataFrame, levels, n_tasks: int, options: int, project_dir: Path,
             run_dir: Path) -> dict:
    attr_names = [f"A{i}" for i in range(1, len(levels) + 1)]
    attr_labels = {a: [f"L{k}" for k in range(1, l + 1)] for a, l in zip(attr_names, levels)}
    common = dict(n_chains=APP_HB["n_chains"], n_burn=APP_HB["n_burn"], n_draws=APP_HB["n_draws"],
                  prior_variance=APP_HB["prior_variance"], prior_mean=APP_HB["prior_mean"],
                  prior_variance_mean=APP_HB["prior_variance_mean"], seed=APP_HB["seed"],
                  target_accept=APP_HB["target_accept"], thinning=APP_HB["thinning"],
                  df_offset=APP_HB["df_offset"], parallel_chains=False, individual_rhat=False,
                  respondent_holdout_enabled=False, respondent_holdout_fraction=0.2)
    if method in ("cbc", "cbc_none"):
        from app.services.hb_analysis_julia import JuliaHBAnalysisService

        svc = JuliaHBAnalysisService(project_dir, run_dir=run_dir)
        # Route: n_concepts = unique alternatives per task in the ChoiceModelR data (incl. None row).
        n_concepts = int(frame.groupby("ques")["alt"].nunique().iloc[0])
        gen = svc.run_hb_analysis_streaming(
            choicemodel_data=frame, attr_names=attr_names, n_levels=list(levels), attr_labels=attr_labels,
            attr_types={a: "categorical" for a in attr_names}, linear_values={}, linear_scales={},
            n_tasks=int(frame["ques"].nunique()), n_concepts=n_concepts, include_none=(method == "cbc_none"),
            constraints={}, estimation_method="full_hb_mcmc", task_id="sim", task_type="CBC",
            random_holdout_enabled=False, random_holdout_task=None, **common)
    elif method in ("bwcbc", "bwcbc_none"):
        from app.services.hb_analysis_julia_bwcbc import JuliaBWCBCHBService

        svc = JuliaBWCBCHBService(project_dir, run_dir=run_dir)
        gen = svc.run_hb_analysis_streaming(
            choicemodel_data=frame, attr_names=attr_names, n_levels=list(levels), attr_labels=attr_labels,
            attr_types={a: "categorical" for a in attr_names}, linear_values={}, linear_scales={},
            n_tasks=int(frame["Task"].nunique()), n_concepts=options, constraints=None,
            include_none_option=(method == "bwcbc_none"), estimation_tasks="both", task_id="sim",
            respondent_mapping={i: f"sim{i}" for i in range(1, int(frame["sys_RespNum"].nunique()) + 1)},
            **common)
    elif method == "drcbc":
        import app.services.drcbc_hb_julia as drmod

        _StubDRCBCConverter.long_df = frame
        _StubDRCBCConverter.attr_names = attr_names
        _StubDRCBCConverter.attr_labels = attr_labels
        drmod.DRCBCDataConverter = _StubDRCBCConverter
        svc = drmod.DRCBCHBService(project_dir, "sim", run_dir=run_dir)
        dr_common = {k: v for k, v in common.items()}
        gen = svc.run_analysis(attr_types={a: "categorical" for a in attr_names}, linear_values={},
                               linear_scales={}, constraints=None, include_none_option=True,
                               estimation_method="full_hb_mcmc", **dr_common)
    else:
        raise ValueError(method)
    return _consume(gen)


# ----------------------------------------------------------------------------------------
# Market simulation through the in-app MarketSimulator
# ----------------------------------------------------------------------------------------
def make_scenarios(levels, seed: int, n_scen: int = N_SCENARIOS) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scen = []
    while len(scen) < n_scen:
        m = int(rng.integers(3, 5))  # 3 or 4 products
        prof = np.stack([rng.integers(0, l, size=m) for l in levels], axis=1)
        if len({tuple(p) for p in prof}) < m:
            continue
        scen.append(prof)
    return scen


def true_shares(pop: dict, scen: np.ndarray, with_none: bool, params=None, asc=None) -> np.ndarray:
    """Mean logit probabilities over the given people (population if params is None)."""
    if params is None:
        rng = np.random.default_rng(12345)
        params = sample_params(pop, N_TRUTH, rng)
        asc = sample_asc(pop, N_TRUTH, rng) if with_none else None
    pw = full_partworths(params, pop["levels"])
    u = profile_utilities(pw, scen)
    if with_none:
        u = np.concatenate([u, asc[:, None]], axis=1)
    return softmax(u).mean(axis=0)


def simulator_shares(results: dict, project_dir: Path, scen: np.ndarray, with_none: bool) -> dict:
    from app.services.market_simulator import MarketSimulator

    sim = MarketSimulator(results, project_dir, task_id="sim")
    products = [{f"A{a + 1}": f"L{int(c) + 1}" for a, c in enumerate(p)} for p in scen]
    names = [f"P{i + 1}" for i in range(len(products))]
    out, warn = {}, []
    for rule, fn in (("first_choice", sim.simulate_first_choice),
                     ("share_of_preference", sim.simulate_share_of_preference),
                     ("rfc", sim.simulate_randomized_first_choice)):
        res = fn(products=products, product_names=names, include_none=with_none, return_details=False)
        sh = res["shares"]
        keys = names + ([k for k in sh if k not in names] if with_none else [])
        out[rule] = np.array([float(sh[k]) for k in keys]) / 100.0
        for k in ("missing_level_warnings", "warnings"):
            if res.get(k):
                warn.append({rule: res.get(k)})
    return {"shares": out, "warnings": warn, "has_none": bool(sim.has_none_option)}


# ----------------------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------------------
def run_one(spec: dict) -> dict:
    """spec: run_id, design, method, n, rep, hetero, none_share."""
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    logging.disable(logging.WARNING)
    if str(CHOICERY) not in sys.path:
        sys.path.insert(0, str(CHOICERY))
    run_id = spec["run_id"]
    base = WORK / run_id
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    project_dir = base / "project"
    run_dir = project_dir / "hb_runs" / "run"
    tmp_root = base / "tmp"
    for d in (run_dir, tmp_root):
        d.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp_root)  # service temp dirs stay under _scratch

    from flask import Flask

    app = Flask("sample_size_sim")
    app.config["JULIA_ANALYSIS_TIMEOUT_SECONDS"] = 4 * 3600
    rec = dict(spec)
    t_start = time.time()
    try:
        d = DESIGNS[spec["design"]]
        levels, tasks, options = d["levels"], d["tasks"], d["options"]
        method = spec["method"]
        with_none = method in ("cbc_none", "drcbc", "bwcbc_none")
        pop = make_population(levels, spec.get("hetero", True), seed=spec["pop_seed"])
        if with_none:
            calibrate_asc(pop, options, spec.get("none_share", 0.3), seed=spec["pop_seed"] + 1)
            rec["asc_mean"] = pop["asc_mean"]
            rec["asc_calibrated_share"] = pop["asc_calibrated_share"]
        rng = np.random.default_rng(spec["data_seed"])
        n = spec["n"]
        t0 = time.time()
        design = generate_design(levels, tasks, options, n, spec["data_seed"], base)
        rec["design_seconds"] = round(time.time() - t0, 1)
        codes = design_codes(design, len(levels))
        params = sample_params(pop, n, rng)
        asc = sample_asc(pop, n, rng) if with_none else None
        resp = simulate_responses(method, pop, params, asc, codes, rng)
        if method in ("cbc", "cbc_none"):
            frame = cbc_frame(codes, resp, with_none)
            if with_none:
                rec["observed_none_share"] = float((resp["choice"] == options).mean())
        elif method == "drcbc":
            frame = drcbc_frame(codes, resp)
            rec["observed_no_buy_share"] = float((~resp["buy"]).mean())
        elif method == "bwcbc_none":
            frame = bwcbc_none_frame(codes, resp)
            rec["observed_no_buy_share"] = float((~resp["buy"]).mean())
            rec["worst_redraws"] = int(resp["worst_redraws"])
        else:
            frame = bwcbc_frame(codes, resp)
            rec["worst_redraws"] = int(resp["worst_redraws"])
        frame.to_csv(base / "input_frame.csv", index=False)

        t0 = time.time()
        with app.app_context():
            outcome = estimate(method, frame, levels, tasks, options, project_dir, run_dir)
        rec["hb_seconds"] = round(time.time() - t0, 1)
        rec["hb_warnings"] = outcome["warnings"]
        if outcome["error"] or not (run_dir / "hb_results.json").exists():
            rec["status"] = "hb_failed"
            rec["error"] = outcome["error"]
            return rec
        with open(run_dir / "hb_results.json", encoding="utf-8-sig") as f:
            results = json.load(f)
        results.setdefault("task_id", "sim")
        rec["status"] = "ok"
        rec["diagnostics"] = {k: results.get(k) for k in (
            "rhat_max", "rhat_mean", "convergence", "convergence_status", "ess_min", "acceptance_rate",
            "avg_rlh", "rlh", "mean_rlh", "n_parameters", "n_respondents") if k in results}
        diag_csv = run_dir / "Diagnostics.csv"
        if diag_csv.exists():
            rec["diagnostics_csv"] = pd.read_csv(diag_csv).to_dict(orient="records")

        # Utility recovery (raw individual utilities; full-level effects coding).
        beta = results.get("beta_individual", {})
        true_pw = full_partworths(params, levels)
        cols = [f"A{a + 1}_L{k + 1}" for a, l in enumerate(levels) for k in range(l)]
        missing_cols = [c for c in cols if c not in beta]
        rec["missing_beta_cols"] = missing_cols
        if not missing_cols:
            est = np.column_stack([np.asarray(beta[c], dtype=float) for c in cols])
            tru = np.concatenate(true_pw, axis=1)
            rec["n_est_rows"] = int(est.shape[0])
            if est.shape[0] == n:
                zs = []
                for i in range(n):
                    r = np.corrcoef(est[i], tru[i])[0, 1]
                    zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
                rec["indiv_corr"] = float(np.tanh(np.mean(zs)))
                true_mean_full = np.concatenate(full_partworths(pop["means"][None, :], levels), axis=1)[0]
                rec["mean_util_rmse_vs_pop"] = float(np.sqrt(np.mean((est.mean(axis=0) - true_mean_full) ** 2)))
                rec["mean_util_rmse_vs_sample"] = float(np.sqrt(np.mean((est.mean(axis=0) - tru.mean(axis=0)) ** 2)))

        # Shares.
        scen_list = make_scenarios(levels, seed=spec["pop_seed"] + 7)
        rows = []
        for si, scen in enumerate(scen_list):
            pop_sh = true_shares(pop, scen, with_none)
            ins_sh = true_shares(pop, scen, with_none, params=params, asc=asc)
            sims = simulator_shares(results, project_dir, scen, with_none)
            if sims["warnings"]:
                rec.setdefault("simulator_warnings", []).extend(sims["warnings"])
            for rule, est_sh in sims["shares"].items():
                if len(est_sh) != len(pop_sh):
                    rec.setdefault("share_length_mismatch", []).append((si, rule, len(est_sh), len(pop_sh)))
                    continue
                for j in range(len(pop_sh)):
                    rows.append({"scenario": si, "n_products": len(scen), "alt": j,
                                 "is_none": bool(with_none and j == len(pop_sh) - 1), "rule": rule,
                                 "est": float(est_sh[j]), "pop": float(pop_sh[j]), "insample": float(ins_sh[j])})
        rec["shares"] = rows
        if with_none:
            rows_po = []
            for si, scen in enumerate(scen_list):
                pop_sh = true_shares(pop, scen, False)
                ins_sh = true_shares(pop, scen, False, params=params)
                sims = simulator_shares(results, project_dir, scen, False)
                for rule, est_sh in sims["shares"].items():
                    if len(est_sh) != len(pop_sh):
                        rec.setdefault("share_length_mismatch_po", []).append((si, rule, len(est_sh), len(pop_sh)))
                        continue
                    for j in range(len(pop_sh)):
                        rows_po.append({"scenario": si, "n_products": len(scen), "alt": j, "is_none": False, "rule": rule,
                                        "est": float(est_sh[j]), "pop": float(pop_sh[j]), "insample": float(ins_sh[j])})
            rec["shares_products_only"] = rows_po

        # Optional (16.11.2026): more markets with a fixed number of products, share of preference only.
        # spec["extra_scenarios"] = {"2": 8, "5": 4} -> 8 markets with 2 products, 4 with 5 products.
        # The six scenarios above and their random numbers stay exactly as they were.
        if spec.get("extra_scenarios"):
            from app.services.market_simulator import MarketSimulator

            sim = MarketSimulator(results, project_dir, task_id="sim")
            extra = []
            for m_key, count in sorted(spec["extra_scenarios"].items()):
                m = int(m_key)
                rng_x = np.random.default_rng(spec["pop_seed"] + 1000 + m)
                made = 0
                while made < int(count):
                    prof = np.stack([rng_x.integers(0, l, size=m) for l in levels], axis=1)
                    if len({tuple(p) for p in prof}) < m:
                        continue
                    products = [{f"A{a + 1}": f"L{int(c) + 1}" for a, c in enumerate(p)} for p in prof]
                    names = [f"P{i + 1}" for i in range(m)]
                    res = sim.simulate_share_of_preference(products=products, product_names=names,
                                                           include_none=False, return_details=False)
                    est_sh = np.array([float(res["shares"][k]) for k in names]) / 100.0
                    pop_sh = true_shares(pop, prof, False)
                    ins_sh = true_shares(pop, prof, False, params=params)
                    for j in range(m):
                        extra.append({"scenario": f"x{m}_{made}", "n_products": m, "alt": j,
                                      "rule": "share_of_preference", "est": float(est_sh[j]),
                                      "pop": float(pop_sh[j]), "insample": float(ins_sh[j])})
                    made += 1
            rec["shares_extra"] = extra
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "exception"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    finally:
        rec["total_seconds"] = round(time.time() - t_start, 1)
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / f"{run_id}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        # Keep hb_results + diagnostics, drop bulky temp data.
        shutil.rmtree(tmp_root, ignore_errors=True)
    return rec


def bwcbc_none_frame(codes, resp) -> pd.DataFrame:
    """Rows exactly as BWCBCConverter emits them with the None follow-up enabled:
    buy  -> worst block + follow-up block (all products + None, chosen = best product);
    no   -> best block + worst block + follow-up block (chosen = None)."""
    n, t, o, a = codes.shape
    rows = []
    for i in range(n):
        for k in range(t):
            b, w, buy = int(resp["best"][i, k]), int(resp["worst"][i, k]), bool(resp["buy"][i, k])
            levels = [[int(codes[i, k, j, ai]) + 1 for ai in range(a)] for j in range(o)]
            blocks = [] if buy else [("best", [1 if j == b else 0 for j in range(o)])]
            blocks.append(("worst", [-1 if j == w else 0 for j in range(o)]))
            for name, resp_vals in blocks:
                for j in range(o):
                    rows.append([i + 1, k + 1, j + 1, *levels[j], resp_vals[j], name, int(resp_vals[j] != 0), 0])
            follow = [1 if (buy and j == b) else 0 for j in range(o)] + [0 if buy else 1]
            for j in range(o + 1):
                lv = levels[j] if j < o else [0] * a
                rows.append([i + 1, k + 1, j + 1, *lv, follow[j], "none_followup", follow[j], int(j == o)])
    cols = ["sys_RespNum", "Task", "Concept"] + [f"A{ai + 1}" for ai in range(a)] + ["Response", "obs_type", "choice", "NoneOpt"]
    return pd.DataFrame(rows, columns=cols)
