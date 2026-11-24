"""Night runs of 21./22.11.2026 after the Codex review (BERICHT.md in 2026-11-21_codex_review).

Three blocks. Estimation, design generation, market simulation and willingness to pay are the Choicery code paths, as
in the earlier simulations (simlib / emplib / wtplib); only the points named here differ.

chains      Does the error of a share come from chains that are too short, from the start value of the chains or from
            the prior? Every cell (design, respondents, spread of the people, population) is answered ONCE (same
            tasks, same answers, checked by a hash of the answers) and estimated with several settings:
              d4      prior variance 4.0, 10,000 + 10,000 iterations (the app default)
              d1      prior variance 1.0
              d025    prior variance 0.25
              d4lo    prior variance 4.0, but the chains start where prior variance 0.25 would start them
              d025hi  prior variance 0.25, but the chains start where prior variance 4.0 would start them
              m4, m1  prior variance 4.0 / 1.0 with 80,000 + 20,000 iterations (the most the HB page allows)
              x4, x1  prior variance 4.0 / 1.0 with 400,000 + 100,000 iterations (reference, a few cells only)
            The start value is the only change to the estimation script of the app (one line, harness only): the app
            starts the covariance between people at the prior covariance.
designpath  The earlier simulations called the design generator without an overlap strategy (old percent path, seed
            without effect). The design page uses overlap strategy 'balanced' for new projects. Same people and
            markets, tasks from the page path and, as the noise reference, a second draw from the old path.
addon       Willingness to pay for an add-on that people agree on (the pizza example of the user): one attribute
            with two levels (without / with), mean utility gain 0.30, and three levels of agreement between people.

Results -> results_<block>/<run_id>.json (+ utilities/<run_id>.npz)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

NIGHT = Path(__file__).resolve().parent
ROOT = NIGHT.parents[1]
SIM = ROOT / "_scratch" / "2026-11-09_sample_size_sim"
HET = ROOT / "_scratch" / "2026-11-18_heterogenitaet_echte_studien"
WTP = ROOT / "_scratch" / "2026-11-21_wtp_simulation"
CHOICERY = ROOT / "Choicery"
for _p in (str(SIM), str(SIM / "grid"), str(HET), str(WTP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Julia is started through its program file, not through the launcher in WindowsApps (24.11.2026). Every start through
# the launcher makes the App Installer check for an update of the Julia package; with many starts at once these checks
# failed and each failure opened an App Installer window on the user's screen (1,926 during the blocks ext1 and ext2).
# The launcher only picks this file, so the estimation is identical (test_julia_direkt.py).
_JULIA_BIN = Path.home() / ".julia" / "juliaup" / "julia-1.12.3+0.x64.w64.mingw32" / "bin"
if (_JULIA_BIN / "julia.exe").exists() and str(_JULIA_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_JULIA_BIN) + os.pathsep + os.environ.get("PATH", "")

# P core 0 (logical 0, 1) and the E cores 16 to 19 and 24 to 31 stay free for the user's own work.
CORES = list(range(2, 16)) + list(range(20, 24))
if os.environ.get("NIGHT_MORE_CORES"):      # night of 23.11.2026, allowed by the user: all but P core 0
    CORES = list(range(2, 32))
OTHER_QUEUE = "other_queue.py"   # jobs of another queue of the user, each pinned to two logical processors


def free_cores() -> list[int]:
    """CORES without the processors the other queue has pinned its running jobs to."""
    import psutil

    busy = set()
    for p in psutil.process_iter(["cmdline"]):
        try:
            if OTHER_QUEUE in " ".join(p.info["cmdline"] or []):
                aff = p.cpu_affinity()
                if len(aff) <= 4:
                    busy.update(aff)
        except Exception:  # noqa: BLE001
            pass
    cores = [c for c in CORES if c not in busy]
    return cores if len(cores) >= 6 else CORES


def pin() -> None:
    try:
        import psutil

        proc = psutil.Process()
        if psutil.cpu_count() == 32:
            proc.cpu_affinity(free_cores())
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)   # the Julia child inherits both
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------------------
# Harness patches (per worker process; the values that change per run are set in the work functions)
# ----------------------------------------------------------------------------------------
_STATE = {"start_factor": None, "frame_sha": None, "addon": None}
_START_LINE = "    Sigma = copy(prior_covariance)\n"


def install_start_patch() -> None:
    """Lets a run start the covariance between people at a multiple of the prior covariance (app: factor 1)."""
    if str(CHOICERY) not in sys.path:
        sys.path.insert(0, str(CHOICERY))
    from app.services.hb_analysis_julia import JuliaHBAnalysisService as Service

    if getattr(Service, "_night_patched", False):
        return
    original = Service._generate_julia_script

    def patched(self, *args, **kwargs):
        script = original(self, *args, **kwargs)
        assert isinstance(script, str) and script.count(_START_LINE) == 1, "start line of the chains not found exactly once"
        factor = _STATE["start_factor"]
        if factor is not None:
            script = script.replace(_START_LINE, f"    Sigma = {float(factor)!r} * copy(prior_covariance)\n")
        return script

    Service._generate_julia_script = patched
    Service._night_patched = True


def _frame_sha(frame) -> str:
    import pandas as pd

    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()[:16]


def _design(levels, tasks, options, versions, seed, workdir: Path, path: str, price_labels=None):
    """Choicery DesignGenerator. path 'page': overlap strategy 'balanced' as the design page sets it for a new project
    (the seed works). path 'legacy': no strategy, as in the earlier simulations; that path draws from Python's global
    random generator, which is seeded here so that a run can be repeated."""
    import random

    import app.services.design_generator as _dg
    from app.services.design_config import DesignConfigurator
    from app.services.design_generator import DesignGenerator

    _dg.MAX_DESIGN_GENERATION_SECONDS = max(_dg.MAX_DESIGN_GENERATION_SECONDS, 3600)   # harness only, as before
    cfg = workdir / "design_cfg"
    if cfg.exists():
        shutil.rmtree(cfg)
    conf = DesignConfigurator(str(cfg))
    for i, n in enumerate(levels, 1):
        conf.add_attribute(f"A{i}", [f"L{k}" for k in range(1, n + 1)])
    if price_labels is not None:
        conf.add_attribute("Price", list(price_labels))
    conf.save_configuration()
    gen = DesignGenerator(str(cfg))
    extra = {"overlap_strategy": "balanced"} if path == "page" else {}
    assert path in ("page", "legacy"), path
    gen.set_parameters(num_participants=versions, num_choice_tasks=tasks, alternatives_per_task=options,
                       method="balanced", design_seed=int(seed), **extra)
    if path == "legacy":
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
    return gen.generate_design().sort_values(["participant_id", "task_id", "alternative_id"]).reset_index(drop=True)


def _diagnostics(run_dir: Path) -> dict:
    """R-hat and effective sample size by group from Parameter_Diagnostics.csv of the estimation."""
    path = run_dir / "Parameter_Diagnostics.csv"
    if not path.exists():
        return {}
    groups = {"alpha": [], "sigma_diag": [], "sigma_off": []}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rhat, ess = float(row["Rhat"]), float(row["BulkESS"])
            except (TypeError, ValueError):
                continue
            if row["Group"] == "Alpha":
                key = "alpha"
            else:
                a, _, b = row["Parameter"].partition(" :: ")
                key = "sigma_diag" if a.strip() == b.strip() else "sigma_off"
            groups[key].append((rhat, ess))
    out = {}
    for key, vals in groups.items():
        if vals:
            r, e = np.array(vals).T
            out[key] = {"n": len(vals), "rhat_max": float(np.nanmax(r)), "rhat_median": float(np.nanmedian(r)),
                        "ess_min": float(np.nanmin(e)), "ess_median": float(np.nanmedian(e))}
    return out


def _write(path: Path, rec: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                    encoding="utf-8")


# ----------------------------------------------------------------------------------------
# Blocks chains and designpath: shares, the pipeline of the formula runs (emplib.run -> simlib.run_one)
# ----------------------------------------------------------------------------------------
def work_shares(spec: dict) -> dict:
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import emplib
    import simlib

    emplib.pin_to_efficiency_cores = pin
    emplib.HERE = NIGHT                       # results and work folders of this night
    install_start_patch()
    _STATE["start_factor"] = spec.get("sigma_start_factor")
    _STATE["frame_sha"] = None
    simlib.APP_HB["prior_variance"] = float(spec.get("prior_variance", 4.0))
    path = spec["design_path"]
    simlib.generate_design = lambda levels, tasks, options, versions, seed, workdir: _design(
        levels, tasks, options, versions, seed, workdir, path)
    if not getattr(simlib, "_night_hash", False):
        original_frame = simlib.cbc_frame

        def cbc_frame(codes, resp, with_none):
            frame = original_frame(codes, resp, with_none)
            _STATE["frame_sha"] = _frame_sha(frame)
            return frame

        simlib.cbc_frame = cbc_frame
        simlib._night_hash = True

    real_shutil = emplib.shutil               # keep the estimation result on disk for the extra numbers
    emplib.shutil = types.SimpleNamespace(rmtree=lambda *a, **k: None)
    try:
        rec = emplib.run(spec)
    finally:
        emplib.shutil = real_shutil
    results_dir = NIGHT / spec["_results"]
    base = NIGHT / ("work_" + spec["_results"]) / spec["run_id"]
    run_dir = base / "project" / "hb_runs" / "run"
    rec["frame_sha"] = _STATE["frame_sha"]
    try:
        if rec.get("status") == "ok" and (run_dir / "hb_results.json").exists():
            results = json.loads((run_dir / "hb_results.json").read_text(encoding="utf-8-sig"))
            for key in ("rhat_alpha_max", "rhat_sigma_max", "ess_mean", "rlh_mean", "iterations", "burn_in"):
                if key in results:
                    rec["diagnostics"][key] = results[key]
            rec["diagnostics_groups"] = _diagnostics(run_dir)
            pop = emplib.tuned_population(spec["levels"], spec, spec["pop_seed"])
            beta = results["beta_individual"]
            cols = [f"A{a + 1}_L{k + 1}" for a, l in enumerate(spec["levels"]) for k in range(l)]
            free = [f"A{a + 1}_L{k + 1}" for a, l in enumerate(spec["levels"]) for k in range(l - 1)]
            est = np.column_stack([np.asarray(beta[c], float) for c in cols])
            est_free = np.column_stack([np.asarray(beta[c], float) for c in free])
            true_sd, true_mean = np.sqrt(pop["var"]), pop["means"]
            rec["sd_ratio"] = float(est_free.std(axis=0, ddof=1).mean() / true_sd.mean())
            rec["mean_slope"] = float(np.polyfit(true_mean, est_free.mean(axis=0), 1)[0])
            (results_dir / "utilities").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(results_dir / "utilities" / f"{spec['run_id']}.npz", est=est.astype(np.float32))
    except Exception as exc:  # noqa: BLE001
        rec["extra_error"] = f"{type(exc).__name__}: {exc}"
    _write(results_dir / f"{spec['run_id']}.json", rec)
    if rec.get("status") == "ok":
        shutil.rmtree(base, ignore_errors=True)
    return rec


DEFAULT_LENGTH = dict(n_burn=10000, n_draws=10000, thinning=5)
PAGE_MAX = dict(n_burn=80000, n_draws=20000, thinning=10)      # the HB page allows 100,000 iterations per chain
REFERENCE = dict(n_burn=400000, n_draws=100000, thinning=50)
VARIANTS = {
    "d4": dict(prior_variance=4.0, **DEFAULT_LENGTH),
    "d1": dict(prior_variance=1.0, **DEFAULT_LENGTH),
    "d025": dict(prior_variance=0.25, **DEFAULT_LENGTH),
    "d4lo": dict(prior_variance=4.0, sigma_start_factor=0.0625, **DEFAULT_LENGTH),
    "d025hi": dict(prior_variance=0.25, sigma_start_factor=16.0, **DEFAULT_LENGTH),
    "m4": dict(prior_variance=4.0, **PAGE_MAX),
    "m1": dict(prior_variance=1.0, **PAGE_MAX),
    "x4": dict(prior_variance=4.0, **REFERENCE),
    "x1": dict(prior_variance=1.0, **REFERENCE),
}
MAIN_VARIANTS = ("d4", "d1", "d025", "d4lo", "d025hi", "m4", "m1")
# (design, respondents, spread, population)
CELLS_SPREAD = [(d, 1000, s, p) for d in ("med", "a5l4t12o3", "a4l4t8o2", "a5l5t8o2") for s in (0.55, 1.0, 1.7) for p in (0, 1)]
CELLS_N = [(d, n, 1.0, p) for d in ("med", "a5l5t8o2") for n in (400, 2500) for p in (0, 1)]
CELLS_CONTROL = [("a4l3t16o4", 1000, s, 0) for s in (0.55, 1.0)]      # chains already fine with the default
CELLS_REFERENCE = [(d, 1000, s, 0) for d in ("med", "a5l5t8o2") for s in (0.55, 1.0)]
CALIBRATION_DESIGNS = ("a5l5t8o2", "a4l4t8o2", "a8l3t12o2", "a5l5t10o3", "m33445t8o3", "med", "a5l4t12o3", "a6l3t8o4",
                       "a7l3t14o3", "a4l4t10o4", "a5l3t12o4", "a4l3t16o4")


def _twins() -> dict:
    """Runs of the formula simulation (results_tuned), by (design, respondents, population, data set)."""
    out = {}
    for path in sorted((HET / "results_tuned").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["design"] in CALIBRATION_DESIGNS:
            out[(r["design"], r["n"], r["pop"], r["rep"])] = r
    return out


def _shares_spec(twin: dict, **changes) -> dict:
    keep = ("design", "levels", "tasks", "options", "n", "rep", "pop", "pop_seed", "data_seed", "hb_seed", "extra_scenarios")
    spec = {k: twin[k] for k in keep}
    spec.update(method="cbc", hetero="tuned", hetero_label="tuned", none_share=None, twin=twin["run_id"])
    spec.update(changes)
    return spec


def cost(s: dict) -> float:
    params = sum(l - 1 for l in s["levels"]) + (1 if s.get("price") else 0)
    return s["n"] * s["tasks"] * s["options"] * params * (s.get("n_burn", 10000) + s.get("n_draws", 10000))


def plan_chains() -> list[dict]:
    gen = json.loads((HET / "tuned_gen.json").read_text(encoding="utf-8"))
    twins = _twins()
    out = []

    def add(cell, variants):
        design, n, spread, pop = cell
        twin = twins[(design, min(n, 1000), pop, 0)]
        assert twin["gen"] == gen and twin["status"] == "ok"
        seed = twin["data_seed"] if n <= 1000 else twin["data_seed"] + 500_000   # no formula run has 2,500 respondents
        cell_id = f"{design}_p{pop}_n{n}_s{round(spread * 100):03d}"
        for v in variants:
            out.append(_shares_spec(twin, run_id=f"{cell_id}_{v}", cell=cell_id, variant=v, n=n, data_seed=seed, spread=spread,
                                    gen={**gen, "sd_median": gen["sd_median"] * spread}, design_path="legacy",
                                    _results="results_chains", stage="chains", **VARIANTS[v]))

    for cell in CELLS_SPREAD + CELLS_N + CELLS_CONTROL:
        add(cell, MAIN_VARIANTS)
    for cell in CELLS_REFERENCE:
        add(cell, ("x4", "x1"))
    assert len({s["run_id"] for s in out}) == len(out)
    return sorted(out, key=lambda s: -cost(s))      # long runs first, so that no long run is left over at the end


def plan_designpath() -> list[dict]:
    gen = json.loads((HET / "tuned_gen.json").read_text(encoding="utf-8"))
    twins = _twins()
    out = []
    for design in CALIBRATION_DESIGNS:
        for n in (400, 1000):
            for pop in range(4):
                twin = twins[(design, n, pop, 0)]
                for path in ("page", "legacy"):
                    out.append(_shares_spec(twin, run_id=f"{twin['run_id']}_{path}", cell=twin["run_id"], variant=path, spread=1.0,
                                            gen=gen, design_path=path, _results="results_designpath", stage="designpath",
                                            prior_variance=4.0, **DEFAULT_LENGTH))
    return sorted(out, key=lambda s: -cost(s))


# ----------------------------------------------------------------------------------------
# Block addon: willingness to pay for an add-on people agree on (wtplib.run_one)
# ----------------------------------------------------------------------------------------
ADDON_LEVELS = [3, 4, 4, 2]          # the add-on is the last categorical attribute: L1 = with, L2 = without
ADDON_ATTR = 3
ADDON_GAIN_MEAN = 0.30               # mean utility gain of the add-on; about 1.5 price units at the typical price effect
ADDON_AGREEMENT = {"agree": 0.10, "mixed": 0.30, "split": 1.90}   # SD of the gain between people
ADDON_NS = (150, 400, 1000, 2000)
ADDON_POPULATIONS = 8
ADDON_REPS = 2
PRICES = [10.0, 15.0, 20.0, 25.0, 30.0]


def _addon_population(spec: dict) -> dict:
    import wtplib

    pop = wtplib._night_make_population(spec)
    a = spec["addon"]
    pos = sum(l - 1 for l in spec["levels"][:a["attr"]])
    n_levels = spec["levels"][a["attr"]]
    pop["means"], pop["var"] = pop["means"].copy(), pop["var"].copy()
    if n_levels == 2:
        pop["means"][pos] = a["gain_mean"] / 2.0          # two levels: +b and -b, so the gain is 2b
        pop["var"][pos] = (a["gain_sd"] / 2.0) ** 2
    else:
        # More levels (block addon5, 23.11.2026): the priced step goes from level 2 to level 1 of the attribute.
        # These two levels carry the gain and its spread between people; the other levels keep the utilities of
        # the generator (with three levels the third one is minus the sum of the two, so it stays near zero).
        assert n_levels >= 3 and "cov" not in pop
        pop["means"][pos], pop["means"][pos + 1] = a["gain_mean"] / 2.0, -a["gain_mean"] / 2.0
        pop["var"][pos] = pop["var"][pos + 1] = a["gain_sd"] ** 2 / 2.0
    return pop


def _addon_questions(pop: dict, seed: int, count: int) -> list[dict]:
    """Markets of 3 or 4 products; product 0 is offered without the add-on at the second tested price and gets it."""
    import wtplib

    attr = _STATE["addon"]["attr"]
    rng = np.random.default_rng(seed)
    levels, n_price = pop["levels"], len(pop["price"]["values"])
    means = wtplib.partworths(pop["means"][None, :], levels)
    out = []
    while len(out) < count:
        m = int(rng.integers(3, 5))
        cat = np.stack([rng.integers(0, l, size=m) for l in levels], axis=1)
        price_idx = rng.integers(0, n_price, size=m)
        price_idx[0] = 1
        cat[0, attr] = 1
        if len({tuple(r) + (int(p),) for r, p in zip(cat, price_idx)}) < m:
            continue
        out.append({"cat": cat, "price_idx": price_idx, "attr": attr, "from": 1, "to": 0,
                    "mean_utility_gain": float(means[attr][0, 0] - means[attr][0, 1])})
    return out


def _wtp_design(levels, price_values, tasks, options, versions, seed, workdir: Path) -> np.ndarray:
    import pandas as pd
    import wtplib

    labels = [wtplib.price_label(v) for v in price_values]
    df = _design(levels, tasks, options, versions, seed, workdir, _STATE["design_path"], price_labels=labels)
    cols = [df[f"A{i}"].str[1:].astype(int).to_numpy() - 1 for i in range(1, len(levels) + 1)]
    cols.append(df["Price"].astype(str).map({lab: k for k, lab in enumerate(labels)}).to_numpy())
    assert not pd.isna(cols[-1]).any()
    codes = np.stack(cols, axis=1).astype(int)
    return codes.reshape(df["participant_id"].nunique(), df["task_id"].nunique(), df["alternative_id"].nunique(), len(levels) + 1)


def work_addon(spec: dict) -> dict:
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    pin()
    if str(CHOICERY) not in sys.path:
        sys.path.insert(0, str(CHOICERY))
    import simlib
    import wtplib

    if not getattr(wtplib, "_night_patched", False):
        wtplib._night_make_population = wtplib.make_population
        wtplib.make_population = _addon_population
        wtplib.make_wtp_questions = _addon_questions
        wtplib.generate_design = _wtp_design
        original_frame = wtplib.cbc_frame

        def cbc_frame(codes, choice, attr_names):
            frame = original_frame(codes, choice, attr_names)
            _STATE["frame_sha"] = _frame_sha(frame)
            return frame

        wtplib.cbc_frame = cbc_frame
        wtplib.HERE = NIGHT
        wtplib._night_patched = True
    install_start_patch()
    _STATE.update(start_factor=spec.get("sigma_start_factor"), frame_sha=None, addon=spec["addon"],
                  design_path=spec.get("design_path", "page"))
    for key, default in (("n_burn", 10000), ("n_draws", 10000), ("thinning", 5)):
        simlib.APP_HB[key] = int(spec.get(key, default))
    rec = wtplib.run_one(spec)
    rec["frame_sha"] = _STATE["frame_sha"]
    _write(NIGHT / spec["_results"] / f"{spec['run_id']}.json", rec)
    return rec


def plan_addon() -> list[dict]:
    gen = json.loads((HET / "tuned_gen.json").read_text(encoding="utf-8"))
    out = []
    for name, gain_sd in ADDON_AGREEMENT.items():
        for pop in range(ADDON_POPULATIONS):
            for n in ADDON_NS:
                for rep in range(ADDON_REPS):
                    pop_seed = 9900 + pop      # the same people, markets and other attributes at every level of agreement
                    out.append(dict(
                        run_id=f"addon_{name}_p{pop}_n{n}_r{rep}", arm=name, design="addon", levels=list(ADDON_LEVELS), tasks=10,
                        options=3, method="cbc", n=n, rep=rep, pop=pop, gen=gen,
                        price={"values": PRICES, "span_mean": 4.0, "cv": 0.87, "dist": "lognormal"}, price_constraint=None,
                        prior_variance=None, hb_seed=42, pop_seed=pop_seed, data_seed=pop_seed * 10 + 1000 * rep + n,
                        addon={"attr": ADDON_ATTR, "gain_mean": ADDON_GAIN_MEAN, "gain_sd": gain_sd}, design_path="page",
                        wtp_questions=10, wtp_questions_rfc=0, wtp_questions_robust=0, _results="results_addon",
                        stage="addon", **DEFAULT_LENGTH))
    return sorted(out, key=lambda s: -cost(s))


PLANS = {"chains": (plan_chains, work_shares), "designpath": (plan_designpath, work_shares), "addon": (plan_addon, work_addon)}
