"""Robust price premium (Sampling of Scenarios) on stored runs: the app value against the true value.

Why (22.11.2026): the user wants a calculator that does not depend on the prior variance; its target becomes the
willingness to pay. The Codex review proposed the robust premium of the app as the target: for a change of one level
the app draws many random competitive situations, finds in each the price premium at which the product is back at its
old share, and reports the median. No new estimation is needed: the estimation result of every earlier run is stored.

App value   MarketSimulator.calculate_sos_wtp with the app defaults (300 scenarios, 4 random competitors drawn from the
            levels of the design, share of preference, focal profile not varied) on the stored estimation result.
True value  no app code: for the very same scenarios the price change that brings the TRUE share of the product back,
            from the true utilities of M_TRUTH people of the population (wtplib.true_market_wtp); median over the
            scenarios. A scenario without a true break-even inside the search range counts as 'not found'.
Check       the single-market premium of question 0 is computed again and compared with the value stored by the run,
            which shows that the rebuilt simulator is the one the run used.

A task = all runs of one population (same simulated people, same questions), so the true values are computed once.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WTP = ROOT / "_scratch" / "2026-11-21_wtp_simulation"
NIGHT = ROOT / "_scratch" / "2026-11-21_nachtlauf"
FAMILIES = {"wtp": WTP / "results_wtp", "addon": NIGHT / "results_addon", "addon2": NIGHT / "results_addon2",
            "addon3": NIGHT / "results_addon3", "addon4": NIGHT / "results_addon4", "addon5": NIGHT / "results_addon5",
            "addon6": NIGHT / "results_addon6", "addon7": NIGHT / "results_addon7",
            "addon8": NIGHT / "results_addon8", "addon9": NIGHT / "results_addon9",
            # other question formats (dual-response, best-worst, best-worst with a buy question), 23.11.2026:
            # _scratch/2026-11-23_formate_aufschlag
            "fmt_smoke": NIGHT / "results_fmt_smoke", "fmt1": NIGHT / "results_fmt1",
            # night block of 23.11.2026 (lean designs, crossed designs, levels x agreement): plan_ext1.py
            "ext1": NIGHT / "results_ext1",
            # 24.11.2026: 50, 100 and 250 respondents for the crossed designs (plan_ext2.py)
            "ext2": NIGHT / "results_ext2"}
RESULTS = HERE / "results_sos"
if os.environ.get("SOS_SMOKE"):      # trial runs of the blocks addon4 and addon5: own families, own result folder
    FAMILIES = {"addon4_smoke": NIGHT / "results_addon4_smoke", "addon5_smoke": NIGHT / "results_addon5_smoke",
                "ext1_smoke": NIGHT / "results_smoke_addon10"}
    RESULTS = HERE / "results_sos_smoke"
M_TRUTH = 20_000
N_QUESTIONS = 3
SOS = dict(n_iterations=300, n_competitors=4, competitor_mode="random_from_design", method="share_of_preference")


def pop_key(rec: dict) -> str:
    """Runs with the same key have the same simulated population and the same questions."""
    parts = {k: rec.get(k) for k in ("levels", "gen", "price", "pop_seed", "addon")}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def list_runs(families=None) -> list[dict]:
    out = []
    for fam, folder in FAMILIES.items():
        if families and fam not in families:
            continue
        for path in sorted(folder.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("status") != "ok" or not rec.get("wtp"):
                continue
            if not (folder / "hb_results" / f"{rec['run_id']}.json.gz").exists():
                continue
            out.append({"family": fam, "run_id": rec["run_id"], "key": pop_key(rec), "n": rec["n"], "arm": rec.get("arm"),
                        "design": rec.get("design")})
    return out


def _setup():
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    logging.disable(logging.WARNING)
    if str(NIGHT) not in sys.path:
        sys.path.insert(0, str(NIGHT))
    import nightlib

    if str(nightlib.CHOICERY) not in sys.path:
        sys.path.insert(0, str(nightlib.CHOICERY))
    nightlib.pin()
    tmp = HERE / "tmp" / str(os.getpid())
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)      # nothing is written into the app folder
    return nightlib, tmp


def population(rec: dict, wtplib) -> dict:
    pop = wtplib.make_population(rec)
    a = rec.get("addon")
    if a:                             # as nightlib._addon_population
        pos = sum(l - 1 for l in rec["levels"][:a["attr"]])
        pop["means"], pop["var"] = pop["means"].copy(), pop["var"].copy()
        if rec["levels"][a["attr"]] == 2:
            pop["means"][pos] = a["gain_mean"] / 2.0
            pop["var"][pos] = (a["gain_sd"] / 2.0) ** 2
        else:                         # more levels (block addon5): the step from level 2 to level 1 carries the gain
            assert "cov" not in pop
            pop["means"][pos], pop["means"][pos + 1] = a["gain_mean"] / 2.0, -a["gain_mean"] / 2.0
            pop["var"][pos] = pop["var"][pos + 1] = a["gain_sd"] ** 2 / 2.0
    return pop


def questions(rec: dict, pop: dict, nightlib, wtplib) -> list[dict]:
    seed = int(rec["pop_seed"]) + 21
    if rec.get("addon"):
        nightlib._STATE["addon"] = rec["addon"]
        return nightlib._addon_questions(pop, seed, N_QUESTIONS)
    return wtplib.make_wtp_questions(pop, seed=seed, count=N_QUESTIONS)


def work(task: dict) -> dict:
    """task: {'key': population key, 'runs': [(family, run_id), ...]} -> one result file per run."""
    t_task = time.time()
    done, failed = 0, []
    try:
        nightlib, tmp = _setup()
        import wtplib
        from app.services.market_simulator import MarketSimulator

        RESULTS.mkdir(parents=True, exist_ok=True)
        truth_cache: dict = {}
        people = None
        for fam, run_id in task["runs"]:
            target = RESULTS / f"{fam}__{run_id}.json"
            if target.exists():
                continue
            t0 = time.time()
            try:
                folder = FAMILIES[fam]
                rec = json.loads((folder / f"{run_id}.json").read_text(encoding="utf-8"))
                with gzip.open(folder / "hb_results" / f"{run_id}.json.gz", "rt", encoding="utf-8") as f:
                    results = json.load(f)
                pop = population(rec, wtplib)
                levels, price_values = pop["levels"], pop["price"]["values"]
                if people is None:
                    params, coef = wtplib.sample_people(pop, M_TRUTH, np.random.default_rng(12345))
                    people = (wtplib.partworths(params, levels), coef)
                qs = questions(rec, pop, nightlib, wtplib)
                sim = MarketSimulator(results, tmp, task_id="sim")
                rows = []
                for qi, q in enumerate(qs):
                    products = wtplib.products_of(q["cat"], q["price_idx"], price_values)
                    names = [f"P{i + 1}" for i in range(len(products))]
                    attribute, to_level = f"A{q['attr'] + 1}", f"L{q['to'] + 1}"
                    row = {"question": qi, "attribute": q["attr"], "from": q["from"], "to": q["to"],
                           "mean_utility_gain": q["mean_utility_gain"], "focal_price": float(price_values[q["price_idx"][0]])}
                    if qi == 0:       # the rebuilt simulator against the value the run stored
                        again = sim.calculate_simulation_wtp(
                            products=products, product_names=names, focal_product_idx=0, attribute=attribute, to_level=to_level,
                            price_attribute=wtplib.PRICE, include_none=False, method="share_of_preference")
                        row["single_again"] = again.get("wtp")
                        row["single_stored"] = ((rec["wtp"][0].get("sop") or {}).get("wtp"))
                    row["single_true"] = rec["wtp"][qi].get("true_pop")
                    row["single_app"] = (rec["wtp"][qi].get("sop") or {}).get("wtp")
                    t1 = time.time()
                    res = sim.calculate_sos_wtp(
                        focal_product=products[0], focal_product_name="P1", attribute=attribute, to_level=to_level,
                        price_attribute=wtplib.PRICE, include_none=False, collect_iteration_details=True, **SOS)
                    row["app_seconds"] = round(time.time() - t1, 1)
                    details = res.get("iteration_details") or []
                    for k in ("median_wtp", "mean_wtp", "se", "std", "ci_lower", "ci_upper", "n_successful", "n_failed",
                              "n_boundary", "n_extrapolated", "n_simulation_respondents", "n_positive_price_beta_excluded",
                              "n_near_zero_price_beta_excluded"):
                        row[f"app_{k}"] = res.get(k)
                    app_w = np.array([d["wtp"] if d["status"] == "success" and d["wtp"] is not None else np.nan for d in details], float)
                    scen = [d["scenario_products_json"] for d in details]
                    sha = hashlib.sha256(json.dumps([qi, q["attr"], q["to"], scen]).encode()).hexdigest()[:16]
                    row["scenarios_sha"] = sha
                    if sha not in truth_cache:
                        t1 = time.time()
                        true_w = []
                        for s in scen:
                            prods = json.loads(s)
                            if not prods:
                                true_w.append(np.nan)
                                continue
                            cat = np.array([[int(str(p[f"A{a + 1}"])[1:]) - 1 for a in range(len(levels))] for p in prods])
                            prices = np.array([float(p[wtplib.PRICE]) for p in prods])
                            w, _, _ = wtplib.true_market_wtp(people[0], people[1], pop, cat, prices, 0, q["attr"], q["to"])
                            true_w.append(np.nan if w is None else w)
                        truth_cache[sha] = (np.array(true_w, float), round(time.time() - t1, 1))
                    true_w, row["truth_seconds"] = truth_cache[sha]
                    ok_app, ok_true = ~np.isnan(app_w), ~np.isnan(true_w)
                    both = ok_app & ok_true
                    row.update(
                        n_scenarios=len(details), true_found=int(ok_true.sum()),
                        app_median=float(np.median(app_w[ok_app])) if ok_app.any() else None,
                        true_median=float(np.median(true_w[ok_true])) if ok_true.any() else None,
                        true_mean=float(np.mean(true_w[ok_true])) if ok_true.any() else None,
                        true_median_matched=float(np.median(true_w[both])) if both.any() else None,
                        app_median_matched=float(np.median(app_w[both])) if both.any() else None,
                        true_p10=float(np.quantile(true_w[ok_true], 0.1)) if ok_true.any() else None,
                        true_p90=float(np.quantile(true_w[ok_true], 0.9)) if ok_true.any() else None,
                        scenario_abs_error_median=float(np.median(np.abs(app_w[both] - true_w[both]))) if both.any() else None,
                        scenario_abs_error_p90=float(np.quantile(np.abs(app_w[both] - true_w[both]), 0.9)) if both.any() else None,
                        scenario_error_median=float(np.median(app_w[both] - true_w[both])) if both.any() else None)
                    rows.append(row)
                out = {k: rec.get(k) for k in ("run_id", "arm", "design", "levels", "tasks", "options", "n", "rep", "pop", "pop_seed",
                                               "price", "addon", "hb_prior_variance", "price_constraint", "stage")}
                out.update(family=fam, key=task["key"], rows=rows, seconds=round(time.time() - t0, 1), m_truth=M_TRUTH, sos=SOS,
                           status="ok")
                target.write_text(json.dumps(out, indent=1), encoding="utf-8")
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((fam, run_id, f"{type(exc).__name__}: {exc}", traceback.format_exc()[-1500:]))
    except Exception as exc:  # noqa: BLE001
        failed.append(("task", task["key"], f"{type(exc).__name__}: {exc}", traceback.format_exc()[-1500:]))
    return {"key": task["key"], "done": done, "failed": failed, "seconds": round(time.time() - t_task, 1)}
