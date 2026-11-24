"""Sample-size simulation with populations taken from real studies (or from a re-tuned generator).

Everything except the population is the stored pipeline of 2026-11-09_sample_size_sim (simlib.run_one):
Choicery DesignGenerator, in-app Julia HB with app defaults, in-app MarketSimulator, same markets rule.

spec["hetero"]:
  "empirical"  population = normal with the mean vector and the FULL covariance matrix that the in-app HB
               estimated for the real study spec["study"] (fits/<study>.json); None constant dropped.
               Optional spec["sd_scale"] / spec["mean_scale"] rescale spread and means.
  "tuned"      independent normal per utility like the 'realistic' generator, with the numbers in spec["gen"]
               (mean_sd, sd_median, sd_logspread), optional equal correlation inside attributes spec["gen"]["rho"].
  anything else -> the stored generators (gridlib).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / "_scratch" / "2026-11-09_sample_size_sim"
for p in (str(SIM), str(SIM / "grid")):
    if p not in sys.path:
        sys.path.insert(0, p)
FITS = HERE / "fits"
CURRENT = {}


def pin_to_efficiency_cores():
    try:
        import psutil

        if psutil.cpu_count() == 32:
            psutil.Process().cpu_affinity(list(range(16, 32)))
    except Exception:  # noqa: BLE001
        pass


def empirical_population(levels, spec) -> dict:
    fit = json.loads((FITS / f"{spec['study']}.json").read_text(encoding="utf-8"))
    assert list(fit["levels"]) == list(levels), (fit["levels"], levels)
    k = sum(l - 1 for l in levels)
    means = np.asarray(fit["alpha"], float)[:k] * float(spec.get("mean_scale", 1.0))
    cov = np.asarray(fit["sigma"], float)[:k, :k] * float(spec.get("sd_scale", 1.0)) ** 2
    cov = (cov + cov.T) / 2
    return {"levels": list(levels), "means": means, "cov": cov, "var": np.diag(cov).copy(), "asc_mean": None,
            "asc_var": 1.0}


def tuned_population(levels, spec, seed: int) -> dict:
    g = spec["gen"]
    rng = np.random.default_rng(seed)
    k = sum(l - 1 for l in levels)
    means = rng.standard_normal(k) * g["mean_sd"]
    sd = np.array([g["sd_median"] * math.exp(rng.standard_normal() * g["sd_logspread"]) for _ in range(k)])
    pop = {"levels": list(levels), "means": means, "var": sd ** 2, "asc_mean": None, "asc_var": float(sd.mean() ** 2)}
    rho = float(g.get("rho", 0.0))
    if rho:
        corr, pos = np.eye(k), 0
        for l in levels:
            corr[pos:pos + l - 1, pos:pos + l - 1] = rho
            pos += l - 1
        np.fill_diagonal(corr, 1.0)
        pop["cov"] = corr * np.outer(sd, sd)
    return pop


def run(spec: dict) -> dict:
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import gridlib
    import simlib

    pin_to_efficiency_cores()
    CURRENT["spec"] = spec

    def make_population(levels, hetero, seed):
        s = CURRENT["spec"]
        if hetero == "empirical":
            return empirical_population(levels, s)
        if hetero == "tuned":
            return tuned_population(levels, s, seed)
        return gridlib.make_population(levels, hetero, seed)

    def sample_params(pop, n, rng):
        if "cov" in pop:
            return rng.multivariate_normal(pop["means"], pop["cov"], size=n, method="eigh")
        z = rng.standard_normal((n, len(pop["means"])))
        return pop["means"] + z * np.sqrt(pop["var"])

    simlib.make_population = make_population
    simlib.sample_params = sample_params
    simlib.RESULTS = HERE / spec["_results"]
    simlib.WORK = HERE / ("work_" + spec["_results"])
    simlib.DESIGNS[spec["design"]] = dict(levels=list(spec["levels"]), tasks=int(spec["tasks"]),
                                          options=int(spec["options"]))
    simlib.APP_HB["seed"] = int(spec.get("hb_seed", 42))
    simlib.APP_HB["n_burn"] = int(spec.get("n_burn", 10000))
    simlib.APP_HB["n_draws"] = int(spec.get("n_draws", 10000))
    simlib.APP_HB["thinning"] = int(spec.get("thinning", 5))
    rec = simlib.run_one(spec)
    if rec.get("status") == "ok":
        shutil.rmtree(simlib.WORK / spec["run_id"], ignore_errors=True)
    return rec


def cost(s):
    return s["n"] * s["tasks"] * sum(l - 1 for l in s["levels"]) * s.get("n_burn", 10000)
