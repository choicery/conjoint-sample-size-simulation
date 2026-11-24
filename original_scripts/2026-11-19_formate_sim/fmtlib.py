"""One run of the question-format simulation (19.11.2026): same pipeline as emplib.run of 18.11., but through simlib_fmt
(adds BW-CBC with None, product-only markets for the None formats, extra markets for every format).

Population: the tuned generator of 18.11. (tuned_gen.json), identical to the plain CBC runs in results_tuned/.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HET = HERE.parent / "2026-11-18_heterogenitaet_echte_studien"
for p in (str(HERE), str(HET)):
    if p not in sys.path:
        sys.path.insert(0, p)
CORES = list(range(26, 32))   # six efficiency cores
CURRENT = {}


def pin():
    try:
        import psutil

        if psutil.cpu_count() == 32:
            psutil.Process().cpu_affinity(CORES)
    except Exception:  # noqa: BLE001
        pass


def run(spec: dict) -> dict:
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    pin()
    import emplib          # only for tuned_population (and the path to the stored simulation)
    import simlib_fmt as simlib

    CURRENT["spec"] = spec

    def make_population(levels, hetero, seed):
        assert hetero == "tuned", hetero
        return emplib.tuned_population(levels, CURRENT["spec"], seed)

    def sample_params(pop, n, rng):
        z = rng.standard_normal((n, len(pop["means"])))
        return pop["means"] + z * np.sqrt(pop["var"])

    simlib.make_population = make_population
    simlib.sample_params = sample_params
    simlib.RESULTS = HERE / spec["_results"]
    simlib.WORK = HERE / ("work_" + spec["_results"])
    simlib.DESIGNS[spec["design"]] = dict(levels=list(spec["levels"]), tasks=int(spec["tasks"]), options=int(spec["options"]))
    simlib.APP_HB["seed"] = int(spec.get("hb_seed", 42))
    rec = simlib.run_one(spec)
    if rec.get("status") == "ok":
        shutil.rmtree(simlib.WORK / spec["run_id"], ignore_errors=True)
    return rec


def cost(s):
    weight = {"drcbc": 1.35, "bwcbc": 1.45, "bwcbc_none": 1.6}.get(s["method"], 1.0)
    return weight * s["n"] * s["tasks"] * sum(l - 1 for l in s["levels"])
