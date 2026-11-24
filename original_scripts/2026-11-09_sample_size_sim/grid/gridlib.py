"""Full sample-size grid on top of the pilot library (simlib).

Adds: inline design specs, a 'realistic' heterogeneity level calibrated on real HB
utilities (see calibration.json), a per-run HB seed for the seed test, and grid-local
result/work folders. Estimation and simulation stay the in-app code paths of simlib.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

GRID = Path(__file__).resolve().parent
sys.path.insert(0, str(GRID.parent))
import simlib  # noqa: E402

simlib.RESULTS = GRID / "results"
simlib.WORK = GRID / "work"

# Realistic level (calibration.json): real HB point estimates have |mean| median 0.45 to 0.84,
# p90 1.6 to 2.8, and point-estimate SD median about 0.8 (shrunk). With the shrinkage seen in
# the pilot (point SD about 0.5 of true SD) the population SD is about 1.4 (variance about 2);
# one Sawtooth population covariance export gave variances from 0.7 to 25 (median 4.8).
REAL_MEAN_SD = 1.0          # means ~ N(0, 1): |mean| median 0.67, p90 1.65
REAL_SD_MEDIAN = 1.4        # population SD per parameter, lognormal
REAL_SD_LOGSPREAD = 0.5


def _draw_real_sd(rng) -> float:
    return float(REAL_SD_MEDIAN * math.exp(rng.standard_normal() * REAL_SD_LOGSPREAD))


def make_population(levels, hetero, seed: int) -> dict:
    if hetero != "realistic":
        return _orig_make_population(levels, hetero != "homogeneous", seed)
    rng = np.random.default_rng(seed)
    k = sum(l - 1 for l in levels)
    means = rng.standard_normal(k) * REAL_MEAN_SD
    var = np.array([_draw_real_sd(rng) ** 2 for _ in range(k)])
    return {"levels": list(levels), "means": means, "var": var, "asc_mean": None,
            "asc_var": _draw_real_sd(rng) ** 2}


_orig_make_population = simlib.make_population
simlib.make_population = make_population


def run(spec: dict) -> dict:
    """Register the inline design, set the HB seed, run through simlib.run_one, drop work files."""
    simlib.RESULTS = GRID / "results"
    simlib.WORK = GRID / "work"
    simlib.make_population = make_population
    simlib.DESIGNS[spec["design"]] = dict(levels=list(spec["levels"]), tasks=int(spec["tasks"]),
                                          options=int(spec["options"]))
    simlib.APP_HB["seed"] = int(spec.get("hb_seed", 42))
    rec = simlib.run_one(spec)
    if rec.get("status") == "ok":
        shutil.rmtree(simlib.WORK / spec["run_id"], ignore_errors=True)
    return rec
