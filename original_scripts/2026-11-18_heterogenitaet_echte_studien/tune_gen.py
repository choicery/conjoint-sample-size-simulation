"""Choose the numbers of the re-tuned generator so that it matches the real market studies (medians over families)."""
import json
import sys

import numpy as np
import pandas as pd

import summarize_real as SR
from fitlib import transform_full

d = pd.read_csv("tabellen/echte_studien.csv")
d["group"] = d["study"].map(SR.group_of)
mk = d[d.group == "market"]
fam = mk.groupby("family").agg(levels=("levels", "first"), m=("m", "median"), h=("h", "median"), sd_median=("sd_median", "median"),
                               M=("M", "median"), H=("H", "median"), abs_mean_median=("abs_mean_median", "median"))
rng = np.random.default_rng(5)


def ref(levels, g, pops=150):
    k = sum(l - 1 for l in levels)
    t = transform_full(levels)
    rows = []
    for _ in range(pops):
        means = rng.standard_normal(k) * g["mean_sd"]
        var = (g["sd_median"] * np.exp(rng.standard_normal(k) * g["sd_logspread"])) ** 2
        mf, cf = t @ means, t @ np.diag(var) @ t.T
        sd = np.sqrt(np.diag(cf))
        M, H = SR.pair_stats(levels, mf, cf, rng, pairs=400)
        rows.append({"m": np.sqrt(np.mean(mf ** 2)), "h": np.sqrt(np.mean(sd ** 2)), "sd_median": np.median(sd), "M": M, "H": H,
                     "abs_mean_median": np.median(np.abs(mf))})
    return pd.DataFrame(rows).median()


cands = {"realistic (stored runs)": dict(mean_sd=1.0, sd_median=1.4, sd_logspread=0.5)}
for ms in (0.85, 0.9, 1.0):
    for sdm in (0.85, 0.9, 0.95, 1.0):
        cands[f"mean {ms} sd {sdm}"] = dict(mean_sd=ms, sd_median=sdm, sd_logspread=0.4)
for name, g in cands.items():
    r = pd.DataFrame({f: ref([int(x) for x in row["levels"].split()], g) for f, row in fam.iterrows()}).T
    ratio = fam[["m", "h", "sd_median", "M", "H", "abs_mean_median"]] / r[["m", "h", "sd_median", "M", "H", "abs_mean_median"]]
    print(f"{name:26s} echt/Generator (Median über {len(fam)} Familien): " + "  ".join(f"{c} {ratio[c].median():.2f}" for c in ratio.columns))
