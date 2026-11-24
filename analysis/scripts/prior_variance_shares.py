# Packaged copy of the script auswertung_chains.py as it ran for the article. Changed for this archive: it reads the files
# through _data.py (folder or ZIP file in ../../data).
"""Evaluation of the block 'chains' (see nightlib.py): prior, start value and length of the chains on identical answers.

Error of a share = app value - true value in the population, percentage points, share of preference, the six markets
of the formula runs. 'estimation' = app value - true value of the same respondents.

Usage: python auswertung_chains.py [results folder]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _data import studies

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parent

rows, bad = [], []
for r in studies("prior_variance_runs"):
    if r.get("status") != "ok" or "mean_slope" not in r:
        bad.append((r.get("run_id"), r.get("status"), str(r.get("error"))[:120], r.get("extra_error")))
        continue
    sh = [s for s in r["shares"] if s["rule"] == "share_of_preference"]
    ex = sh + [s for s in r.get("shares_extra") or [] if s["rule"] == "share_of_preference"]
    d, g = r["diagnostics"], r.get("diagnostics_groups") or {}
    rows.append(dict(
        cell=r["cell"], variant=r["variant"], design=r["design"], n=r["n"], spread=r["spread"], pop=r["pop"], sha=r.get("frame_sha"),
        mse=np.mean([(100 * (s["est"] - s["pop"])) ** 2 for s in sh]), mse_est=np.mean([(100 * (s["est"] - s["insample"])) ** 2 for s in sh]),
        mse_samp=np.mean([(100 * (s["insample"] - s["pop"])) ** 2 for s in sh]),
        mse_all=np.mean([(100 * (s["est"] - s["pop"])) ** 2 for s in ex]), bias_big=np.mean([100 * (s["est"] - s["pop"]) for s in sh if s["pop"] > 0.5] or [np.nan]),
        slope=r["mean_slope"], sd_ratio=r["sd_ratio"], rhat_alpha=d.get("rhat_alpha_max"), rhat_sigma=d.get("rhat_sigma_max"),
        rhat_diag=(g.get("sigma_diag") or {}).get("rhat_max"), ess=d.get("ess_min"), rlh=d.get("rlh_mean"), minutes=r["hb_seconds"] / 60))
d = pd.DataFrame(rows)
print(f"prior_variance_runs: {len(d)} runs evaluated, {len(bad)} not ok")
for b in bad:
    print("   not ok:", b)
if d.empty:
    sys.exit()
mixed = d.groupby("cell")["sha"].nunique()
print("cells:", d["cell"].nunique(), "| cells whose variants did NOT get identical answers:", int((mixed > 1).sum()))

ORDER = ["d4", "d1", "d025", "d4lo", "d025hi", "m4", "m1", "x4", "x1"]
LABEL = {"d4": "prior 4.0, default length (app default)", "d1": "prior 1.0, default length", "d025": "prior 0.25, default length",
         "d4lo": "prior 4.0, start as with 0.25", "d025hi": "prior 0.25, start as with 4.0", "m4": "prior 4.0, 80,000 + 20,000",
         "m1": "prior 1.0, 80,000 + 20,000", "x4": "prior 4.0, 400,000 + 100,000", "x1": "prior 1.0, 400,000 + 100,000"}


def table(sel: pd.DataFrame, title: str, variants=None, complete=True) -> None:
    variants = [v for v in (variants or ORDER) if v in set(sel["variant"])]
    if complete:   # only cells in which every variant of the table is finished
        have = sel[sel["variant"].isin(variants)].groupby("cell")["variant"].nunique()
        sel = sel[sel["cell"].isin(have[have == len(variants)].index)]
    print(f"\n{title}   [{sel['cell'].nunique()} cells]")
    if sel.empty:
        return
    print(f"   {'':40s} | share error RMSE | estimation part | means stretched | SD est/true | R-hat Sigma med | runs > 1.10 | ESS min | RLH  | minutes")
    for v in variants:
        x = sel[sel["variant"] == v]
        print(f"   {LABEL[v]:40s} | {math.sqrt(x['mse'].mean()):8.2f}         | {math.sqrt(x['mse_est'].mean()):8.2f}        | {x['slope'].mean():8.2f}        "
              f"| {x['sd_ratio'].mean():6.2f}      | {x['rhat_sigma'].median():8.2f}        | {100 * (x['rhat_sigma'] > 1.10).mean():5.0f}%      "
              f"| {x['ess'].median():5.0f}   | {x['rlh'].mean():.2f} | {x['minutes'].median():5.1f}")


def paired(sel: pd.DataFrame, a: str, b: str) -> str:
    p = sel[sel["variant"].isin([a, b])].pivot_table(index="cell", columns="variant", values="mse")
    p = p.dropna()
    if p.empty or a not in p or b not in p:
        return f"{a} vs {b}: no complete cells"
    ra, rb = np.sqrt(p[a]), np.sqrt(p[b])
    return (f"{a} vs {b}: RMSE {math.sqrt(p[a].mean()):.2f} vs {math.sqrt(p[b].mean()):.2f} | {a} smaller in {int((ra < rb).sum())} of {len(p)} cells"
            f" | mean difference per cell {float((ra - rb).mean()):+.2f} (SE {float((ra - rb).std(ddof=1) / math.sqrt(len(p))):.2f})")


main = d[(d["n"] == 1000) & d["design"].isin(["med", "a5l4t12o3", "a4l4t8o2", "a5l5t8o2"])]
print("\n" + "=" * 110 + "\n1 PRIOR, START VALUE AND LENGTH, 1,000 respondents, four designs, by spread of the people")
for s in sorted(main["spread"].unique()):
    table(main[main["spread"] == s], f"spread {s}", ORDER[:7])
table(main, "all three spreads together", ORDER[:7])

print("\n" + "=" * 110 + "\n2 PRIOR OR START VALUE? (default length; RMSE of a share)")
for s in sorted(main["spread"].unique()):
    x = main[main["spread"] == s]
    have = x[x["variant"].isin(["d4", "d4lo", "d025hi", "d025"])].groupby("cell")["variant"].nunique()
    x = x[x["cell"].isin(have[have == 4].index)]
    if x.empty:
        continue
    r = {v: math.sqrt(x[x["variant"] == v]["mse"].mean()) for v in ("d4", "d4lo", "d025hi", "d025")}
    sl = {v: x[x["variant"] == v]["slope"].mean() for v in ("d4", "d4lo", "d025hi", "d025")}
    print(f"   spread {s} [{x['cell'].nunique()} cells]            start as with 4.0 | start as with 0.25")
    print(f"      prior 4.0    RMSE {r['d4']:5.2f} | {r['d4lo']:5.2f}      means stretched {sl['d4']:.2f} | {sl['d4lo']:.2f}")
    print(f"      prior 0.25   RMSE {r['d025hi']:5.2f} | {r['d025']:5.2f}      means stretched {sl['d025hi']:.2f} | {sl['d025']:.2f}")

print("\n" + "=" * 110 + "\n3 PAIRED COMPARISONS (same answers), 1,000 respondents, four designs")
for s in list(sorted(main["spread"].unique())) + [None]:
    x = main if s is None else main[main["spread"] == s]
    print(f"   spread {s if s is not None else 'all'}:")
    for a, b in (("m4", "d4"), ("m1", "d1"), ("d1", "d4"), ("m1", "m4"), ("d025", "d4"), ("d4lo", "d4"), ("d025hi", "d025")):
        print("      " + paired(x, a, b))

print("\n" + "=" * 110 + "\n4 BY DESIGN, 1,000 respondents (all spreads; RMSE of a share / R-hat Sigma median)")
for des in ["a5l5t8o2", "a4l4t8o2", "med", "a5l4t12o3", "a4l3t16o4"]:
    x = d[(d["n"] == 1000) & (d["design"] == des)]
    if x.empty:
        continue
    cells = x.groupby("cell")["variant"].nunique()
    line = " | ".join(f"{v} {math.sqrt(x[x['variant'] == v]['mse'].mean()):.2f} / {x[x['variant'] == v]['rhat_sigma'].median():.2f}"
                      for v in ORDER if v in set(x["variant"]))
    print(f"   {des:10s} [{len(cells)} cells]  {line}")

print("\n" + "=" * 110 + "\n5 MORE RESPONDENTS: does the error keep falling? (spread 1.0; designs med and a5l5t8o2)")
byn = d[(d["spread"] == 1.0) & d["design"].isin(["med", "a5l5t8o2"])]
for des in ("med", "a5l5t8o2", None):
    x = byn if des is None else byn[byn["design"] == des]
    print(f"   {des or 'both designs'}: RMSE of a share (estimation part) [cells]")
    for v in ORDER[:7]:
        cells = []
        for n in (400, 1000, 2500):
            y = x[(x["variant"] == v) & (x["n"] == n)]
            cells.append(f"N {n}: {math.sqrt(y['mse'].mean()):5.2f} ({math.sqrt(y['mse_est'].mean()):5.2f}) [{len(y)}]" if len(y) else f"N {n}:   -")
        print(f"      {LABEL[v]:40s} " + " | ".join(cells))

print("\n" + "=" * 110 + "\n6 REFERENCE: very long chains (per cell: RMSE of a share / means stretched / SD est/true / R-hat Sigma)")
ref_cells = sorted(set(d[d["variant"].isin(["x4", "x1"])]["cell"]))
for c in ref_cells:
    x = d[d["cell"] == c].set_index("variant")
    print(f"   {c}")
    for v in ("d4", "m4", "x4", "d1", "m1", "x1", "d025"):
        if v in x.index:
            y = x.loc[v]
            print(f"      {LABEL[v]:40s} {math.sqrt(y['mse']):5.2f} / {y['slope']:.2f} / {y['sd_ratio']:.2f} / {y['rhat_sigma']:.2f}   ({y['minutes']:.0f} min)")

print("\n" + "=" * 110 + "\n7 CONTROL: design with chains that are fine at the default (a4l3t16o4)")
table(d[d["design"] == "a4l3t16o4"], "a4l3t16o4, 1,000 respondents", ORDER[:7])
print("\n" + "=" * 110 + "\n8 SIMILAR OR DIFFERENT PEOPLE (1,000 respondents, four designs; same people scaled, same tasks, same seeds)")
print("   RMSE of a share: total | sample part (true value of the respondents - population) | estimation part; cells complete in all three spreads")
key = main.assign(base=main["design"] + "_p" + main["pop"].astype(str))
for v in ORDER[:7]:
    x = key[key["variant"] == v]
    have = x.groupby("base")["spread"].nunique()
    x = x[x["base"].isin(have[have == 3].index)]
    if x.empty:
        continue
    parts = []
    for s in sorted(x["spread"].unique()):
        y = x[x["spread"] == s]
        parts.append(f"spread {s}: {math.sqrt(y['mse'].mean()):5.2f} | {math.sqrt(y['mse_samp'].mean()):4.2f} | {math.sqrt(y['mse_est'].mean()):5.2f}")
    pv = x.pivot_table(index="base", columns="spread", values="mse").pipe(np.sqrt)
    lo, hi = pv[pv.columns[0]], pv[pv.columns[-1]]
    print(f"   {LABEL[v]:40s} " + "  ||  ".join(parts) + f"  ||  similar people worse than different people in {int((lo > hi).sum())} of {len(pv)} cells"
          f" (mean difference {float((lo - hi).mean()):+.2f}, SE {float((lo - hi).std(ddof=1) / math.sqrt(len(pv))):.2f})")
print("   by design, app default (prior 4.0, default length): total RMSE at each spread [cells]")
for des in ["a5l5t8o2", "a4l4t8o2", "med", "a5l4t12o3"]:
    x = key[(key["variant"] == "d4") & (key["design"] == des)]
    print(f"      {des:10s} " + " | ".join(f"spread {s}: {math.sqrt(x[x['spread'] == s]['mse'].mean()) if (x['spread'] == s).any() else float('nan'):5.2f} [{int((x['spread'] == s).sum())}]"
                                       for s in sorted(key["spread"].unique())))
print("\nminutes per run (estimation):", d.groupby("variant")["minutes"].median().round(1).to_dict())
