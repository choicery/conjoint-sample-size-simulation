"""Pure sampling error of a choice share: true share of the drawn respondents minus true share of the target group.

Why (23.11.2026): the user decided that the calculator shows only the part of the uncertainty the sample can settle.
This part does not involve the estimation at all (no prior variance, no design of the choice tasks): both shares are
computed from the TRUE utilities, once for the N simulated respondents and once for the whole simulated population.

Data: every stored run of the share blocks (results_homogen, _spreadn, _designpath, _chains, _floor of the night-run
folder) carries 'insample' and 'pop' for every product of six markets (3 or 4 products) and of 36 further markets.
Runs on the same drawn people (other answers, other estimation settings, repeats, other result folder) are counted
once: a sample is a design, a population, N, a spread and the seed of the draw; every share comparison counts once.

Theory: a share is the mean of the personal choice probabilities p_i. Its sampling variance is Var(p_i) / N, and
Var(p_i) = h x P x (1 - P) with P the share and h between 0 (everybody has the same probability, 'clones') and 1
(everybody is fully decided; the classic formula of a poll). So: margin = k x sqrt(h x P x (1 - P) / N).

Usage: python stichprobenfehler.py > ergebnisse_stichprobe.txt
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
NIGHT = Path(__file__).resolve().parents[1] / "2026-11-21_nachtlauf"
FOLDERS = ("results_homogen", "results_spreadn", "results_designpath", "results_chains", "results_floor")
SOP = "share_of_preference"
NAME = {0.55: "homogeneous (0.55)", 1.0: "typical (1.0)", 1.7: "heterogeneous (1.7)"}
rows, seen, files = [], set(), 0
for folder in FOLDERS:
    for path in sorted((NIGHT / folder).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("status") != "ok" or not r.get("shares"):
            continue
        files += 1
        spread = float(r.get("spread") or 1.0)
        data = f"{r['design']}|{r['pop']}|{r['n']}|{spread}|{r.get('data_seed')}"
        for src, extra in ((r["shares"], 0), (r.get("shares_extra") or [], 1)):
            for s in src:
                if s.get("rule", SOP) != SOP or s.get("is_none") or s.get("insample") is None or s.get("pop") is None:
                    continue
                rows.append(dict(data=data, block=folder[8:], design=r["design"], pop=r["pop"], n=r["n"], spread=spread, extra=extra,
                                 scenario=s["scenario"], alt=s.get("alt"), products=s.get("n_products"), P=s["pop"], e=100 * (s["insample"] - s["pop"])))
d = pd.DataFrame(rows).drop_duplicates(["data", "extra", "scenario", "alt"]).reset_index(drop=True)
d["var"] = d["e"] ** 2 * d["n"]                               # in points^2 x respondents
d["unit"] = 1e4 * d["P"] * (1 - d["P"])                       # classic poll variance x N, points^2
print(f"{files} stored runs, {d['data'].nunique()} different samples, {len(d)} different shares ({int((d['extra'] == 0).sum())} of the six main markets)")
print("products per market:", dict(d.drop_duplicates(["data", "extra", "scenario"])["products"].value_counts().sort_index()))
SPREADS, NS = sorted(d["spread"].unique()), sorted(d["n"].unique())
rm = lambda x: math.sqrt(np.mean(x ** 2)) if len(x) else float("nan")
main = d[d["extra"] == 0]

print("\n== 1 RMSE of the sampling error of a share (points), six main markets | in brackets RMSE x sqrt(N) | data sets")
for s in SPREADS:
    x = main[main["spread"] == s]
    print(f"   {NAME[s]:20s} " + " | ".join(f"N {n}: {rm(x[x['n'] == n]['e']):4.2f} ({rm(x[x['n'] == n]['e']) * n ** .5:4.1f}) {x[x['n'] == n]['data'].nunique():3d}" for n in NS if (x["n"] == n).any()))
print("   all markets (main and further):")
for s in SPREADS:
    x = d[d["spread"] == s]
    print(f"   {NAME[s]:20s} " + " | ".join(f"N {n}: {rm(x[x['n'] == n]['e']):4.2f} ({rm(x[x['n'] == n]['e']) * n ** .5:4.1f})" for n in NS if (x["n"] == n).any()))

print("\n== 2 Does the sampling error follow 1 / sqrt(N)? Slope of log RMSE on log N (theory: -0.50), all markets")
for s in SPREADS:
    x = d[d["spread"] == s]
    pts = [(math.log(n), math.log(rm(x[x["n"] == n]["e"]))) for n in NS if (x["n"] == n).any()]
    slope = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]
    print(f"   {NAME[s]:20s} slope {slope:+.3f} over N {min(n for n in NS if (x['n'] == n).any())} to {max(n for n in NS if (x['n'] == n).any())}")

print("\n== 3 h = sampling variance as a share of the classic poll variance P(1-P)/N, by size of the true share (all markets)")
bins = [0, .1, .2, .3, .4, .5, .7, 1.0]
d["bin"] = pd.cut(d["P"], bins)
print("   share of the product   " + " | ".join(f"{NAME[s]:>20s}" for s in SPREADS) + " |  shares")
for b, x in d.groupby("bin", observed=True):
    print(f"   {str(b):22s} " + " | ".join(f"h {x[x['spread'] == s]['var'].sum() / x[x['spread'] == s]['unit'].sum():5.3f} (RMSExsqrtN {math.sqrt(x[x['spread'] == s]['var'].mean()):4.1f})" for s in SPREADS) + f" | {len(x):6d}")
H = {s: float(d[d["spread"] == s]["var"].sum() / d[d["spread"] == s]["unit"].sum()) for s in SPREADS}
C = {s: float(math.sqrt(d[d["spread"] == s]["var"].mean())) for s in SPREADS}
print("   pooled h:            " + " | ".join(f"{NAME[s]}: {H[s]:.3f}" for s in SPREADS))
print("   pooled RMSE x sqrt(N): " + " | ".join(f"{NAME[s]}: {C[s]:.1f}" for s in SPREADS) + "   (a share of 50% in a poll: 50.0)")

print("\n== 4 h by design and by number of products (does the market matter?), all markets")
for s in SPREADS:
    x = d[d["spread"] == s]
    by_design = x.groupby("design").apply(lambda y: y["var"].sum() / y["unit"].sum(), include_groups=False)
    by_products = x.groupby("products").apply(lambda y: y["var"].sum() / y["unit"].sum(), include_groups=False)
    by_pop = x.groupby(["design", "pop"]).apply(lambda y: y["var"].sum() / y["unit"].sum(), include_groups=False)
    print(f"   {NAME[s]:20s} designs: {by_design.min():.3f} to {by_design.max():.3f} ({len(by_design)}) | populations x designs: 10% {by_pop.quantile(.1):.3f}, median {by_pop.median():.3f}, 90% {by_pop.quantile(.9):.3f} ({len(by_pop)})"
          f" | products: " + ", ".join(f"{int(k)}: {v:.3f}" for k, v in by_products.items()))

print("\n== 5 Coverage: share of the sampling errors inside the margin (target 90% with k = 1.65, 95% with k = 1.96)")
d["m_h"] = np.sqrt(d["spread"].map(H) * d["unit"] / d["n"])                 # with the size of the share
d["m_c"] = d["spread"].map(C) / np.sqrt(d["n"])                             # one number per group
d["m_max"] = np.sqrt(d["spread"].map(H) * 2500 / d["n"])                    # for a share of 50% (the largest error)
for label, col in (("formula with the share  k x sqrt(h P(1-P)/N)", "m_h"), ("one number per group   k x c / sqrt(N)", "m_c"), ("for a share of 50%     k x sqrt(h x 0.25/N)", "m_max")):
    print(f"   {label}")
    for s in SPREADS:
        x = d[d["spread"] == s]
        print(f"      {NAME[s]:20s} 90%: {100 * (x['e'].abs() <= 1.65 * x[col]).mean():4.1f}%  95%: {100 * (x['e'].abs() <= 1.96 * x[col]).mean():4.1f}%   by N (90%): "
              + " ".join(f"{n}: {100 * (x[x['n'] == n]['e'].abs() <= 1.65 * x[x['n'] == n][col]).mean():3.0f}" for n in NS if (x["n"] == n).any()))
x = d[d["P"] >= .3]
print("   shares of 30% or more, one number per group: " + " | ".join(f"{NAME[s]}: {100 * (x[x['spread'] == s]['e'].abs() <= 1.65 * x[x['spread'] == s]['m_c']).mean():4.1f}%" for s in SPREADS))
print("   shares of 30% or more, formula with the share: " + " | ".join(f"{NAME[s]}: {100 * (x[x['spread'] == s]['e'].abs() <= 1.65 * x[x['spread'] == s]['m_h']).mean():4.1f}%" for s in SPREADS))

CALC = {0.55: 0.351, 1.0: 0.563, 1.7: 0.719}      # the calculator's constants: h computed from the populations (h_kurve.py)
d["m_calc"] = np.sqrt(d["spread"].map(CALC) * d["unit"] / d["n"])
print("   the calculator's own constants (h = 0.351 / 0.563 / 0.719), formula with the share")
for s_ in SPREADS:
    x = d[d["spread"] == s_]
    big = x[x["P"] >= .3]
    print(f"      {NAME[s_]:20s} 90%: {100 * (x['e'].abs() <= 1.65 * x['m_calc']).mean():4.1f}%  95%: {100 * (x['e'].abs() <= 1.96 * x['m_calc']).mean():4.1f}%"
          f"   shares of 30% or more: {100 * (big['e'].abs() <= 1.65 * big['m_calc']).mean():4.1f}%   by N (90%): "
          + " ".join(f"{n}: {100 * (x[x['n'] == n]['e'].abs() <= 1.65 * x[x['n'] == n]['m_calc']).mean():3.0f}" for n in NS if (x["n"] == n).any()))

print("\n== 6 Respondents for a margin (90%), from the formula; and the margin at fixed sample sizes")
print("   a) one number per group (average over the shares of a market)")
for s in SPREADS:
    print(f"      {NAME[s]:20s} " + " | ".join(f"±{t}: {math.ceil((1.65 * C[s] / t) ** 2):5d}" for t in (5, 4, 3, 2, 1.5, 1))
          + "   ||   " + " | ".join(f"N {n}: ±{1.65 * C[s] / n ** .5:3.1f}" for n in (50, 100, 200, 400, 1000)))
print("   b) for a share of 50% (largest sampling error)")
for s in SPREADS:
    c50 = math.sqrt(H[s] * 2500)
    print(f"      {NAME[s]:20s} " + " | ".join(f"±{t}: {math.ceil((1.65 * c50 / t) ** 2):5d}" for t in (5, 4, 3, 2, 1.5, 1))
          + "   ||   " + " | ".join(f"N {n}: ±{1.65 * c50 / n ** .5:3.1f}" for n in (50, 100, 200, 400, 1000)))
print(f"      classic poll (h = 1)  " + " | ".join(f"±{t}: {math.ceil((1.65 * 50 / t) ** 2):5d}" for t in (5, 4, 3, 2, 1.5, 1))
      + "   ||   " + " | ".join(f"N {n}: ±{1.65 * 50 / n ** .5:3.1f}" for n in (50, 100, 200, 400, 1000)))

print("\n== 7 How firm are h and c? Whole populations x designs drawn again with replacement (2,000 times), 5% to 95%")
rng = np.random.default_rng(7)
for s in SPREADS:
    x = d[d["spread"] == s]
    g = x.groupby(["design", "pop"]).agg(v=("var", "sum"), u=("unit", "sum"), k=("var", "size"))
    v, u, k = g["v"].to_numpy(), g["u"].to_numpy(), g["k"].to_numpy()
    hs, cs = [], []
    for _ in range(2000):
        i = rng.integers(0, len(g), len(g))
        hs.append(v[i].sum() / u[i].sum())
        cs.append(math.sqrt(v[i].sum() / k[i].sum()))
    print(f"   {NAME[s]:20s} h {H[s]:.3f} ({np.quantile(hs, .05):.3f} to {np.quantile(hs, .95):.3f})   c {C[s]:.1f} ({np.quantile(cs, .05):.1f} to {np.quantile(cs, .95):.1f})   {len(g)} populations x designs")
