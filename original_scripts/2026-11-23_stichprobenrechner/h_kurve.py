"""h for any spread between people, computed directly from the simulated populations (no estimation, no answers).

h = Var(personal choice probability) / (P x (1 - P)), summed over the products of many random markets.
With h the sampling margin of a share is  k x sqrt(h x P x (1 - P) / N)  (see stichprobenfehler.py, where h was
measured from drawn samples at the spreads 0.55, 1.0 and 1.7). Here: the same populations (six designs x four
populations of block 'homogen'), 20,000 people each, 240 random markets with 2 to 5 products.

Usage: python h_kurve.py > ergebnisse_h_kurve.txt
"""
import json
import math
import os
import sys
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):      # one thread: other runs use the processors
    os.environ[_k] = "1"

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
NIGHT = ROOT / "2026-11-21_nachtlauf"
for p in (ROOT / "2026-11-09_sample_size_sim", ROOT / "2026-11-18_heterogenitaet_echte_studien"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(NIGHT))
import emplib  # noqa: E402
import nightlib  # noqa: E402
import simlib  # noqa: E402

nightlib.pin()      # the user's processors stay free

specs = json.loads((NIGHT / "homogen.json").read_text(encoding="utf-8"))
base = {}
for s in specs:
    if abs(s["spread"] - 1.0) < 1e-9:
        base.setdefault((s["design"], s["pop"]), s)
print(f"{len(base)} populations x designs:", sorted({k[0] for k in base}))
SPREADS = (0.4, 0.55, 0.75, 1.0, 1.3, 1.7, 2.2)
PEOPLE, MARKETS = 20_000, 60           # markets per number of products (2, 3, 4, 5)
rows = {}
for (design, pop_i), s in sorted(base.items()):
    levels = s["levels"]
    rng_m = np.random.default_rng(1000 + s["pop_seed"])
    markets = []
    for m in (2, 3, 4, 5):
        made = 0
        while made < MARKETS:
            prof = np.stack([rng_m.integers(0, l, size=m) for l in levels], axis=1)
            if len({tuple(p) for p in prof}) < m:
                continue
            markets.append(prof)
            made += 1
    for spread in SPREADS:
        spec = dict(s, gen={**s["gen"], "sd_median": s["gen"]["sd_median"] / s["spread"] * spread})
        pop = emplib.tuned_population(levels, spec, s["pop_seed"])
        rng = np.random.default_rng(777)
        if "cov" in pop:
            params = rng.multivariate_normal(pop["means"], pop["cov"], size=PEOPLE, method="eigh")
        else:
            params = pop["means"] + rng.standard_normal((PEOPLE, len(pop["means"]))) * np.sqrt(pop["var"])
        pw = simlib.full_partworths(params, levels)
        var = unit = 0.0
        top = []
        by_m = {}
        for prof in markets:
            p = simlib.softmax(simlib.profile_utilities(pw, prof))
            P = p.mean(axis=0)
            v, u = float(p.var(axis=0).sum()), float((P * (1 - P)).sum())
            var, unit = var + v, unit + u
            a = by_m.setdefault(len(prof), [0.0, 0.0])
            a[0] += v
            a[1] += u
            if len(prof) in (3, 4):
                top.append(P.max())
        rows.setdefault(spread, []).append(dict(design=design, h=var / unit, top=float(np.mean(top)),
                                                by_m={m: a[0] / a[1] for m, a in by_m.items()}))

print("\nspread | h (all populations) | 10% to 90% of the populations x designs | sqrt(h) = share of the classic poll margin | largest share in a market of 3 or 4 | h by products 2 / 3 / 4 / 5")
H = {}
for spread in SPREADS:
    x = rows[spread]
    hs = np.array([r["h"] for r in x])
    H[spread] = float(hs.mean())
    bm = [np.mean([r["by_m"][m] for r in x]) for m in (2, 3, 4, 5)]
    print(f"  {spread:4.2f} | {hs.mean():.3f}               | {np.quantile(hs, .1):.3f} to {np.quantile(hs, .9):.3f}                          | {math.sqrt(hs.mean()):.2f}"
          f"                                        | {100 * np.mean([r['top'] for r in x]):3.0f}%                                 | " + " / ".join(f"{v:.3f}" for v in bm))
print("\nmeasured from drawn samples (stichprobenfehler.py): 0.55 -> 0.356, 1.0 -> 0.574, 1.7 -> 0.708")
print("by design at spread 1.0: " + ", ".join(f"{d}: {np.mean([r['h'] for r in rows[1.0] if r['design'] == d]):.3f}" for d in sorted({r['design'] for r in rows[1.0]})))
print("\nfor the calculator (spread -> h): " + json.dumps({str(k): round(v, 3) for k, v in H.items()}))
