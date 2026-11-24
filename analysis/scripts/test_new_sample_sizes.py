# Packaged copy of the script auswertung_ext2.py. Changed for this archive: it reads the files through _data.py.
"""The formula of the price premium on studies it has not seen (block ext2): 50, 100 and 250 respondents for the designs
that cross 8, 12 and 16 tasks with 2, 3 and 4 options, and 250 respondents for 8 tasks of 2, 10 of 3 and 16 of 4.

The formula was fixed before these studies were run, and they are not part of its fit:
    margin = 2.25 x sqrt(B^2 + V / n),  B = 0.0032 x r^-1.08,  V = 0.136 x r^-1.06,
    r = tasks x (options - 1)^0.19 / utilities        (the price counts as one utility)
Studies: CBC, default settings of the estimation, a feature with two levels that nearly everybody prefers, worth about
8% of the tested price range, typical reaction to price; attributes with 3, 4, 4 levels + feature + price.
Fixed before the runs, for every design with its new sample sizes together: at least 90% of the premiums inside the
margin = the formula holds there; 85 to under 90% = borderline; under 85% = it does not hold.

Usage: python test_new_sample_sizes.py
"""
import json
import math
import sys

import numpy as np

from _data import studies

sys.stdout.reconfigure(encoding="utf-8")
rows = []
for r in studies("premium_studies", "ext2"):
    if r.get("status") != "ok":
        continue
    span = max(r["price"]["values"]) - min(r["price"]["values"])
    ratio = r["tasks"] * (r["options"] - 1) ** 0.19 / (sum(l - 1 for l in r["levels"]) + 1)
    margin = 2.25 * math.sqrt((0.0032 * ratio ** -1.08) ** 2 + 0.136 * ratio ** -1.06 / r["n"])
    for row in r["rows"]:
        if row["app_median"] is None or row["true_median"] is None:
            continue
        e = (row["app_median"] - row["true_median"]) / span
        rows.append(dict(study=r["run_id"], design=(r["tasks"], r["options"]), n=r["n"], pop=r["pop"], z=e / margin, e=100 * abs(e), margin=100 * margin))
studies_n = len({x["study"] for x in rows})
print(f"{studies_n} studies, {len(rows)} premiums")
if not rows:
    sys.exit("no studies found")


def verdict(share):
    return "holds" if share >= 0.90 else "borderline" if share >= 0.85 else "DOES NOT HOLD"


def inside(group):
    return float(np.mean([abs(x["z"]) <= 1 for x in group]))


def with_range(group, draws=2000, seed=11):
    """Share inside, with whole populations drawn again (5% to 95%)."""
    rng = np.random.default_rng(seed)
    pops = sorted({x["pop"] for x in group})
    by_pop = {p: [abs(x["z"]) <= 1 for x in group if x["pop"] == p] for p in pops}
    values = [np.mean(np.concatenate([by_pop[pops[i]] for i in rng.integers(0, len(pops), len(pops))])) for _ in range(draws)]
    return f"{100 * inside(group):5.1f}% ({100 * np.quantile(values, .05):4.1f} to {100 * np.quantile(values, .95):4.1f})"


print("\n== 1 Inside the margin, by design (new sample sizes together; range: populations drawn again, 5% to 95%)")
for design in sorted({x["design"] for x in rows}):
    group = [x for x in rows if x["design"] == design]
    cells = " | ".join(f"N {n}: {100 * inside([x for x in group if x['n'] == n]):5.1f}%, measured 90% ±{np.quantile([x['e'] for x in group if x['n'] == n], .9):4.1f} against ±{[x['margin'] for x in group if x['n'] == n][0]:4.1f}"
                       for n in sorted({x["n"] for x in group}))
    print(f"   {design[0]:2d} tasks of {design[1]}: {with_range(group)} of {len(group):3d} premiums -> {verdict(inside(group))}   || {cells}")
print("\n== 2 By respondents and all together")
for n in sorted({x["n"] for x in rows}):
    group = [x for x in rows if x["n"] == n]
    print(f"   N {n:4d}: {with_range(group)} of {len(group)} premiums | mean error / margin {np.mean([abs(x['z']) for x in group]):.2f}")
print(f"   all:    {with_range(rows)} of {len(rows)} premiums | all three premiums of a study inside: "
      f"{100 * np.mean([all(abs(x['z']) <= 1 for x in rows if x['study'] == s) for s in sorted({x['study'] for x in rows})]):.1f}% of {studies_n} studies")
print(f"   sides: true premium below the lower end {100 * np.mean([x['z'] > 1 for x in rows]):.1f}% | above the upper end {100 * np.mean([x['z'] < -1 for x in rows]):.1f}%")
