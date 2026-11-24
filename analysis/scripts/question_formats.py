"""Price premium with dual-response, best-worst and best-worst with a buy question, against the CBC study with the same
people, tasks and prices (block fmt1, 384 studies). Packaged copy of the evaluation as it ran, reading through _data.py.

Error = |premium of the market simulator - true premium| in percent of the tested price range. A study gives three
premiums that are close to each other, so the unit is the study: its error is the mean of the three. Ratio = mean error
with the format / mean error with CBC over the same studies. Uncertainty: the populations are drawn again with
replacement (all studies of a population stay together), 4,000 times; the range holds 90% of the ratios.
"""
import sys
from collections import defaultdict

import numpy as np

from _data import files

sys.stdout.reconfigure(encoding="utf-8")
NAMES = {"dual_response": "Dual-response", "best_worst": "Best-worst", "best_worst_with_buy_question": "Best-worst with buy question"}
DESIGN = {"addon": "10 tasks of 3", "addon_s16x4": "16 tasks of 4", "addon_s8x2": "8 tasks of 2"}
ALL = dict(files("premium_studies"))


def study_error(r):
    price_range = max(r["price"]["values"]) - min(r["price"]["values"])
    e = [100 * abs(x["app_median"] - x["true_median"]) / price_range for x in r["rows"] if x["app_median"] is not None and x["true_median"] is not None]
    return float(np.mean(e)), e


rows = []
for name in sorted(ALL):
    r = ALL[name]
    if r["family"] != "fmt1":
        continue
    e, values = study_error(r)
    e_cbc, values_cbc = study_error(ALL[r["cbc_partner"]])
    split = r["addon"]["gain_sd"] > 1.5
    rows.append(dict(method=r["question_format"], design=r["design"], split=split, n=r["n"], cluster=(r["design"], split, r["pop"]), e=e, cbc=e_cbc,
                     values=values, values_cbc=values_cbc))
print(f"{len(rows)} studies with another question format; error in percent of the price range")
rng = np.random.default_rng(20261123)


def line(label, cell, boot=4000):
    if not cell:
        return
    clusters = defaultdict(list)
    for r in cell:
        clusters[r["cluster"]].append(r)
    keys = list(clusters)
    sums = np.array([[sum(r["e"] for r in clusters[k]), sum(r["cbc"] for r in clusters[k])] for k in keys])
    p = sums[:, 0].sum() / sums[:, 1].sum()
    b = sums[rng.integers(0, len(keys), size=(boot, len(keys)))]
    ratios = b[:, :, 0].sum(axis=1) / b[:, :, 1].sum(axis=1)
    lo, hi = float(np.quantile(ratios, 0.05)), float(np.quantile(ratios, 0.95))
    print(f"   {label:44s} error x {p:4.2f} ({lo:4.2f} to {hi:4.2f}) | respondents x {p * p:4.2f} ({lo * lo:4.2f} to {hi * hi:4.2f}) | "
          f"{len(cell):3d} studies, {len(keys):2d} populations | smaller than CBC in {sum(r['e'] < r['cbc'] for r in cell):3d}")


print("\n== 1 all studies of a format: error against CBC (1.00 = the same), with the 5% to 95% range")
for method in NAMES:
    line(NAMES[method], [r for r in rows if r["method"] == method])
print("\n== 2 by how much people agree on the feature (10 tasks of 3 options)")
for method in NAMES:
    for split in (False, True):
        line(f"{NAMES[method]}, {'split' if split else 'agree'}", [r for r in rows if r["method"] == method and r["design"] == "addon" and r["split"] == split])
print("\n== 3 by design (people agree)")
for method in NAMES:
    for design in DESIGN:
        line(f"{NAMES[method]}, {DESIGN[design]}", [r for r in rows if r["method"] == method and r["design"] == design and not r["split"]])
print("\n== 4 by sample size (all designs)")
for method in NAMES:
    for n in sorted({r["n"] for r in rows}):
        line(f"{NAMES[method]}, N {n}", [r for r in rows if r["method"] == method and r["n"] == n])
print("\n== 5 what the task factors measured on choice shares (1.4, 1.8, 2.2 CBC tasks per task) give in the formula of the premium")
for method, factor in (("dual_response", 1.4), ("best_worst", 1.8), ("best_worst_with_buy_question", 2.2)):
    def mae(ratio, n):
        return np.sqrt((0.0087 * ratio ** -0.60) ** 2 + 0.119 * ratio ** -0.39 / n)
    expect = np.mean([mae(2.0 * factor, n) / mae(2.0, n) for n in (150, 400, 1000)])
    print(f"   {NAMES[method]:30s} expected error x {expect:4.2f} (10 tasks of 3 options, {factor} times the tasks)")
