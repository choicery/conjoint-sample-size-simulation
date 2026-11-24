"""The numbers behind Figures 1 to 3 of the article and the comparisons quoted with them. Written for this archive.

Studies: CBC, default settings of the estimation, the reference case of the calculator (typical reaction to price, a
level worth about 8% of the tested price range). The margin is the one of the calculator:
    margin = 2.25 x sqrt(B^2 + V / n) x (levels / 2)^0.43,  B = 0.0032 x r^-1.08,  V = 0.136 x r^-1.06,
    r = tasks x (options - 1)^0.19 / utilities        (the price counts as one utility)
Ranges in brackets: the same value with the simulated populations drawn again (5% to 95%).
"""
import math
import sys
from collections import defaultdict

import numpy as np

from _data import studies

sys.stdout.reconfigure(encoding="utf-8")
OTHER_SETTINGS = {"agree_long", "agree_pv1", "agree_pv025", "thin_agree_pv025", "split_pv025"}      # longer chains or another prior variance
rng = np.random.default_rng(20261124)


def margin(tasks, options, utilities, n, levels=2):
    r = tasks * (options - 1) ** 0.19 / utilities
    return 100 * 2.25 * math.sqrt((0.0032 * r ** -1.08) ** 2 + 0.136 * r ** -1.06 / n) * (levels / 2) ** 0.43


rows = []
for r in list(studies("premium_studies", "addon")) + list(studies("premium_studies", "ext1")) + list(studies("premium_studies", "ext2")):
    a = r.get("addon")
    if not a or r.get("status") != "ok" or r["arm"] in OTHER_SETTINGS or (r.get("hb_prior_variance") or 4.0) != 4.0 or r["price"]["span_mean"] < 3 or a["gain_mean"] > 0.5:
        continue
    span = max(r["price"]["values"]) - min(r["price"]["values"])
    errors = [100 * abs(x["app_median"] - x["true_median"]) / span for x in r["rows"] if x["app_median"] is not None and x["true_median"] is not None]
    if errors:
        rows.append(dict(T=r["tasks"], O=r["options"], attrs=tuple(r["levels"]), L=r["levels"][a["attr"]], n=r["n"], pop=r["pop"], sd=round(float(a["gain_sd"]), 2), e=errors))


def cell(**want):
    return [s for s in rows if all(s[k] == v for k, v in want.items())]


def q90(group, draws=2000):
    by_pop = defaultdict(list)
    for s in group:
        by_pop[s["pop"]].extend(s["e"])
    pops = list(by_pop)
    point = float(np.quantile(np.concatenate([by_pop[p] for p in pops]), 0.9))
    values = [np.quantile(np.concatenate([by_pop[pops[i]] for i in rng.integers(0, len(pops), len(pops))]), 0.9) for _ in range(draws)]
    return f"{point:5.2f} ({np.quantile(values, 0.05):5.2f} to {np.quantile(values, 0.95):5.2f})"


def paired(top, base, key, draws=2000):
    """Mean absolute error of the studies in top against the studies in base of the same cell (key) and population."""
    sums, base_mean = defaultdict(lambda: [0.0, 0.0, 0]), defaultdict(list)
    for s in base:
        base_mean[(key(s), s["pop"])].append(float(np.mean(s["e"])))
    for s in top:
        partner = base_mean.get((key(s), s["pop"]))
        if partner:
            sums[s["pop"]][0] += float(np.mean(s["e"]))
            sums[s["pop"]][1] += float(np.mean(partner))
            sums[s["pop"]][2] += 1
    pops = list(sums)
    values = []
    for _ in range(draws):
        pick = [pops[i] for i in rng.integers(0, len(pops), len(pops))]
        values.append(sum(sums[p][0] for p in pick) / sum(sums[p][1] for p in pick))
    return (f"{sum(sums[p][0] for p in pops) / sum(sums[p][1] for p in pops):4.2f} ({np.quantile(values, 0.05):4.2f} to {np.quantile(values, 0.95):4.2f}),"
            f" {sum(sums[p][2] for p in pops)} studies, {len(pops)} populations")


print(f"{len(rows)} studies of the reference case")
print("\n== Figure 1: 8, 12 and 16 tasks (attributes 3, 4, 4 + feature + price), the studies with 2, 3 and 4 options per task;")
print("   each error measured against the margin of its own design and shown on the scale of 3 options: measured 90% value | margin for 3 options | studies")
for tasks in (8, 12, 16):
    for n in (50, 100, 150, 250, 400, 1000):
        group = [dict(s, e=[x * margin(tasks, 3, 10, n) / margin(s["T"], s["O"], 10, n) for x in s["e"]])
                 for s in rows if s["T"] == tasks and s["O"] in (2, 3, 4) and s["attrs"] == (3, 4, 4, 2) and s["sd"] == 0.10 and s["n"] == n]
        print(f"   {tasks:2d} tasks, N {n:5d}: {q90(group)} | {margin(tasks, 3, 10, n):5.2f} | {len(group)}")

print("\n== Tasks against options: ratio of the mean absolute errors, 50 to 1,000 respondents together")
standard = [s for s in rows if s["attrs"] == (3, 4, 4, 2) and s["sd"] == 0.10 and s["n"] in (50, 100, 150, 250, 400, 1000)]
for a, b in (((16, 2), (8, 3)), ((12, 2), (10, 3)), ((12, 3), (8, 4))):
    print(f"   {a[0]} tasks of {a[1]} against {b[0]} tasks of {b[1]}: " + paired([s for s in standard if (s["T"], s["O"]) == a], [s for s in standard if (s["T"], s["O"]) == b], lambda s: s["n"]))

print("\n== Figure 2: levels of the priced attribute, 10 tasks of 3 options, 400 respondents: measured 90% value | margin of the calculator | studies")
for levels in (2, 3, 4, 5, 6, 8, 10, 12):
    group = cell(T=10, O=3, attrs=(3, 4, 4, levels), sd=0.10, n=400)
    print(f"   {levels:2d} levels: {q90(group)} | {margin(10, 3, 8 + levels, 400, levels):5.2f} | {len(group)}")      # 2 + 3 + 3 utilities of the other attributes, levels - 1, and 1 for the price
agree = [s for s in rows if s["sd"] == 0.10 and s["attrs"][:3] == (3, 4, 4) and len(s["attrs"]) == 4 and s["n"] <= 1000]
print("   mean absolute error against two levels in the same design, with the same sample size and populations:")
for levels in (3, 4, 5, 6):
    print(f"      {levels} levels: " + paired([s for s in agree if s["L"] == levels], [s for s in agree if s["L"] == 2], lambda s: (s["T"], s["O"], s["attrs"][:-1], s["n"])))

print("\n== Figure 3: share of people who prefer the other level, attributes with two levels: mean absolute error against a level nearly everybody prefers")
binary = [s for s in rows if s["L"] == 2 and s["n"] <= 2000]
phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
for sd in (0.30, 0.57, 0.98, 1.90):
    print(f"   {100 * phi(-0.30 / sd):4.1f}% prefer the other level: " + paired([s for s in binary if s["sd"] == sd], [s for s in binary if s["sd"] == 0.10], lambda s: (s["T"], s["O"], s["attrs"], s["n"])))
