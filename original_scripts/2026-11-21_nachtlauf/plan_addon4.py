"""Writes addon4.json: the price premium of a yes/no add-on over several designs (run: python run_night.py addon4.json <workers>).

Why (23.11.2026): the user wants a calculator that does not depend on the prior variance. For a yes/no add-on the robust
premium hardly depends on it (blocks addon to addon3), but only two designs and only the typical price reaction were
measured. This block adds what a formula over designs needs:

 - two level structures:  S = 3, 4, 4 levels + the add-on (9 utilities + price),  L = 4, 4, 5, 5 levels + the add-on (15 + price)
 - three task settings:   8 tasks of 2, 10 tasks of 3, 16 tasks of 4
   -> S 10x3 is the design of block addon ('addon'), L 8x2 the thin design of block addon2 ('addon_thin'); four are new
 - price reaction typical (utility span 4.0 over the tested prices) and weak (2.0)
 - people agree on the add-on (SD of the gain 0.10) or are split (1.90)
 - 150, 400 and 1,000 respondents, 8 populations, one data set each
 - a bigger add-on (mean gain 1.0 instead of 0.30) in a few cells, to see how the error grows with the premium

The core cells come first (longest runs first), the bigger add-on last, so that a stop leaves a usable part.
Usage: python plan_addon4.py [--smoke]     (--smoke writes addon4_smoke.json: five small runs)
"""
import json
import sys
from pathlib import Path

import nightlib

HERE = Path(__file__).resolve().parent
gen = json.loads((nightlib.HET / "tuned_gen.json").read_text(encoding="utf-8"))
PRICE = {"typ": {"values": nightlib.PRICES, "span_mean": 4.0, "cv": 0.87, "dist": "lognormal"},
         "weak": {"values": nightlib.PRICES, "span_mean": 2.0, "cv": 0.87, "dist": "lognormal"}}
AGREEMENT = {"agree": 0.10, "split": 1.90}
S, L = [3, 4, 4, 2], [4, 4, 5, 5, 2]
DESIGNS = {      # name of the design: key used in the arm, levels, tasks, options
    "addon": ("s10x3", S, 10, 3), "addon_thin": ("l8x2", L, 8, 2),
    "addon_s8x2": ("s8x2", S, 8, 2), "addon_l10x3": ("l10x3", L, 10, 3),
    "addon_s16x4": ("s16x4", S, 16, 4), "addon_l16x4": ("l16x4", L, 16, 4),
}
NEW = ("addon_s8x2", "addon_l10x3", "addon_s16x4", "addon_l16x4")
NS = (150, 400, 1000)
POPULATIONS = 8
SMOKE = "--smoke" in sys.argv
RESULTS = "results_addon4_smoke" if SMOKE else "results_addon4"


def specs(design, price, agreement, ns, gain_mean=0.30, pops=range(POPULATIONS)):
    key, levels, tasks, options = DESIGNS[design]
    arm = f"{key}_{price}_{agreement}" + ("_big" if gain_mean > 0.5 else "")
    out = []
    for pop in pops:
        for n in ns:
            pop_seed = 9900 + pop      # as in the blocks addon to addon3
            out.append(dict(
                run_id=f"addon4_{arm}_p{pop}_n{n}_r0", arm=arm, design=design, levels=list(levels), tasks=tasks, options=options,
                method="cbc", n=n, rep=0, pop=pop, gen=gen, price=PRICE[price], price_constraint=None, prior_variance=None,
                hb_seed=42, pop_seed=pop_seed, data_seed=pop_seed * 10 + n,
                addon={"attr": len(levels) - 1, "gain_mean": gain_mean, "gain_sd": AGREEMENT[agreement]}, design_path="page",
                wtp_questions=10, wtp_questions_rfc=0, wtp_questions_robust=0, _results=RESULTS, _work="addon",
                stage="addon4", **nightlib.DEFAULT_LENGTH))
    return out


core, big = [], []
if SMOKE:
    for design in NEW:
        core += specs(design, "typ", "agree", (150,), pops=(0,))
    core += specs("addon_l16x4", "weak", "split", (150,), pops=(0,))
else:
    for design in NEW:                                   # four new designs, complete
        for price in PRICE:
            for agreement in AGREEMENT:
                core += specs(design, price, agreement, NS)
    for design in ("addon", "addon_thin"):               # the two measured designs: weak price reaction is missing
        for agreement in AGREEMENT:
            core += specs(design, "weak", agreement, NS)
    for agreement in AGREEMENT:                          # the thin design has no runs with 150 respondents yet
        core += specs("addon_thin", "typ", agreement, (150,))
    big += specs("addon", "typ", "agree", (150, 1000), gain_mean=1.0)        # 400 and 2,000 exist (block addon2)
    big += specs("addon", "typ", "split", (400, 1000), gain_mean=1.0)
    big += specs("addon", "weak", "agree", (400, 1000), gain_mean=1.0)
    big += specs("addon_thin", "typ", "agree", (400, 1000), gain_mean=1.0)
    big += specs("addon_s16x4", "typ", "agree", (400, 1000), gain_mean=1.0)
    big += specs("addon_l10x3", "typ", "agree", (400, 1000), gain_mean=1.0)
core.sort(key=lambda s: -nightlib.cost(s))
big.sort(key=lambda s: -nightlib.cost(s))
out = core + big
assert len({s["run_id"] for s in out}) == len(out)
name = "addon4_smoke.json" if SMOKE else "addon4.json"
(HERE / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
arms = {}
for s in out:
    arms[s["arm"]] = arms.get(s["arm"], 0) + 1
ratio = {k: round(t * (o - 1) / (sum(x - 1 for x in lv) + 1), 2) for d, (k, lv, t, o) in DESIGNS.items()}
print(name, len(out), "runs:", len(core), "core,", len(big), "bigger add-on")
print("comparisons per utility (price counted as one):", ratio)
print(arms)
