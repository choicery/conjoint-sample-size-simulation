"""Writes addon7.json: the middle between 'people agree' and 'people are split' on the priced feature.

Why (23.11.2026): the calculator has two cases for the feature that is priced. 'Most people want it' was measured with
0.1% and 16% of the people rejecting the feature (SD of the gain 0.10 and 0.30, same precision), 'it divides people'
with 44% rejecting it (SD 1.90, about twice the error). Nothing is measured in between, and a typical step between
two levels lies there: in the typical simulated population about a quarter to a third of the people prefer the other
level. The user asked for a 'typical' value and agreed to these runs.

Two spreads of the gain at the same mean gain of 0.30: SD 0.57 (30% reject the feature) and SD 0.98 (38%).
Designs: attributes with 3, 4, 4 levels + the yes/no feature + price, with 8 tasks of 2, 10 tasks of 3 and 16 tasks
of 4 options. Typical price reaction; 150, 400 and 1,000 respondents; 8 populations. Same seeds as the blocks addon
and addon4, so the runs differ from their 'agree' and 'split' twins only in the spread of the gain.
Usage: python plan_addon7.py
"""
import json
from pathlib import Path

import nightlib

HERE = Path(__file__).resolve().parent
gen = json.loads((nightlib.HET / "tuned_gen.json").read_text(encoding="utf-8"))
PRICE = {"values": nightlib.PRICES, "span_mean": 4.0, "cv": 0.87, "dist": "lognormal"}
DESIGNS = {"addon_s8x2": ("s8x2", 8, 2), "addon": ("s10x3", 10, 3), "addon_s16x4": ("s16x4", 16, 4)}
out = []
for design, (key, tasks, options) in DESIGNS.items():
    for gain_sd in (0.57, 0.98):
        arm = f"{key}_typ_sd{round(gain_sd * 100):03d}"
        for pop in range(8):
            for n in (150, 400, 1000):
                pop_seed = 9900 + pop
                out.append(dict(
                    run_id=f"addon7_{arm}_p{pop}_n{n}_r0", arm=arm, design=design, levels=[3, 4, 4, 2], tasks=tasks, options=options,
                    method="cbc", n=n, rep=0, pop=pop, gen=gen, price=PRICE, price_constraint=None, prior_variance=None,
                    hb_seed=42, pop_seed=pop_seed, data_seed=pop_seed * 10 + n,
                    addon={"attr": 3, "gain_mean": 0.30, "gain_sd": gain_sd}, design_path="page",
                    wtp_questions=10, wtp_questions_rfc=0, wtp_questions_robust=0, _results="results_addon7", _work="addon",
                    stage="addon7", **nightlib.DEFAULT_LENGTH))
out.sort(key=lambda s: -nightlib.cost(s))
assert len({s["run_id"] for s in out}) == len(out)
(HERE / "addon7.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("addon7.json", len(out), "runs")
