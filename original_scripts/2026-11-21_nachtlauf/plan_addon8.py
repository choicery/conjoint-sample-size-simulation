"""Writes addon8.json: the price premium with 50 and 100 respondents.

Why (23.11.2026): the premium was simulated with 150 to 2,000 respondents only, so the calculator shows no value for
smaller samples (for example for Johnson's rule with 84 respondents). The user asked why 50 and 100 were left out and
agreed to these runs.

Yes/no feature that people agree on (mean gain 0.30, SD 0.10), typical price reaction; attributes with 3, 4, 4 levels
+ the feature + price, with 8 tasks of 2, 10 tasks of 3 and 16 tasks of 4 options; 50 and 100 respondents;
8 populations x 2 data sets. Same seeds as the blocks addon and addon4.
Usage: python plan_addon8.py
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
    arm = f"{key}_typ_agree"
    for pop in range(8):
        for n in (50, 100):
            for rep in (0, 1):
                pop_seed = 9900 + pop
                out.append(dict(
                    run_id=f"addon8_{arm}_p{pop}_n{n}_r{rep}", arm=arm, design=design, levels=[3, 4, 4, 2], tasks=tasks, options=options,
                    method="cbc", n=n, rep=rep, pop=pop, gen=gen, price=PRICE, price_constraint=None, prior_variance=None,
                    hb_seed=42, pop_seed=pop_seed, data_seed=pop_seed * 10 + 1000 * rep + n,
                    addon={"attr": 3, "gain_mean": 0.30, "gain_sd": 0.10}, design_path="page",
                    wtp_questions=10, wtp_questions_rfc=0, wtp_questions_robust=0, _results="results_addon8", _work="addon",
                    stage="addon8", **nightlib.DEFAULT_LENGTH))
out.sort(key=lambda s: -nightlib.cost(s))
assert len({s["run_id"] for s in out}) == len(out)
(HERE / "addon8.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("addon8.json", len(out), "runs")
