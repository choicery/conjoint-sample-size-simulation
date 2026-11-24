"""Writes addon5.json: a first look at the premium for a step between two levels of an attribute with MORE than two levels.

Why (23.11.2026): the formula of block addon4 holds for a yes/no attribute only. Most designs have attributes with
three or more levels (the user's example: 3, 4, 4, 3), so the question is how much less exact the premium gets when
the priced attribute has 3, 4, 5 or 6 levels. The user asked for a quick, small run first.

Design: the main design of the blocks addon and addon4 (attributes with 3, 4 and 4 levels, the priced attribute last,
price with five levels, 10 tasks of 3 options), with the priced attribute at 3, 4, 5 and 6 levels. Typical price
reaction, people agree on the step (mean gain 0.30, SD 0.10), 400 respondents, 8 populations x 2 data sets.
The reference with two levels are the 16 studies 'agree' with 400 respondents of block addon (same seeds).
Usage: python plan_addon5.py [--smoke]
"""
import json
import sys
from pathlib import Path

import nightlib

HERE = Path(__file__).resolve().parent
gen = json.loads((nightlib.HET / "tuned_gen.json").read_text(encoding="utf-8"))
PRICE = {"values": nightlib.PRICES, "span_mean": 4.0, "cv": 0.87, "dist": "lognormal"}
SMOKE = "--smoke" in sys.argv
RESULTS = "results_addon5_smoke" if SMOKE else "results_addon5"
out = []
for n_levels in ((3, 6) if SMOKE else (3, 4, 5, 6)):
    levels = [3, 4, 4, n_levels]
    for pop in ((0,) if SMOKE else range(8)):
        for rep in ((0,) if SMOKE else (0, 1)):
            n = 150 if SMOKE else 400
            pop_seed = 9900 + pop
            arm = f"s10x3_l{n_levels}_typ_agree"
            out.append(dict(
                run_id=f"addon5_{arm}_p{pop}_n{n}_r{rep}", arm=arm, design=f"addon_levels{n_levels}", levels=levels, tasks=10, options=3,
                method="cbc", n=n, rep=rep, pop=pop, gen=gen, price=PRICE, price_constraint=None, prior_variance=None,
                hb_seed=42, pop_seed=pop_seed, data_seed=pop_seed * 10 + 1000 * rep + n,
                addon={"attr": 3, "gain_mean": 0.30, "gain_sd": 0.10}, design_path="page",
                wtp_questions=10, wtp_questions_rfc=0, wtp_questions_robust=0, _results=RESULTS, _work="addon",
                stage="addon5", **nightlib.DEFAULT_LENGTH))
out.sort(key=lambda s: -nightlib.cost(s))
assert len({s["run_id"] for s in out}) == len(out)
name = "addon5_smoke.json" if SMOKE else "addon5.json"
(HERE / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(name, len(out), "runs;", {s["design"]: round(s["tasks"] * (s["options"] - 1) / (sum(l - 1 for l in s["levels"]) + 1), 2) for s in out})
