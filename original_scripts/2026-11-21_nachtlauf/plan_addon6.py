"""Writes addon6.json: more levels of the priced attribute, on more designs and sample sizes (follows block addon5).

Block addon5 showed for one design and 400 respondents that the premium gets less exact when the priced attribute has
3 to 6 levels (about with the square root of the number of levels, but roughly measured). This block makes the rule
firm: the same attributes (3, 4, 4 levels + the priced attribute + price) with
  - 10 tasks of 3 options at 150 and 1,000 respondents (400 exists in addon5),
  - 8 tasks of 2 options and 16 tasks of 4 options at 400 respondents,
each with 3, 4, 5 and 6 levels of the priced attribute, typical price reaction, people agree on the step.
For the two new task settings the second data set with two levels is added (the first one exists in block addon4).
Usage: python plan_addon6.py
"""
import json
from pathlib import Path

import nightlib

HERE = Path(__file__).resolve().parent
gen = json.loads((nightlib.HET / "tuned_gen.json").read_text(encoding="utf-8"))
PRICE = {"values": nightlib.PRICES, "span_mean": 4.0, "cv": 0.87, "dist": "lognormal"}
out = []


def add(key, tasks, options, n_levels, n, reps):
    levels = [3, 4, 4, n_levels]
    arm = f"{key}_l{n_levels}_typ_agree"
    for pop in range(8):
        for rep in reps:
            pop_seed = 9900 + pop
            out.append(dict(
                run_id=f"addon6_{arm}_p{pop}_n{n}_r{rep}", arm=arm, design=f"addon_{key}_levels{n_levels}", levels=levels, tasks=tasks,
                options=options, method="cbc", n=n, rep=rep, pop=pop, gen=gen, price=PRICE, price_constraint=None, prior_variance=None,
                hb_seed=42, pop_seed=pop_seed, data_seed=pop_seed * 10 + 1000 * rep + n,
                addon={"attr": 3, "gain_mean": 0.30, "gain_sd": 0.10}, design_path="page",
                wtp_questions=10, wtp_questions_rfc=0, wtp_questions_robust=0, _results="results_addon6", _work="addon",
                stage="addon6", **nightlib.DEFAULT_LENGTH))


for n_levels in (3, 4, 5, 6):
    for n in (150, 1000):
        add("s10x3", 10, 3, n_levels, n, (0,))
    add("s8x2", 8, 2, n_levels, 400, (0, 1))
    add("s16x4", 16, 4, n_levels, 400, (0, 1))
add("s8x2", 8, 2, 2, 400, (1,))          # two levels: data set 0 is in block addon4
add("s16x4", 16, 4, 2, 400, (1,))
out.sort(key=lambda s: -nightlib.cost(s))
assert len({s["run_id"] for s in out}) == len(out)
(HERE / "addon6.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
cells = {}
for s in out:
    k = (s["design"].split("_levels")[0], s["n"])
    cells[k] = cells.get(k, 0) + 1
print("addon6.json", len(out), "runs", cells)
