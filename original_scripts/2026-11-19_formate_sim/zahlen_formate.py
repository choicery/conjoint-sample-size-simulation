"""Numbers and Figure 3 of the draft article from the question-format runs (19.11.2026).

Reads:  results_formate/ and the paired CBC runs (through auswertung_formate.load), formate.json (auswertung_formate.py).
Writes: zahlen_formate.json (read by ../2026-11-18_blog_neu/blog_entwurf/build_draft.py as Z["F"])
        ../2026-11-18_blog_neu/grafiken/grafik3_neu.svg.html

The calculator of the draft uses the rounded factors in CALC_NEW (sample-size-math.js, TASK_FACTOR); the build checks that.
"""
import json
import sys
from pathlib import Path

import numpy as np

import auswertung_formate as A

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "2026-11-18_blog_neu" / "grafiken"
CALC_NEW = {"drcbc": 1.4, "bwcbc": 1.8, "bwcbc_none": 2.2}
SHORT = {"drcbc": "dr", "bwcbc": "bw", "bwcbc_none": "bwn", "cbc": "cbc"}
EXAMPLE_R = 10 * 2 / 14      # example design of the page: levels 3, 4, 4, 3, 5, with 10 tasks and 3 options


def de(x, nd=1):
    return f"{x:.{nd}f}"


def needed(ratio, m, k=A.K90):
    room = (m / k) ** 2 - (A.B * ratio ** A.BE) ** 2
    return None if room <= 0 else int(np.ceil(A.V * ratio ** A.VE / room - 1e-9))


res = json.loads((HERE / "formate.json").read_text(encoding="utf-8"))
rows, runs = A.load()
sop = rows[rows["rule"] == "share_of_preference"]
po = sop[sop["market"].isin(["std", "extra"])]          # markets without None, identical for every format and for CBC
assert not po["is_none"].any()
done = runs[runs["method"].isin(A.FORMATS)]
assert len(done) == 1134 and (done["status"] == "ok").all() and int(done["hb_warnings"].sum()) == 0

F = {"RUNS": f"{len(done):,}", "DESIGNS": int(done["design"].nunique()),
     "DESIGNS_BW": int(done.loc[done["method"] == "bwcbc", "design"].nunique()),
     "POPULATIONS": int(done["pop"].nunique()), "CALC": {SHORT[m]: de(f) for m, f in CALC_NEW.items()}}
assert F["DESIGNS"] == 23 and F["DESIGNS_BW"] == 20 and F["POPULATIONS"] == 3
for m in A.FORMATS:
    s, r = SHORT[m], res[m]
    F[s] = {"f": de(r["f_rel"], 2), "lo": de(r["f_rel_lo"], 2), "hi": de(r["f_rel_hi"], 2), "runs": int(r["runs"]),
            "rich": de(r["f_rel r ab 2,5"], 1), "recovery": de(r["f_recovery"], 1),
            "cov90": de(A.cover(po[po["method"] == m], CALC_NEW[m])), "cov95": de(A.cover(po[po["method"] == m], CALC_NEW[m], A.K95), 0),
            "n8": f"{needed(EXAMPLE_R * CALC_NEW[m], 8):,}"}
    # the rounded factor of the calculator lies inside the measured 90 % range
    assert r["f_rel_lo"] - 0.02 <= CALC_NEW[m] <= r["f_rel_hi"] + 0.02, (m, r["f_rel_lo"], r["f_rel_hi"])
cbc = po[po["method"] == "cbc"]
F["cbc"] = {"cov90": de(A.cover(cbc, 1.0)), "n8": f"{needed(EXAMPLE_R, 8):,}"}

# root mean square error with 400 people, cells that have every format
cells_all = set.intersection(*[set(po.loc[po["method"] == m, "cell"]) for m in po["method"].unique()])
q = po[po["cell"].isin(cells_all) & (po["n"] == 400)]
F["RMS400"] = {SHORT[m]: de(float(np.sqrt((g["err"] ** 2).mean()))) for m, g in q.groupby("method")}
F["RMS_CELLS"] = len(cells_all)

# markets that include None (dual-response and best-worst with None), margin with the calculator's factor
none = sop[sop["market"] == "none"]
bias, cov_none, cov_prod = [], [], []
for m in ("drcbc", "bwcbc_none"):
    g = none[none["method"] == m]
    bias.append(float(g.loc[g["is_none"], "err"].mean()))
    cov_none.append(A.cover(g[g["is_none"]], CALC_NEW[m]))
    cov_prod.append(A.cover(g[~g["is_none"]], CALC_NEW[m]))
F["NONE_BIAS_MIN"], F["NONE_BIAS_MAX"] = de(min(bias)), de(max(bias))
F["NONE_COV_MIN"], F["NONE_COV_MAX"] = de(min(cov_none), 0), de(max(cov_none), 0)
F["NONE_PROD_COV_MIN"], F["NONE_PROD_COV_MAX"] = de(min(cov_prod), 0), de(max(cov_prod), 0)
nobuy = done.loc[done["method"].isin(["drcbc", "bwcbc_none"]), "nobuy"]
F["NOBUY"] = de(100 * float(nobuy.mean()), 0)
# factors per population, to back "similar across populations"
per_pop = {}
for m in A.FORMATS:
    vals = []
    for p in (0, 1, 2):
        qq = po[(po["method"] == m) & (po["pop"] == p)]
        rr = po[(po["method"] == "cbc") & (po["pop"] == p) & po["cell"].isin(qq["cell"].unique())]
        vals.append(A.fit_factor(A.cells_of(qq)) / A.fit_factor(A.cells_of(rr)))
    per_pop[SHORT[m]] = [round(v, 2) for v in vals]
    F[SHORT[m]]["pop_min"], F[SHORT[m]]["pop_max"] = de(min(vals), 1), de(max(vals), 1)
F["PER_POP"] = per_pop
F["BW_TASKS_AS_CBC"] = int(round(10 * CALC_NEW["bwcbc"]))
# few reachable people: 90 % margin of the example design with 100 people, with the factors of the calculator
F["M100"] = {SHORT[m]: de(A.K90 * float(np.sqrt(A.pred2(EXAMPLE_R, 100, f))))
             for m, f in {"cbc": 1.0, **CALC_NEW}.items()}

(HERE / "zahlen_formate.json").write_text(json.dumps(F, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(F, indent=1, ensure_ascii=False))

# ---- Figure 3: worth of one task in CBC tasks, with the 90 % range from resampling designs -----------------------
X0, SCALE = 262, 133.0
bars = [("CBC", 1.0, None, "#c9d6d1"),
        ("Dual-response", res["drcbc"]["f_rel"], F["dr"], "#1a9c8d"),
        ("Best-worst", res["bwcbc"]["f_rel"], F["bw"], "#1a9c8d"),
        ("Best-worst with None", res["bwcbc_none"]["f_rel"], F["bwn"], "#1a9c8d")]
labels, rects, values, ranges = [], [], [], []
for i, (name, f, info, color) in enumerate(bars):
    y = 58 + 64 * i
    w = round(SCALE * f)
    labels.append(f'<text x="0" y="{y + 26}">{name}</text>')
    rects.append(f'<rect fill="{color}" height="36" rx="6" width="{w}" x="{X0}" y="{y}"></rect>')
    values.append(f'<text x="{X0 + w + 10}" y="{y + 27}">{de(f, 2) if info else "1"}</text>')
    ranges.append(f'<text x="760" y="{y + 27}">{info["lo"] + " to " + info["hi"] if info else "reference"}</text>')
title = ("Worth of one task in CBC tasks, measured on choice shares: "
         f"dual-response CBC {F['dr']['f']} (90% range {F['dr']['lo']} to {F['dr']['hi']}), "
         f"best-worst CBC {F['bw']['f']} ({F['bw']['lo']} to {F['bw']['hi']}), "
         f"best-worst CBC with None {F['bwn']['f']} ({F['bwn']['lo']} to {F['bwn']['hi']}).")
svg = f"""<svg aria-labelledby="sim3-title" role="img" viewBox="0 0 760 340">
<title id="sim3-title">{title}</title>
<text fill="#55635f" font-size="17" x="{X0}" y="34">One task counts as this many CBC tasks</text>
<text fill="#55635f" font-size="17" text-anchor="end" x="760" y="34">90% range</text>
<g fill="#0d1412" font-size="20" font-weight="600">{''.join(labels)}</g>
{''.join(rects)}
<g fill="#0d1412" font-size="24" font-weight="800">{''.join(values)}</g>
<g fill="#55635f" font-size="18" text-anchor="end">{''.join(ranges)}</g>
<text fill="#0f6f64" font-size="19" x="0" y="328">10 best-worst tasks carry about as much as {F['BW_TASKS_AS_CBC']} CBC tasks.</text>
</svg>
"""
(FIGS / "grafik3_neu.svg.html").write_text(svg, encoding="utf-8", newline="\n")
print("figure written")
