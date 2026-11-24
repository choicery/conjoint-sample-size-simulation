"""Blog evaluation of 17.11. (auswertung_gesamtfehler.py), method unchanged, on the runs with the tuned generator.

User decision 18.11.2026: rebuild the whole simulation section on the tuned-generator runs ("Alles neu aufbauen")
and refit the formula on them ("Neu angepasst"). The formula was fixed in FORMEL_18_11.md before the check runs.

Nothing in the folders of 17.11. is changed: the old script is imported and its sources and outputs are redirected.
  main runs    2026-11-18_heterogenitaet_echte_studien/results_tuned        (67 designs, 4 populations, 150/400/1,000)
  small        .../results_tuned_small                                       (50/100/250), when present
  check runs   .../results_tuned_check                                       (example design, 3 new designs), when present
Writes into this folder: ergebnisse_gesamtfehler.txt, tabellen/, plots/, grafiken/ (the SVG fragments of the figures).

The old script selects its main runs with hetero == "realistic" and folder in (results, results_more, results_pop).
The loader below labels the tuned runs that way so that the old code path runs unchanged; the label has no other meaning.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "2026-11-17_sample_size_auswertung"
HET = HERE.parent / "2026-11-18_heterogenitaet_echte_studien"
sys.path.insert(0, str(OLD))
import auswertung as A  # noqa: E402

# The script of 17.11. with five small changes, written next to this file as gesamtfehler_neu.py:
# the example target of the hold-out checks is +-8 instead of +-7 (with the new formula +-7 is out of reach at 95 %),
# and the heading inside figure 4 is shorter (it ran into "Best reachable").
PATCHES = [
    ("ex_hold[level].append(needed(q, EXAMPLE_R, 7, k))", "ex_hold[level].append(needed(q, EXAMPLE_R, 8, k) or 10 ** 9)"),
    ("größte {hold_out[level]['max']} | Beispiel ±7: ", "größte {hold_out[level]['max']} | Beispiel ±8: "),
    ("geprüft (90 %): {pct((zscore(q, te) <= k).groupby(te['n']).mean())} | Beispiel ±7: ",
     "geprüft (90 %): {pct((zscore(q, te) <= k).groupby(te['n']).mean())} | Beispiel ±8: "),
    ('f"{needed(q, EXAMPLE_R, 7, k)}")', 'f"{needed(q, EXAMPLE_R, 8, k)}")'),
    # the factors are the ones fixed in FORMEL_18_11.md, not rounded again here (measured: see the line "genau")
    ('par["k90"], par["k95"] = round(float(zr.quantile(0.90)), 2), round(float(zr.quantile(0.95)), 2)',
     'par["k90"], par["k95"] = PREREG["k90"], PREREG["k95"]; say(f"gemessene Quantile zur gerundeten Formel: {float(zr.quantile(0.90)):.3f} / {float(zr.quantile(0.95)):.3f}")'),
    (">90% margin with 400 people, percentage points</text>", ">Margin with 400 people, percentage points</text>"),
]
_src = (OLD / "auswertung_gesamtfehler.py").read_text(encoding="utf-8")
for _a, _b in PATCHES:
    assert _src.count(_a) == 1, (_src.count(_a), _a)
    _src = _src.replace(_a, _b)
(HERE / "gesamtfehler_neu.py").write_text(_src, encoding="utf-8", newline="\n")
sys.path.insert(0, str(HERE))
import gesamtfehler_neu as G  # noqa: E402

FORMULA = dict(b=4.3, be=-0.57, v=5000.0, ve=-0.81, k90=1.65, k95=2.0)  # FORMEL_18_11.md
FOLDERS = {"results_small": "results_tuned_small", "results_check": "results_tuned_check", "results_range": "results_tuned_range"}

for mod in (A, G):
    mod.OUT, mod.PLOTS, mod.TABLES, mod.BLOG = HERE, HERE / "plots", HERE / "tabellen", HERE / "grafiken"
for d in (G.PLOTS, G.TABLES, G.BLOG):
    d.mkdir(exist_ok=True)
G.GRID = HET
G.PREREG = dict(FORMULA)
_old_load_folder = G.load_folder


def load_runs():
    rows, runs = [], []
    for p in sorted((HET / "results_tuned").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok" or not r.get("shares"):
            runs.append({"folder": "results_pop", "run_id": r["run_id"], "status": r.get("status")})
            continue
        levels = list(r["levels"])
        tasks, options = int(r["tasks"]), int(r["options"])
        K = sum(l - 1 for l in levels)
        info = {"folder": "results_pop", "run_id": r["run_id"], "design": r["design"], "method": r["method"], "hetero": "realistic",
                "pop": int(r.get("pop", 0)), "n": int(r["n"]), "rep": int(r["rep"]), "hb_seed": int(r.get("hb_seed", 42)),
                "n_burn": int(r.get("n_burn", 10000)), "A": len(levels), "Lmax": max(levels), "mixed": len(set(levels)) > 1,
                "K": K, "T": tasks, "O": options, "ratio": tasks * (options - 1) / K}
        diag = r.get("diagnostics") or {}
        runs.append({**info, "status": "ok", "hb_s": r.get("hb_seconds"), "rhat_max": diag.get("rhat_max"),
                     "ess_min": diag.get("ess_min"), "corr": r.get("indiv_corr")})
        for market, key in (("std", "shares"), ("extra", "shares_extra")):
            for x in r.get(key) or []:
                rows.append({**info, "market": market, "scenario": str(x["scenario"]), "n_products": x["n_products"],
                             "alt": x["alt"], "rule": x["rule"], "est": x["est"], "truth": x["insample"],
                             "pop_share": x["pop"], "is_none": bool(x.get("is_none"))})
    df = pd.DataFrame(rows)
    df["err"] = df["est"] - df["truth"]
    return df, pd.DataFrame(runs)


def load_folder(folder):
    real = FOLDERS.get(folder)
    if real is None or not (HET / real).exists():
        return pd.DataFrame()
    return _old_load_folder(real)


ORIG_LOAD_RUNS = A.load_runs  # the stored simulation of 17.11. (A.GRID is left as it was)
A.load_runs = load_runs
G.load_folder = load_folder

if __name__ == "__main__":
    # figure 6 (number of products) is read by the old figure code; it is not part of the draft
    src = OLD / "blog_vorschlag" / "grafik6_neu_zahl_der_produkte.svg.html"
    if not (G.BLOG / src.name).exists():
        shutil.copy2(src, G.BLOG / src.name)
    G.main()
    par = json.loads((G.TABLES / "formel_gesamtfehler.json").read_text(encoding="utf-8"))["par"]
    same = all(abs(float(par[k]) - FORMULA[k]) < 1e-9 for k in FORMULA)
    print("\nFormel der Auswertung = festgelegte Formel:", same, par)
    assert same, (par, FORMULA)
