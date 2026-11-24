"""Kleine Stichproben (17.11.2026): hält die Formel für den gesamten Fehler auch bei 50, 100 und 250 Personen?

Liest: results_small/ (1.608 Läufe: 67 Designs x 4 Populationen x 50/100/250 Personen x 2 Datensätze; Populationen 0 und 1
am 17.11. mittags, 2 und 3 am Nachmittag) und zum Vergleich die gespeicherten Hauptläufe derselben Populationen bei 150, 400 und 1000.
Die Formel ist die gerundete aus dem Vorschlag und wird hier NICHT neu angepasst.
Schreibt: ergebnisse_kleine_stichproben.txt, tabellen/kleine_stichproben_*.csv
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auswertung as A  # noqa: E402
import auswertung_gesamtfehler as G  # noqa: E402

PAR = dict(b=3.5, be=-0.64, v=5200.0, ve=-0.75, k90=1.63, k95=2.0)
FACTORS = {"share_of_preference": (1.0, 1.0), "rfc": (1.15, 1.14), "first_choice": (1.67, 1.61)}
_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    _lines.append(line)
    print(line)


def boot(units, k, rng, reps=2000):
    vals = []
    for _ in range(reps):
        pick = units.iloc[rng.integers(0, len(units), len(units))]
        vals.append(100 * float(np.mean(np.concatenate(pick.to_list()) <= k)))
    return np.quantile(vals, [0.05, 0.95])


def main():
    small = G.load_folder("results_small")
    df, _ = A.load_runs()
    df = G.prepare(df)
    base = df[(df["method"] == "cbc") & (df["hetero"] == "realistic") & (df["hb_seed"] == 42) & (df["n_burn"] == 10000)
              & df["folder"].isin(["results", "results_more", "results_pop"])]
    cols = ["folder", "run_id", "design", "pop", "n", "rep", "ratio", "x", "market", "scenario", "n_products", "alt", "rule", "est", "truth",
            "pop_share", "is_none", "total"]
    allrows = pd.concat([small[cols], base[cols]], ignore_index=True)
    allrows = allrows[(allrows["market"] == "std") & (~allrows["is_none"])].copy()
    allrows["rank"] = allrows.groupby(["design", "pop", "n", "rep", "scenario", "rule"])["pop_share"].rank(ascending=False, method="first")
    allrows["rbin"] = pd.cut(allrows["ratio"], [0.3, 0.75, 1.5, 3, 7])
    sop = allrows[allrows["rule"] == "share_of_preference"].copy()
    sop["z"] = G.zscore(PAR, sop)
    say(f"Formel (nicht neu angepasst): Spanne = k × Wurzel(b² + v/n), b = {PAR['b']} × r^{PAR['be']}, v = {PAR['v']:.0f} × r^{PAR['ve']}, "
        f"k = {PAR['k90']} (90 %), {PAR['k95']} (95 %)")
    say("Läufe je Stichprobengröße (4 Populationen je Design, Share of preference):",
        {int(n): int(g.groupby(['folder', 'run_id']).ngroups) for n, g in sop.groupby('n')},
        "| Designs:", sop["design"].nunique())

    say("\n=== 1. Abdeckung und Fehler nach Stichprobengröße (50, 100, 250 neu; 150, 400, 1000 gespeichert) ===")
    rng = np.random.default_rng(17)
    rows = []
    for n, g in sop.groupby("n"):
        units = g.groupby(["design", "pop"])["z"].apply(np.array)
        lo90, hi90 = boot(units, PAR["k90"], rng)
        lo95, hi95 = boot(units, PAR["k95"], rng)
        cd = (g["z"] <= PAR["k90"]).groupby(g["design"]).mean()
        page = Z = G.Z95 * np.sqrt((G.PAGE["scale"] * g["ratio"] ** G.PAGE["exp"]) ** 2 + 2500 / g["n"])
        rows.append({"n": int(n), "neu": n in (50, 100, 250), "in_90": 100 * float((g["z"] <= PAR["k90"]).mean()), "lo90": lo90, "hi90": hi90,
                     "in_95": 100 * float((g["z"] <= PAR["k95"]).mean()), "lo95": lo95, "hi95": hi95,
                     "gemessen_durch_formel": G.rel_rms(PAR, g), "rms_fehler": math.sqrt(float((g["total"] ** 2).mean())),
                     "design_min_90": 100 * float(cd.min()), "design_median_90": 100 * float(cd.median()),
                     "designs_unter_80": int((cd < 0.80).sum()),
                     "groesstes_produkt_90": 100 * float((g["z"] <= PAR["k90"])[g["rank"] == 1].mean()),
                     "andere_90": 100 * float((g["z"] <= PAR["k90"])[g["rank"] > 1].mean()),
                     "seite_heute_95": 100 * float((g["total"].abs() <= page).mean())})
    tab = pd.DataFrame(rows)
    say(tab.round(2).to_string(index=False))
    tab.round(3).to_csv(G.TABLES / "kleine_stichproben_nach_n.csv", index=False)

    say("\n=== 2. Gemessener Fehler / Formel nach r und Stichprobengröße ===")
    rel = {(str(rb), int(n)): G.rel_rms(PAR, g) for (rb, n), g in sop.groupby(["rbin", "n"], observed=True)}
    rel = pd.Series(rel).unstack()
    say(rel.round(2).to_string())
    say("-- Abdeckung der 90-%-Spanne nach r und Stichprobengröße")
    cov = (100 * (sop["z"] <= PAR["k90"]).groupby([sop["rbin"].astype(str), sop["n"]]).mean()).unstack()
    say(cov.round(1).to_string())
    rel.round(3).to_csv(G.TABLES / "kleine_stichproben_fehler_nach_r.csv")
    cov.round(2).to_csv(G.TABLES / "kleine_stichproben_abdeckung_nach_r.csv")

    say("\n=== 3. Mittlerer quadrierter Fehler: gemessen gegen Formel (Quadratpunkte, jedes Design gleich gewichtet) ===")
    for n, g in sop.groupby("n"):
        c = g.groupby("design").agg(mse=("total", lambda v: float((v ** 2).mean())), x=("x", "first"))
        model = G.model_ms(PAR, c["x"], n)
        say(f"N {n:>4}: gemessen {c['mse'].mean():7.2f} | Formel {model.mean():7.2f} | Verhältnis {c['mse'].mean() / model.mean():.2f} | "
            f"Designs mit Fehler über 1,3 mal Formel: {int((np.sqrt(c['mse'] / model) > 1.3).sum())} | unter 0,7 mal: "
            f"{int((np.sqrt(c['mse'] / model) < 0.7).sum())}")

    say("\n=== 4. Andere Simulatorregeln mit ihren Faktoren (90-%-Spanne) ===")
    for rule, (fb, fv) in FACTORS.items():
        r_ = allrows[allrows["rule"] == rule]
        zz = G.zscore(PAR, r_, fb, fv)
        say(f"{rule}: innerhalb nach N {G.pct((zz <= PAR['k90']).groupby(r_['n']).mean())} | gemessen/Formel "
            + str({int(n): round(G.rel_rms(PAR, g, fb, fv), 2) for n, g in r_.groupby('n')}))

    say("\n=== 5. Richtiger Gewinner (Share of preference) ===")
    key = ["design", "pop", "n", "rep", "scenario"]
    top = sop.sort_values("pop_share", ascending=False).groupby(key).agg(first=("pop_share", lambda v: v.iloc[0]),
                                                                             second=("pop_share", lambda v: v.iloc[1]))
    est_top = sop.loc[sop.groupby(key)["est"].idxmax()].set_index(key)["rank"]
    top["right"] = est_top.reindex(top.index) == 1
    top["gap"] = pd.cut(100 * (top["first"] - top["second"]), [0, 5, 10, 100], labels=["bis 5", "5 bis 10", "über 10"])
    say("alle Märkte:", G.pct(top.groupby("n")["right"].mean()))
    say((100 * top.reset_index().pivot_table(index="gap", columns="n", values="right", observed=True)).round(1).to_string())

    say("\n=== 6. Auffälligkeiten der Läufe ===")
    import json
    warn = {}
    for p in sorted((G.GRID / "results_small").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        n = int(r["n"])
        w = warn.setdefault(n, {"runs": 0, "hb_warnings": 0, "missing_beta_cols": 0, "indiv_corr": []})
        w["runs"] += 1
        w["hb_warnings"] += bool(r.get("hb_warnings"))
        w["missing_beta_cols"] += bool(r.get("missing_beta_cols"))
        if isinstance(r.get("indiv_corr"), (int, float)):
            w["indiv_corr"].append(r["indiv_corr"])
    for n in sorted(warn):
        w = warn[n]
        say(f"N {n}: {w['runs']} Läufe, mit HB-Warnung {w['hb_warnings']}, mit fehlenden Beta-Spalten {w['missing_beta_cols']}, "
            f"Korrelation der individuellen Nutzen (Median) {np.median(w['indiv_corr']) if w['indiv_corr'] else float('nan'):.3f}")

    say("\n=== 7. Was die Formel für das Beispieldesign bei kleinen Stichproben sagt (r = 1,43) ===")
    for n in (50, 100, 150, 250, 300, 400):
        say(f"N {n}: ±{G.margin(PAR, G.EXAMPLE_R, n, PAR['k90']):.1f} (90 %), ±{G.margin(PAR, G.EXAMPLE_R, n, PAR['k95']):.1f} (95 %)")
    (G.OUT / "ergebnisse_kleine_stichproben.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
