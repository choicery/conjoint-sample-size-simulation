"""Night runs 18.11.2026 against the sample-size formula of the blog draft (fixed on 17.11., not refitted here).

  margin90 = 1.63 * sqrt(b^2 + v / n),  b = 3.5 * r^-0.64,  v = 5200 * r^-0.75,  r = tasks * (options - 1) / utilities
Error = simulated share minus true share of the population, percentage points, share of preference,
markets with 3 or 4 products (the six standard markets of every run), exactly as in auswertung_gesamtfehler.py.

Usage: python auswertung_sim.py results_emp [results_tuned ...]
Output: ergebnisse_<folder>.txt, tabellen/<folder>_*.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tabellen"
K90, K95, B_SCALE, B_EXP, V_SCALE, V_EXP = 1.63, 2.0, 3.5, -0.64, 5200.0, -0.75


def load(folder: str, base: Path = HERE) -> pd.DataFrame:
    rows = []
    for p in sorted((base / folder).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok" or not r.get("shares"):
            continue
        k = sum(l - 1 for l in r["levels"])
        info = {"run_id": r["run_id"], "design": r["design"], "study": r.get("family") or r.get("study") or r["design"],
                "which": r.get("which", ""), "pop": int(r.get("pop", 0)), "n": int(r["n"]), "rep": int(r["rep"]),
                "ratio": int(r["tasks"]) * (int(r["options"]) - 1) / k, "k": k, "attributes": len(r["levels"]),
                "indiv_corr": r.get("indiv_corr")}
        for market, key in (("std", "shares"), ("extra", "shares_extra")):
            for x in r.get(key) or []:
                if x.get("is_none"):
                    continue
                rows.append({**info, "market": market, "scenario": str(x["scenario"]), "n_products": x["n_products"],
                             "alt": x["alt"], "rule": x["rule"], "est": x["est"], "pop_share": x["pop"], "insample": x["insample"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["err"] = (df["est"] - df["pop_share"]) * 100
    df["b"] = B_SCALE * df["ratio"] ** B_EXP
    df["v"] = V_SCALE * df["ratio"] ** V_EXP
    df["pred2"] = df["b"] ** 2 + df["v"] / df["n"]
    df["z"] = df["err"].abs() / np.sqrt(df["pred2"])
    return df


def line(name, q) -> str:
    return (f"{name:38s} Anteile {len(q):6d}  Fehler/Formel {np.sqrt((q['err'] ** 2).sum() / q['pred2'].sum()):.2f}  "
            f"90-%-Spanne deckt {100 * (q['z'] <= K90).mean():5.1f} %  95-%-Spanne deckt {100 * (q['z'] <= K95).mean():5.1f} %  "
            f"mittl. |Fehler| {q['err'].abs().mean():.2f} pp")


def winners(q) -> float:
    """Share of markets in which the product with the largest true share also has the largest simulated share."""
    g = q.groupby(["run_id", "scenario"])
    ok = g.apply(lambda m: m.loc[m["pop_share"].idxmax(), "alt"] == m.loc[m["est"].idxmax(), "alt"], include_groups=False)
    return float(ok.mean())


def split_b_v(q) -> tuple[float, float]:
    """Least squares on design-by-n cells: mse = fb * b^2 + fv * v / n. Returns (sqrt(fb), fv)."""
    c = q.groupby(["design", "n"]).agg(mse=("err", lambda e: float((e ** 2).mean())), b=("b", "first"), v=("v", "first")).reset_index()
    x = np.column_stack([c["b"] ** 2, c["v"] / c["n"]])
    w = 1 / (c["b"] ** 2 + c["v"] / c["n"])  # relative errors count
    coef, *_ = np.linalg.lstsq(x * w.to_numpy()[:, None], c["mse"].to_numpy() * w.to_numpy(), rcond=None)
    return float(np.sqrt(max(coef[0], 0))), float(coef[1])


def report(folder: str) -> None:
    df = load(folder)
    out = []
    say = out.append
    if df.empty:
        print("keine Läufe in", folder)
        return
    sop = df[(df["market"] == "std") & (df["rule"] == "share_of_preference")]
    runs = sop["run_id"].nunique()
    say(f"{folder}: {runs} Läufe, {sop['design'].nunique()} Designs, {sop['study'].nunique()} Studien/Gruppen")
    say("Formel unverändert vom 17.11. (b = 3,5 r^-0,64; v = 5.200 r^-0,75; k = 1,63 / 2,0)")
    say("")
    say(line("alle", sop))
    for n, q in sop.groupby("n"):
        say(line(f"  {n} Personen", q))
    if sop["which"].nunique() > 1:
        for w, q in sop.groupby("which"):
            say(line(f"  Design: {w}", q))
    for lab, q in (("  r unter 1", sop[sop["ratio"] < 1]), ("  r 1 bis 2,5", sop[(sop["ratio"] >= 1) & (sop["ratio"] < 2.5)]),
                   ("  r ab 2,5", sop[sop["ratio"] >= 2.5])):
        if len(q):
            say(line(lab, q))
    fb, fv = split_b_v(sop)
    say("")
    say(f"Aufteilung (kleinste Quadrate über Design-mal-N-Zellen): bleibender Teil b = {fb:.2f} mal Formel, "
        f"schrumpfender Teil v = {fv:.2f} mal Formel")
    say(f"Richtiger Gewinner (wahres größtes Produkt hat größten simulierten Anteil): "
        + ", ".join(f"{n}: {100 * winners(q):.1f} %" for n, q in sop.groupby("n")))
    lead = sop[sop["pop_share"] == sop.groupby(["run_id", "scenario"])["pop_share"].transform("max")]
    say(f"Größtes Produkt im Markt: im Mittel {lead['err'].mean():+.2f} pp, 90-%-Spanne deckt {100 * (lead['z'] <= K90).mean():.1f} %; "
        f"übrige Produkte {100 * (sop.drop(lead.index)['z'] <= K90).mean():.1f} %")
    say("")
    say("Je Studie bzw. Design-Gruppe")
    rows = []
    for (study, design), q in sop.groupby(["study", "design"]):
        rows.append({"study": study, "design": design, "which": q["which"].iloc[0], "attributes": q["attributes"].iloc[0],
                     "k": q["k"].iloc[0], "ratio": q["ratio"].iloc[0], "runs": q["run_id"].nunique(), "shares": len(q),
                     "err_over_formula": np.sqrt((q["err"] ** 2).sum() / q["pred2"].sum()),
                     "cover90": 100 * (q["z"] <= K90).mean(), "cover95": 100 * (q["z"] <= K95).mean(),
                     "mean_abs_err": q["err"].abs().mean(), "indiv_corr": q.drop_duplicates("run_id")["indiv_corr"].mean()})
    t = pd.DataFrame(rows)
    TABLES.mkdir(exist_ok=True)
    t.round(3).to_csv(TABLES / f"{folder}_je_design.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    if len(t) <= 80:
        say(t.round(2).to_string(index=False))
    say("")
    say(f"Fehler/Formel über die Designs: Median {t['err_over_formula'].median():.2f}, Quartile {t['err_over_formula'].quantile(0.25):.2f} "
        f"bis {t['err_over_formula'].quantile(0.75):.2f}, kleinster {t['err_over_formula'].min():.2f}, größter {t['err_over_formula'].max():.2f}")
    say(f"Designs mit Abdeckung der 90-%-Spanne unter 85 %: {(t['cover90'] < 85).sum()} von {len(t)}; über 95 %: {(t['cover90'] > 95).sum()}")
    # error by true share (figure 5 of the draft)
    centres = list(range(5, 75, 5))
    b = pd.cut(sop["pop_share"] * 100, [c - 2.5 for c in centres] + [72.5], labels=centres)
    per = sop.assign(bin=b).groupby(["bin", "design"], observed=True)["err"].mean().groupby(level=0, observed=True).mean()
    say("")
    say("Mittlerer Fehler nach wahrem Anteil (pp; jedes Design zählt einmal): "
        + ", ".join(f"{int(c)} %: {v:+.1f}" for c, v in per.items()))
    for rule in ("rfc", "first_choice"):
        q = df[(df["market"] == "std") & (df["rule"] == rule)]
        if len(q):
            say(f"{rule}: Fehler gegen Share of preference = {np.sqrt((q['err'] ** 2).mean() / (sop['err'] ** 2).mean()):.2f} mal")
    (HERE / f"ergebnisse_{folder}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)
        print()
