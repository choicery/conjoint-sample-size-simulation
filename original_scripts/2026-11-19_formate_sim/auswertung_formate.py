"""Question formats against plain CBC (runs of run_formate.py, 19.11.2026).

Formula of the page (FORMEL_18_11.md), fitted to plain CBC:  margin = k * sqrt(b^2 + v / n),
    b = 4.3 * (f r)^-0.57,  v = 5000 * (f r)^-0.81,  k = 1.65 (90 %) / 2.0 (95 %),  r = tasks * (options - 1) / utilities.
f is the task factor of the format: one task of the format counts like f CBC tasks. The calculator uses 1.6 (BW-CBC) and 1.3 (DR-CBC).

Error = simulated share minus true share in the population, percentage points, share of preference.
Markets without None (the six standard markets and the 12 extra markets of every run): identical for every format and for the
plain CBC run of the same cell (same population, same respondents, same task design), so the formats compare directly.
Markets with None (None formats only): the six standard markets with the None alternative.

Usage: python auswertung_formate.py            -> ergebnisse_formate.txt, tabellen/*.csv, formate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HET = HERE.parent / "2026-11-18_heterogenitaet_echte_studien"
B, BE, V, VE, K90, K95 = 4.3, -0.57, 5000.0, -0.81, 1.65, 2.0
CALC = {"cbc": 1.0, "drcbc": 1.3, "bwcbc": 1.6, "bwcbc_none": 1.6}
NAMES = {"cbc": "CBC", "drcbc": "DR-CBC", "bwcbc": "BW-CBC", "bwcbc_none": "BW-CBC mit None"}
FORMATS = ("drcbc", "bwcbc", "bwcbc_none")
RNG = np.random.default_rng(20261119)


def rows_of(r, method):
    k = sum(l - 1 for l in r["levels"])
    info = {"run_id": r["run_id"], "cell": f"{r['design']}_p{r['pop']}_n{r['n']}_r{r['rep']}", "design": r["design"], "method": method,
            "pop": int(r["pop"]), "n": int(r["n"]), "rep": int(r["rep"]), "ratio": int(r["tasks"]) * (int(r["options"]) - 1) / k, "k": k,
            "options": int(r["options"]), "tasks": int(r["tasks"])}
    with_none = method in ("drcbc", "bwcbc_none")
    sets = [("std", r.get("shares_products_only") if with_none else r.get("shares")), ("extra", r.get("shares_extra"))]
    if with_none:
        sets.append(("none", r.get("shares")))
    out = []
    for market, rows in sets:
        for x in rows or []:
            out.append({**info, "market": market, "scenario": str(x["scenario"]), "alt": x["alt"], "is_none": bool(x.get("is_none")), "rule": x["rule"],
                        "est": x["est"], "truth": x["pop"], "err": 100 * (x["est"] - x["pop"])})
    return out


def load():
    rows, runs = [], []
    for p in sorted((HERE / "results_formate").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs.append({"run_id": r["run_id"], "cell": f"{r['design']}_p{r['pop']}_n{r['n']}_r{r['rep']}", "design": r["design"], "method": r["method"],
                     "pop": r["pop"], "n": r["n"], "rep": r["rep"], "status": r.get("status"), "hb_s": r.get("hb_seconds"), "total_s": r.get("total_seconds"),
                     "corr": r.get("indiv_corr"), "nobuy": r.get("observed_no_buy_share"), "rhat_max": (r.get("diagnostics") or {}).get("rhat_max"),
                     "hb_warnings": len(r.get("hb_warnings") or []), "error": r.get("error")})
        if r.get("status") == "ok":
            rows += rows_of(r, r["method"])
    runs = pd.DataFrame(runs)
    cells = sorted(set(runs.loc[runs["status"] == "ok", "cell"]))
    for cell in cells:   # the plain CBC run of the same cell
        d, p, n, rep = cell.rsplit("_", 3)
        f = HET / "results_tuned" / f"{d}_cbc_tuned_{p}_{n}_{rep}.json"
        r = json.loads(f.read_text(encoding="utf-8"))
        assert r.get("status") == "ok", f
        rows += rows_of(r, "cbc")
        runs = pd.concat([runs, pd.DataFrame([{"run_id": r["run_id"], "cell": cell, "design": r["design"], "method": "cbc", "pop": r["pop"], "n": r["n"],
                                               "rep": r["rep"], "status": "ok", "hb_s": r.get("hb_seconds"), "total_s": r.get("total_seconds"),
                                               "corr": r.get("indiv_corr"), "rhat_max": (r.get("diagnostics") or {}).get("rhat_max")}])],
                         ignore_index=True)
    return pd.DataFrame(rows), runs


def pred2(ratio, n, f=1.0):
    return (B * (f * ratio) ** BE) ** 2 + V * (f * ratio) ** VE / n


def cells_of(q):
    return q.groupby(["design", "n"]).agg(mse=("err", lambda e: float((e ** 2).mean())), ratio=("ratio", "first")).reset_index()


def fit_factor(c):
    """Task factor f that makes the formula fit the design-by-n cells best (relative errors, grid on log f)."""
    grid = np.exp(np.linspace(np.log(0.4), np.log(6.0), 541))
    r, n, mse = c["ratio"].to_numpy(), c["n"].to_numpy(), c["mse"].to_numpy()
    loss = [float(((np.log(mse) - np.log(pred2(r, n, f))) ** 2).sum()) for f in grid]
    return float(grid[int(np.argmin(loss))])


def boot_factor(q, ref=None, reps=400):
    """Factor with a 90 % interval from resampling designs; with ref: factor relative to the CBC runs of the same cells."""
    c = cells_of(q)
    cr = cells_of(ref) if ref is not None else None
    point = fit_factor(c) / (fit_factor(cr) if cr is not None else 1.0)
    designs = sorted(c["design"].unique())
    vals = []
    for _ in range(reps):
        pick = RNG.choice(designs, size=len(designs), replace=True)
        cc = pd.concat([c[c["design"] == d] for d in pick])
        f = fit_factor(cc)
        if cr is not None:
            f /= fit_factor(pd.concat([cr[cr["design"] == d] for d in pick]))
        vals.append(f)
    return point, float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def cover(q, f, k=K90):
    z = q["err"].abs() / np.sqrt(pred2(q["ratio"], q["n"], f))
    return 100 * float((z <= k).mean())


def main():
    df, runs = load()
    out = []
    say = out.append
    res = {}
    done = runs[runs["method"] != "cbc"]
    say(f"Formatläufe: {len(done)} Dateien, davon ok {int((done['status'] == 'ok').sum())}; je Format: "
        + ", ".join(f"{NAMES[m]} {int(((done['method'] == m) & (done['status'] == 'ok')).sum())}" for m in FORMATS))
    bad = done[done["status"] != "ok"]
    if len(bad):
        say("NICHT ok: " + "; ".join(f"{r.run_id} ({r.status}: {str(r.error)[:80]})" for r in bad.itertuples()))
    say(f"HB-Warnungen in {int((done['hb_warnings'] > 0).sum())} Läufen; größtes R-hat je Format (Median / Maximum): "
        + ", ".join(f"{NAMES[m]} {done.loc[done['method'] == m, 'rhat_max'].median():.2f} / {done.loc[done['method'] == m, 'rhat_max'].max():.2f}" for m in FORMATS)
        + f"; CBC derselben Zellen {runs.loc[runs['method'] == 'cbc', 'rhat_max'].median():.2f} / {runs.loc[runs['method'] == 'cbc', 'rhat_max'].max():.2f}")
    say("Anteil Nicht-Kauf (beobachtet): " + ", ".join(f"{NAMES[m]} {100 * done.loc[done['method'] == m, 'nobuy'].mean():.0f} %" for m in ("drcbc", "bwcbc_none")))
    say("")
    sop = df[df["rule"] == "share_of_preference"]
    po = sop[sop["market"].isin(["std", "extra"])]
    say("1. FAKTOR JE AUFGABE aus den Anteilen (Märkte ohne None, Share of Preference)")
    say("   Faktor f: eine Aufgabe des Formats zählt wie f CBC-Aufgaben. 'gegen CBC' = geteilt durch denselben Fit für die CBC-Läufe derselben Zellen.")
    for m in FORMATS:
        q = po[po["method"] == m]
        if q.empty:
            continue
        ref = po[(po["method"] == "cbc") & po["cell"].isin(q["cell"].unique())]
        f_abs, lo_a, hi_a = boot_factor(q)
        f_rel, lo, hi = boot_factor(q, ref)
        f_ref = fit_factor(cells_of(ref))
        res[m] = {"runs": int(q["run_id"].nunique()), "designs": int(q["design"].nunique()), "f_abs": f_abs, "f_abs_lo": lo_a, "f_abs_hi": hi_a, "f_rel": f_rel,
                  "f_rel_lo": lo, "f_rel_hi": hi, "f_cbc_same_cells": f_ref}
        say(f"   {NAMES[m]:16s} Läufe {q['run_id'].nunique():4d}, Designs {q['design'].nunique():2d}: f = {f_abs:.2f} (90 %: {lo_a:.2f} bis {hi_a:.2f}); "
            f"CBC derselben Zellen {f_ref:.2f}; gegen CBC f = {f_rel:.2f} (90 %: {lo:.2f} bis {hi:.2f}) | Rechner bisher {CALC[m]}")
        for lab, mask in (("nur Standardmärkte", q["market"] == "std"), ("r unter 1", q["ratio"] < 1), ("r 1 bis 2,5", (q["ratio"] >= 1) & (q["ratio"] < 2.5)),
                          ("r ab 2,5", q["ratio"] >= 2.5), ("3 Optionen", q["options"] == 3), ("4 Optionen", q["options"] == 4), ("2 Optionen", q["options"] == 2),
                          ("150 Personen", q["n"] == 150), ("400 Personen", q["n"] == 400), ("1.000 Personen", q["n"] == 1000)):
            qq = q[mask]
            if qq.empty:
                continue
            rr = ref[ref["cell"].isin(qq["cell"].unique()) & (ref["market"].isin(qq["market"].unique()))]
            res[m][f"f_rel {lab}"] = fit_factor(cells_of(qq)) / fit_factor(cells_of(rr))
            say(f"      {lab:20s} gegen CBC f = {res[m][f'f_rel {lab}']:.2f}  (Designs {qq['design'].nunique()})")
    say("")
    say("2. FEHLER IM DIREKTEN VERGLEICH (gleiche Zellen, Märkte ohne None): Wurzel aus mittlerem Fehlerquadrat, Prozentpunkte")
    t = po.groupby(["method", "n"])["err"].apply(lambda e: float(np.sqrt((e ** 2).mean()))).unstack("n")
    cells_all = set.intersection(*[set(po.loc[po["method"] == m, "cell"]) for m in po["method"].unique()])
    tb = po[po["cell"].isin(cells_all)].groupby(["method", "n"])["err"].apply(lambda e: float(np.sqrt((e ** 2).mean()))).unstack("n")
    say(f"   Zellen mit allen Formaten: {len(cells_all)}")
    say("   " + tb.rename(index=NAMES).round(2).to_string().replace("\n", "\n   "))
    say("")
    say("3. HÄLT DIE SPANNE DES RECHNERS? Anteil der Anteile innerhalb der Spanne (Märkte ohne None), mit bisherigem und mit neuem Faktor")
    for m in FORMATS:
        q = po[po["method"] == m]
        if q.empty:
            continue
        f_new = res[m]["f_abs"]
        res[m]["cover90_old"], res[m]["cover95_old"] = cover(q, CALC[m]), cover(q, CALC[m], K95)
        res[m]["cover90_new"], res[m]["cover95_new"] = cover(q, f_new), cover(q, f_new, K95)
        say(f"   {NAMES[m]:16s} bisher (f = {CALC[m]}): 90 % deckt {res[m]['cover90_old']:.1f} %, 95 % deckt {res[m]['cover95_old']:.1f} % | "
            f"neu (f = {f_new:.2f}): {res[m]['cover90_new']:.1f} % / {res[m]['cover95_new']:.1f} % | je N mit neuem f: "
            + ", ".join(f"{n}: {cover(g, f_new):.1f} %" for n, g in q.groupby("n")))
    q = po[po["method"] == "cbc"]
    say(f"   {'CBC (Referenz)':16s} f = 1: 90 % deckt {cover(q, 1.0):.1f} %, 95 % deckt {cover(q, 1.0, K95):.1f} %")
    say("")
    say("4. MÄRKTE MIT NONE (nur Formate mit None): Spanne mit neuem Faktor")
    nm = sop[sop["market"] == "none"]
    for m in ("drcbc", "bwcbc_none"):
        q = nm[nm["method"] == m]
        if q.empty:
            continue
        f_new = res[m]["f_abs"]
        prod, none = q[~q["is_none"]], q[q["is_none"]]
        res[m]["none_market_products90"], res[m]["none_market_none90"] = cover(prod, f_new), cover(none, f_new)
        res[m]["none_bias"] = float(none["err"].mean())
        say(f"   {NAMES[m]:16s} Produkte: 90 % deckt {cover(prod, f_new):.1f} %, mittl. |Fehler| {prod['err'].abs().mean():.2f} pp | None-Anteil: 90 % deckt "
            f"{cover(none, f_new):.1f} %, mittlerer Fehler {none['err'].mean():+.2f} pp, mittl. |Fehler| {none['err'].abs().mean():.2f} pp")
    say("")
    say("5. ANDERE SIMULATOR-REGELN gegen Share of Preference (Märkte ohne None, Standardmärkte): Fehler-Verhältnis")
    std = df[df["market"] == "std"]
    for m in ("cbc",) + FORMATS:
        q = std[std["method"] == m]
        if q.empty:
            continue
        base = float((q.loc[q["rule"] == "share_of_preference", "err"] ** 2).mean())
        say(f"   {NAMES[m]:16s} " + ", ".join(f"{rule}: {np.sqrt(float((q.loc[q['rule'] == rule, 'err'] ** 2).mean()) / base):.2f}" for rule in ("rfc", "first_choice")))
    say("")
    say("6. INDIVIDUELLE NUTZENWERTE: Korrelation geschätzt gegen wahr (Mittel über Läufe), und Faktor aus der Rückgewinnung")
    ok = runs[runs["status"] == "ok"].copy()
    ok["z"] = np.arctanh(ok["corr"].clip(-0.999999, 0.999999))
    ratio = df.drop_duplicates("design").set_index("design")["ratio"]
    ok["ratio"] = ok["design"].map(ratio)
    cb = ok[ok["method"] == "cbc"]
    slope = np.polyfit(np.log(cb["ratio"]), cb["z"], 1)[0]
    say(f"   CBC: atanh(Korrelation) steigt um {slope:.3f} je log(r)")
    for m in ("cbc",) + FORMATS:
        q = ok[ok["method"] == m]
        if q.empty:
            continue
        line = f"   {NAMES[m]:16s} Korrelation {np.tanh(q['z'].mean()):.3f}"
        if m != "cbc":
            pair = q.merge(cb[["cell", "z"]], on="cell", suffixes=("", "_cbc"))
            f_rec = float(np.exp((pair["z"] - pair["z_cbc"]).mean() / slope))
            res[m]["f_recovery"] = f_rec
            line += f" | Faktor aus Rückgewinnung {f_rec:.2f}"
        say(line)
    say("")
    say("7. LAUFZEIT: Median HB-Sekunden je Format und N")
    say("   " + ok.groupby(["method", "n"])["hb_s"].median().unstack("n").rename(index=NAMES).round(0).to_string().replace("\n", "\n   "))
    (HERE / "tabellen").mkdir(exist_ok=True)
    per = []
    for (m, d), q in po.groupby(["method", "design"]):
        per.append({"method": m, "design": d, "ratio": q["ratio"].iloc[0], "options": q["options"].iloc[0], "runs": q["run_id"].nunique(),
                    "rmse": float(np.sqrt((q["err"] ** 2).mean())), "cover90_f1": cover(q, 1.0), "cover90_calc": cover(q, CALC[m])})
    pd.DataFrame(per).round(3).to_csv(HERE / "tabellen" / "formate_je_design.csv", index=False)
    runs.to_csv(HERE / "tabellen" / "formate_laeufe.csv", index=False)
    (HERE / "formate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (HERE / "ergebnisse_formate.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
