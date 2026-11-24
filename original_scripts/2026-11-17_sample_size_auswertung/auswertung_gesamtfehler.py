"""Formel für den gesamten Fehler eines simulierten Anteils (nach dem Codex-Review vom 17.11.2026).

Liest nur: die Ergebnisordner unter _scratch/2026-11-09_sample_size_sim/grid/ (über auswertung.load_runs) und,
falls vorhanden, results_check/ und results_seedlong/ (Prüfläufe vom 17.11.).
Schreibt: ergebnisse_gesamtfehler.txt, tabellen/gesamtfehler_*.csv, tabellen/formel_gesamtfehler.json,
plots/14...19*.png, blog_vorschlag/*.svg.html und die Vorschau.

Zielgröße: Fehler = simulierter Anteil minus wahrer Anteil der Population (nicht: derselben Personen).
Er enthält alles zusammen: wer teilnimmt, Zufall in den Antworten, Schätzung, Rauschen der Ketten.

Formel:  Spanne = k * Wurzel(b^2 + v / n)      (Prozentpunkte)
   b = B_SCALE * r^B_EXP     der Teil, der bei mehr Personen bleibt
   v = V_SCALE * r^V_EXP     der Teil, der mit mehr Personen schrumpft (Quadratpunkte je Person)
   k = Faktor für das Niveau (90 % oder 95 %), gemessen als Quantil von |Fehler| / Wurzel(b^2 + v/n)
   r = Aufgaben * (Optionen - 1) * Aufgabenfaktor / Summe(Stufen - 1)
Anpassung: Quasi-Likelihood mit Log-Link auf den mittleren quadrierten Fehler je Design und Stichprobengröße.
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auswertung as A  # noqa: E402
import blog_grafiken_sechs as S6  # noqa: E402

plt = A.plt
OUT, GRID, PLOTS, TABLES, BLOG = A.OUT, A.GRID, A.PLOTS, A.TABLES, A.BLOG
Z90, Z95 = 1.6448536269514722, 1.959963984540054
PAGE = dict(scale=3.7, exp=-0.70, rfc=1.15, fc=1.62)  # formula on the page today
PREREG = dict(b=3.517, be=-0.635, v=5157.0, ve=-0.748, k90=1.627, k95=1.996)  # fixed before the check runs
EXAMPLE_R = 10 * 2 / 14
RULE_NAMES = {"share_of_preference": "Share of preference", "rfc": "Randomized first choice", "first_choice": "First choice"}
_report = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    _report.append(line)
    print(line)


# ---------------------------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------------------------
def prepare(df):
    df = df.copy()
    df["total"] = (df["est"] - df["pop_share"]) * 100
    df["x"] = np.log(df["ratio"])
    return df


def load_folder(folder):
    """Same row layout as auswertung.load_runs, for the check folders written on 17.11."""
    rows = []
    for p in sorted((GRID / folder).glob("*.json")) if (GRID / folder).exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok" or not r.get("shares"):
            continue
        levels = list(r["levels"])
        K = sum(l - 1 for l in levels)
        info = {"folder": folder, "run_id": r["run_id"], "design": r["design"], "method": r["method"], "pop": int(r.get("pop", 0)),
                "n": int(r["n"]), "rep": int(r["rep"]), "hb_seed": int(r.get("hb_seed", 42)), "n_burn": int(r.get("n_burn", 10000)),
                "ratio": int(r["tasks"]) * (int(r["options"]) - 1) / K, "hb_s": r.get("hb_seconds")}
        for market, key in (("std", "shares"), ("extra", "shares_extra")):
            for x in r.get(key) or []:
                rows.append({**info, "market": market, "scenario": str(x["scenario"]), "n_products": x["n_products"], "alt": x["alt"],
                             "rule": x["rule"], "est": x["est"], "truth": x["insample"], "pop_share": x["pop"],
                             "is_none": bool(x.get("is_none"))})
    return prepare(pd.DataFrame(rows)) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------------------------
# Formel
# ---------------------------------------------------------------------------------------------
def cells(rows):
    return rows.groupby(["design", "n"]).agg(mse=("total", lambda v: float((v ** 2).mean())), x=("x", "first")).reset_index()


def fit_total(d):
    def pred(q):
        return np.exp(q[0] + q[1] * d["x"]) + np.exp(q[2] + q[3] * d["x"]) / d["n"]

    obj = lambda q: float(np.sum(d["mse"] / pred(q) + np.log(pred(q))))
    opts = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000}
    best = min((minimize(obj, x0=s, method="Nelder-Mead", options=opts) for s in ([2.5, -1.24, 8.0, -1.0], [2.0, -1.0, 8.5, -0.5])),
               key=lambda r: r.fun)
    q = best.x
    return dict(b=math.exp(q[0] / 2), be=q[1] / 2, v=math.exp(q[2]), ve=q[3])


def fit_factors(d, par):
    """Constant factors on b and v for another simulator method, exponents as for share of preference."""
    def obj(q):
        pred = (math.exp(q[0]) * par["b"] * np.exp(par["be"] * d["x"])) ** 2 + math.exp(q[1]) * par["v"] * np.exp(par["ve"] * d["x"]) / d["n"]
        return float(np.sum(d["mse"] / pred + np.log(pred)))

    q = minimize(obj, x0=[0.2, 0.2], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11}).x
    return math.exp(q[0]), math.exp(q[1])


def model_ms(par, x, n, fb=1.0, fv=1.0):
    return (fb * par["b"] * np.exp(par["be"] * x)) ** 2 + fv * par["v"] * np.exp(par["ve"] * x) / n


def zscore(par, rows, fb=1.0, fv=1.0, task_factor=1.0):
    return rows["total"].abs() / np.sqrt(model_ms(par, rows["x"] + math.log(task_factor), rows["n"], fb, fv))


def rel_rms(par, rows, fb=1.0, fv=1.0, task_factor=1.0):
    return math.sqrt(float((rows["total"] ** 2).mean() / model_ms(par, rows["x"] + math.log(task_factor), rows["n"], fb, fv).mean()))


def b_of(par, r, fb=1.0):
    return fb * par["b"] * r ** par["be"]


def v_of(par, r, fv=1.0):
    return fv * par["v"] * r ** par["ve"]


def margin(par, r, n, k, fb=1.0, fv=1.0):
    return k * math.sqrt(b_of(par, r, fb) ** 2 + (v_of(par, r, fv) / n if n else 0.0))


def needed(par, r, m, k, fb=1.0, fv=1.0):
    room = (m / k) ** 2 - b_of(par, r, fb) ** 2
    return None if room <= 0 else math.ceil(v_of(par, r, fv) / room - 1e-9)


def needed_page(r, m, factor=1.0):
    b = PAGE["scale"] * r ** PAGE["exp"] * factor
    room = (m / Z95) ** 2 - b * b
    return None if room <= 0 else math.ceil(2500 / room - 1e-9)


def fmt_n(v, floor=50):
    if v is None:
        return "nicht erreichbar"
    return f"{max(v, floor):,}".replace(",", ".") + (f" (rechnerisch {v})" if v < floor else "")


def pct(series):
    return {int(k): round(float(v), 1) for k, v in (100 * series).items()}


# ---------------------------------------------------------------------------------------------
def main():
    df, runs = A.load_runs()
    df = prepare(df)
    main_rows = df[(df["method"] == "cbc") & (df["hetero"] == "realistic") & (df["hb_seed"] == 42) & (df["n_burn"] == 10000)
                   & df["folder"].isin(["results", "results_more", "results_pop"]) & (~df["is_none"])]
    std = main_rows[main_rows["market"] == "std"]
    sop = std[std["rule"] == "share_of_preference"].copy()
    n_runs = sop.groupby(["folder", "run_id"]).ngroups
    n_designs = sop["design"].nunique()
    say(f"Hauptläufe: {n_runs} Läufe, {n_designs} Designs, {len(sop)} simulierte Anteile (Share of preference, Märkte mit 3 oder 4 Produkten)")
    sop["rank"] = sop.groupby(["design", "pop", "n", "rep", "scenario"])["pop_share"].rank(ascending=False, method="first")
    sop["rbin"] = pd.cut(sop["ratio"], [0.3, 0.75, 1.5, 3, 7])

    # ---- 1. heutige Formel -----------------------------------------------------------------
    say("\n=== 1. Was die Formel der Seite wirklich abdeckt (Fehler gegen den wahren Anteil der Population) ===")
    cov_rows = []
    for name, (a, e) in (("Seite heute 3.7 / -0.70", (PAGE["scale"], PAGE["exp"])), ("Vorschlag vom 17.11. früh 3.6 / -0.62", (3.6, -0.62))):
        for zname, z in (("1.96", Z95), ("1.645", Z90)):
            m = z * np.sqrt((a * sop["ratio"] ** e) ** 2 + 2500 / sop["n"])
            c = sop["total"].abs() <= m
            by_n = pct(c.groupby(sop["n"]).mean())
            cd = c.groupby(sop["design"]).mean()
            say(f"{name}, Faktor {zname}: nach N {by_n} | schlechtestes Design {100 * cd.min():.1f} | größtes Produkt "
                f"{100 * c[sop['rank'] == 1].mean():.1f}, andere {100 * c[sop['rank'] > 1].mean():.1f}")
            cov_rows.append({"formel": name, "faktor": zname, **{f"N{k}": v for k, v in by_n.items()}})
    page_cov = pct((sop["total"].abs() <= Z95 * np.sqrt((PAGE["scale"] * sop["ratio"] ** PAGE["exp"]) ** 2 + 2500 / sop["n"])).groupby(sop["n"]).mean())

    # ---- 2. Zerlegung ----------------------------------------------------------------------
    say("\n=== 2. Woraus der Fehler besteht (Quadratpunkte, jedes Design gleich gewichtet) ===")
    sop["samp"] = (sop["truth"] - sop["pop_share"]) * 100
    sop["e_in"] = (sop["est"] - sop["truth"]) * 100
    decomp = []
    for n, g in sop.groupby("n"):
        prod = A.b2_per_product(g)
        fixed = 1e4 * float(prod.groupby("design")["b2"].mean().mean())
        tot = float(g.groupby("design")["total"].apply(lambda v: (v ** 2).mean()).mean())
        samp = float(g.groupby("design")["samp"].apply(lambda v: (v ** 2).mean()).mean())
        shrink = tot - fixed
        decomp.append({"n": int(n), "gesamt": tot, "bleibt": fixed, "schrumpft": shrink, "davon_stichprobe": samp,
                       "davon_schaetzung": shrink - samp, "lehrbuch_0.25/n": 2500 / n, "schrumpft_mal_n": shrink * n})
        say(f"N {n:>4}: gesamt {tot:6.2f} = bleibt {fixed:5.2f} + schrumpft {shrink:5.2f} (davon wer teilnimmt {samp:.2f} = "
            f"{100 * samp / shrink:.0f} %, Rauschen der Schätzung {shrink - samp:.2f}) | Lehrbuch 0,25/n: {2500 / n:.2f} | "
            f"schrumpfender Teil × n: {shrink * n:.0f}")
    decomp = pd.DataFrame(decomp)
    decomp.round(3).to_csv(TABLES / "gesamtfehler_zerlegung.csv", index=False)

    # ---- 3. Formel für den gesamten Fehler ---------------------------------------------------
    say("\n=== 3. Formel für den gesamten Fehler ===")
    exact = fit_total(cells(sop))
    z = zscore(exact, sop)
    exact["k90"], exact["k95"] = float(z.quantile(0.90)), float(z.quantile(0.95))
    par = dict(b=round(exact["b"], 1), be=round(exact["be"], 2), v=float(round(exact["v"], -2)), ve=round(exact["ve"], 2))
    zr = zscore(par, sop)
    par["k90"], par["k95"] = round(float(zr.quantile(0.90)), 2), round(float(zr.quantile(0.95)), 2)
    say(f"genau:    b = {exact['b']:.3f} × r^{exact['be']:.3f} | v = {exact['v']:.0f} × r^{exact['ve']:.3f} | k90 {exact['k90']:.3f} | k95 {exact['k95']:.3f}")
    say(f"gerundet: b = {par['b']:.1f} × r^{par['be']:.2f} | v = {par['v']:.0f} × r^{par['ve']:.2f} | k90 {par['k90']:.2f} | k95 {par['k95']:.2f}"
        f"   (Normalverteilung: 1.645 und 1.960)")
    say(f"vor den Prüfläufen festgelegt (run_check.py): {PREREG}")
    t = A.design_table(A.b2_per_product(sop))
    pa, pe, _, _ = A.fit_mean(t)
    say(f"zum Vergleich: fester Teil aus Paaren unabhängiger Läufe (auswertung.py): {pa:.2f} × r^{pe:.2f}; "
        f"beim Beispiel {pa * EXAMPLE_R ** pe:.2f} gegen {b_of(par, EXAMPLE_R):.2f} Punkte")
    for label, key in (("N", "n"), ("Vergleichen je Nutzenwert", "rbin")):
        cal = {str(name): round(rel_rms(par, q), 3) for name, q in sop.groupby(key, observed=True)}
        c90 = (100 * (zr <= par["k90"]).groupby(sop[key], observed=True).mean()).round(1).to_dict()
        c95 = (100 * (zr <= par["k95"]).groupby(sop[key], observed=True).mean()).round(1).to_dict()
        say(f"nach {label}: gemessen/Formel {cal}\n     innerhalb der 90-%-Spanne {c90}\n     innerhalb der 95-%-Spanne {c95}")
    cd90 = (zr <= par["k90"]).groupby(sop["design"]).mean()
    cd95 = (zr <= par["k95"]).groupby(sop["design"]).mean()
    say(f"je Design, 90-%-Spanne: schlechtestes {100 * cd90.min():.1f} | unteres Zehntel {100 * cd90.quantile(.1):.1f} | Median "
        f"{100 * cd90.median():.1f} | bestes {100 * cd90.max():.1f}")
    say(f"je Design, 95-%-Spanne: schlechtestes {100 * cd95.min():.1f} | unteres Zehntel {100 * cd95.quantile(.1):.1f} | Median "
        f"{100 * cd95.median():.1f} | bestes {100 * cd95.max():.1f}")
    by_rank = {}
    for level, k in (("90", par["k90"]), ("95", par["k95"])):
        by_rank[level] = {"lead": 100 * float((zr <= k)[sop["rank"] == 1].mean()), "other": 100 * float((zr <= k)[sop["rank"] > 1].mean())}
        say(f"{level}-%-Spanne: größtes Produkt {by_rank[level]['lead']:.1f} | andere Produkte {by_rank[level]['other']:.1f}")

    say("-- Prüfung an Daten, die nicht zur Anpassung benutzt wurden (20 Aufteilungen der Designs in zwei Hälften; 4 Aufteilungen der Populationen)")
    rng = np.random.default_rng(5)
    designs = np.array(sorted(sop["design"].unique()))
    hold = {"90": [], "95": []}
    ex_hold = {"90": [], "95": []}
    for _ in range(20):
        pick = set(rng.choice(designs, size=len(designs) // 2, replace=False))
        tr, te = sop[sop["design"].isin(pick)], sop[~sop["design"].isin(pick)]
        q = fit_total(cells(tr))
        ztr, zte = zscore(q, tr), zscore(q, te)
        for level, lv in (("90", 0.90), ("95", 0.95)):
            k = float(ztr.quantile(lv))
            hold[level].append(100 * (zte <= k).groupby(te["n"]).mean())
            ex_hold[level].append(needed(q, EXAMPLE_R, 7, k))
    hold_out = {}
    for level in ("90", "95"):
        c = pd.DataFrame(hold[level])
        hold_out[level] = {"mean": c.mean().round(1).to_dict(), "min": c.min().round(1).to_dict(), "max": c.max().round(1).to_dict(),
                           "example_min": min(ex_hold[level]), "example_median": int(np.median(ex_hold[level])),
                           "example_max": max(ex_hold[level])}
        say(f"   Designs, Niveau {level} %: Abdeckung in der anderen Hälfte Mittel {hold_out[level]['mean']} | kleinste {hold_out[level]['min']} | "
            f"größte {hold_out[level]['max']} | Beispiel ±7: {min(ex_hold[level])} bis {max(ex_hold[level])}, Median "
            f"{int(np.median(ex_hold[level]))}")
    for train in ((0, 1), (2, 3), (0, 2), (1, 3)):
        tr, te = sop[sop["pop"].isin(train)], sop[~sop["pop"].isin(train)]
        q = fit_total(cells(tr))
        k = float(zscore(q, tr).quantile(0.90))
        say(f"   Populationen {train} angepasst, andere geprüft (90 %): {pct((zscore(q, te) <= k).groupby(te['n']).mean())} | Beispiel ±7: "
            f"{needed(q, EXAMPLE_R, 7, k)}")

    # ---- 4. Simulator-Methoden ---------------------------------------------------------------
    say("\n=== 4. Simulator-Methoden (Wahrheit für alle drei: mittlere Logit-Wahlwahrscheinlichkeit der Population) ===")
    factors = {"share_of_preference": (1.0, 1.0)}
    rule_out = {}
    for rule in ("rfc", "first_choice"):
        rows = std[std["rule"] == rule]
        own = fit_total(cells(rows))
        fb, fv = fit_factors(cells(rows), par)
        fbr, fvr = round(fb, 2), round(fv, 2)
        factors[rule] = (fbr, fvr)
        zz = zscore(par, rows, fbr, fvr)
        rule_out[rule] = {"fb": fbr, "fv": fvr, "own": own, "cov90": pct((zz <= par["k90"]).groupby(rows["n"]).mean()),
                          "cov95": pct((zz <= par["k95"]).groupby(rows["n"]).mean())}
        say(f"{RULE_NAMES[rule]}: eigene Kurven b = {own['b']:.2f} × r^{own['be']:.2f}, v = {own['v']:.0f} × r^{own['ve']:.2f} | "
            f"feste Faktoren: b × {fb:.3f}, v × {fv:.3f} (gerundet {fbr}, {fvr}; Seite heute: b × {PAGE['rfc' if rule == 'rfc' else 'fc']})")
        say(f"     Faktor auf b nach r (eigene Kurve / Share of preference): "
            + " | ".join(f"r {r:.2f}: {own['b'] * r ** own['be'] / b_of(par, r):.2f}" for r in (0.5, EXAMPLE_R, 4)))
        rb = pd.cut(rows["ratio"], [0.3, 0.75, 1.5, 3, 7])
        say(f"     mit festen Faktoren und denselben k: 90-%-Spanne {rule_out[rule]['cov90']}, 95-%-Spanne {rule_out[rule]['cov95']} | "
            f"nach r (90 %): {(100 * (zz <= par['k90']).groupby(rb, observed=True).mean()).round(1).tolist()} | gemessen/Formel "
            f"{rel_rms(par, rows, fbr, fvr):.3f}")

    # ---- 5. Varianten ------------------------------------------------------------------------
    say("\n=== 5. Wo die Formel weniger gut passt ===")
    old = df[(df["folder"] == "results") & (df["hb_seed"] == 42) & (df["rule"] == "share_of_preference") & (df["market"] == "std")]
    variants = []
    for label, sel, tf in (("CBC wie in der Formel (nur N 400, zum Vergleich)", (old["method"] == "cbc") & (old["hetero"] == "realistic") & (old["n"] == 400) & (~old["is_none"]), 1.0),
                           ("CBC mit None", (old["method"] == "cbc_none") & (~old["is_none"]), 1.0),
                           ("Best-worst, Aufgabenfaktor 1,6", (old["method"] == "bwcbc") & (~old["is_none"]), 1.6),
                           ("Dual-response, Aufgabenfaktor 1,3", (old["method"] == "drcbc") & (~old["is_none"]), 1.3),
                           ("stärker verschiedene Personen", (old["method"] == "cbc") & (old["hetero"] != "realistic") & (~old["is_none"]), 1.0)):
        rows = old[sel]
        if not len(rows):
            continue
        zz = zscore(par, rows, task_factor=tf)
        variants.append({"variante": label, "laeufe": rows.groupby("run_id").ngroups, "designs": rows["design"].nunique(),
                         "gemessen_durch_formel": rel_rms(par, rows, task_factor=tf), "in_90": 100 * float((zz <= par["k90"]).mean()),
                         "in_95": 100 * float((zz <= par["k95"]).mean())})
        say(f"{label}: {variants[-1]['laeufe']} Läufe, {variants[-1]['designs']} Designs | gemessen/Formel {variants[-1]['gemessen_durch_formel']:.2f} | "
            f"innerhalb 90-%-Spanne {variants[-1]['in_90']:.1f} | 95-%-Spanne {variants[-1]['in_95']:.1f}")
    allm = main_rows[main_rows["rule"] == "share_of_preference"]
    by_products = {}
    for m, g in allm.groupby("n_products"):
        zz = zscore(par, g)
        by_products[int(m)] = {"rel": rel_rms(par, g), "in_90": 100 * float((zz <= par["k90"]).mean()), "in_95": 100 * float((zz <= par["k95"]).mean())}
        say(f"Märkte mit {m} Produkten: gemessen/Formel {by_products[int(m)]['rel']:.2f} | innerhalb 90-%-Spanne {by_products[int(m)]['in_90']:.1f} | "
            f"95-%-Spanne {by_products[int(m)]['in_95']:.1f}")
    pd.DataFrame(variants).round(3).to_csv(TABLES / "gesamtfehler_varianten.csv", index=False)

    # ---- 6. Folgen für den Rechner -------------------------------------------------------------
    say("\n=== 6. Folgen für den Rechner (Mindestzahl 300 wie heute) ===")
    lim90, lim95 = par["k90"] * b_of(par, EXAMPLE_R), par["k95"] * b_of(par, EXAMPLE_R)
    say(f"Beispiel der Seite (r = {EXAMPLE_R:.3f}): b = {b_of(par, EXAMPLE_R):.2f} Punkte, v = {v_of(par, EXAMPLE_R):.0f} | beste erreichbare "
        f"Genauigkeit ±{lim90:.1f} (90 %), ±{lim95:.1f} (95 %); heute ±{Z95 * PAGE['scale'] * EXAMPLE_R ** PAGE['exp']:.1f}")
    cons = []
    designs_tab = (("Beispiel der Seite (3, 4, 4, 3, 5; 10 Aufgaben; 3 Optionen)", EXAMPLE_R),
                   ("4 Attribute mit 3 Stufen, 12 Aufgaben, 4 Optionen", 12 * 3 / 8),
                   ("6 Attribute mit 4 Stufen, 12 Aufgaben, 3 Optionen", 12 * 2 / 18),
                   ("6 Attribute mit 4 Stufen, 8 Aufgaben, 3 Optionen", 8 * 2 / 18),
                   ("8 Attribute mit 5 Stufen, 10 Aufgaben, 3 Optionen", 10 * 2 / 32),
                   ("Beispiel als Best-worst (Aufgabenfaktor 1,6)", EXAMPLE_R * 1.6),
                   ("Beispiel als Dual-response (Aufgabenfaktor 1,3)", EXAMPLE_R * 1.3))
    for name, r in designs_tab:
        for m in (10, 7, 5, 3):
            cons.append({"design": name, "methode": "Share of preference", "ziel": m, "heute": fmt_n(needed_page(r, m), 300),
                         "neu_90": fmt_n(needed(par, r, m, par["k90"])), "neu_95": fmt_n(needed(par, r, m, par["k95"])),
                         "beste_90": round(par["k90"] * b_of(par, r), 1), "beste_95": round(par["k95"] * b_of(par, r), 1),
                         "beste_heute": round(Z95 * PAGE["scale"] * r ** PAGE["exp"], 1)})
    for rule, key in (("rfc", "rfc"), ("first_choice", "fc")):
        fb, fv = factors[rule]
        for m in (10, 7, 5):
            cons.append({"design": designs_tab[0][0], "methode": RULE_NAMES[rule], "ziel": m, "heute": fmt_n(needed_page(EXAMPLE_R, m, PAGE[key]), 300),
                         "neu_90": fmt_n(needed(par, EXAMPLE_R, m, par["k90"], fb, fv)), "neu_95": fmt_n(needed(par, EXAMPLE_R, m, par["k95"], fb, fv)),
                         "beste_90": round(par["k90"] * b_of(par, EXAMPLE_R, fb), 1), "beste_95": round(par["k95"] * b_of(par, EXAMPLE_R, fb), 1),
                         "beste_heute": round(Z95 * PAGE["scale"] * EXAMPLE_R ** PAGE["exp"] * PAGE[key], 1)})
    cons = pd.DataFrame(cons)
    cons.to_csv(TABLES / "gesamtfehler_folgen_fuer_den_rechner.csv", index=False)
    say(cons.to_string(index=False))
    say("-- Spalte 'Precision' der Faustregeln im Beispiel (Personen laut Regel -> Spanne): heute | neu 90 %")
    rules_tab = (("Johnson, 500 je Stufe", 84), ("Orme, 1.000 je Stufe", 167), ("Assele et al.", 210), ("Tang et al.", 280), ("Orme", 300))
    thumb = []
    for name, n in rules_tab:
        today = Z95 * math.sqrt((PAGE["scale"] * EXAMPLE_R ** PAGE["exp"]) ** 2 + 2500 / n)
        thumb.append({"regel": name, "personen": n, "heute": round(today, 1), "neu_90": round(margin(par, EXAMPLE_R, n, par["k90"]), 1)})
        say(f"   {name}: {n} Personen -> ±{today:.1f} | ±{thumb[-1]['neu_90']:.1f}")
    say(f"-- reiner Stichprobenfehler nach Lehrbuch bei 50 % Anteil, 400 Personen: ±{Z95 * math.sqrt(2500 / 400):.1f} (95 %), "
        f"±{Z90 * math.sqrt(2500 / 400):.1f} (90 %); Formel im Beispiel bei 400 Personen: ±{margin(par, EXAMPLE_R, 400, par['k90']):.1f} (90 %), "
        f"±{margin(par, EXAMPLE_R, 400, par['k95']):.1f} (95 %)")

    # ---- 7. Abstand und Gewinner ---------------------------------------------------------------
    say("\n=== 7. Abstand zwischen den zwei größten Produkten und richtiger Gewinner ===")
    keys = ["design", "pop", "n", "rep", "scenario"]
    top = sop[sop["rank"] <= 2].pivot_table(index=keys + ["ratio"], columns="rank", values=["est", "pop_share"]).reset_index()
    top.columns = ["_".join(str(c) for c in col if c != "") for col in top.columns]
    top["d_est"] = 100 * (top["est_1.0"] - top["est_2.0"])
    top["d_pop"] = 100 * (top["pop_share_1.0"] - top["pop_share_2.0"])
    top["d_err"] = top["d_est"] - top["d_pop"]
    est_first = sop.loc[sop.groupby(keys)["est"].idxmax()].set_index(keys)["rank"]
    top["right"] = (est_first.reindex(pd.MultiIndex.from_frame(top[keys])) == 1).to_numpy()
    diff = []
    for n, g in top.groupby("n"):
        one = sop[(sop["n"] == n) & (sop["rank"] <= 2)]["total"]
        diff.append({"n": int(n), "rms_abstand": math.sqrt(float((g["d_err"] ** 2).mean())), "mittel_abstand": float(g["d_err"].mean()),
                     "q90_abstand": float(g["d_err"].abs().quantile(0.9)), "rms_ein_anteil": math.sqrt(float((one ** 2).mean())),
                     "abstand_zu_gross": 100 * float((g["d_err"] > 0).mean()), "richtiger_gewinner": 100 * float(g["right"].mean())})
        say(f"N {n:>4}: Fehler des Abstands {diff[-1]['rms_abstand']:.1f} Punkte (ein einzelner der beiden Anteile {diff[-1]['rms_ein_anteil']:.1f}) | "
            f"Abstand im Mittel um {diff[-1]['mittel_abstand']:+.1f} zu groß, in {diff[-1]['abstand_zu_gross']:.0f} % der Märkte | "
            f"richtiger Gewinner {diff[-1]['richtiger_gewinner']:.1f} %")
    top["gap"] = pd.cut(top["d_pop"], [0, 5, 10, 100], labels=["bis 5 Punkte", "5 bis 10 Punkte", "über 10 Punkte"])
    win = (100 * top.pivot_table(index="gap", columns="n", values="right", observed=True)).round(1)
    gap_share = (100 * top["gap"].value_counts(normalize=True)).round(1).to_dict()
    say("richtiger Gewinner nach wahrem Abstand der zwei größten Produkte (Zeilen) und Personen (Spalten), in %:")
    say(win.to_string())
    say(f"Anteil der Märkte je Klasse: {gap_share}")
    win.to_csv(TABLES / "gesamtfehler_richtiger_gewinner.csv")
    pd.DataFrame(diff).round(3).to_csv(TABLES / "gesamtfehler_abstand.csv", index=False)

    # ---- 7b. alle sechs Stichprobengrößen (50, 100, 250 aus results_small) --------------------
    small = load_folder("results_small")
    six = S6.six_sizes(sop, small) if len(small) else None
    bias = S6.bias_table(sop, small) if len(small) else None
    if six:
        say("\n=== 7b. Zerlegung und richtiger Gewinner für sechs Stichprobengrößen (50, 100, 250: results_small) ===")
        say(pd.DataFrame(six).round(2).T.to_string())
        pd.DataFrame(six).round(3).to_csv(TABLES / "gesamtfehler_sechs_groessen.csv", index=False)
        say("\n=== 7c. Simulierter minus wahrer Anteil nach wahrem Anteil des Produkts (Punkte; jedes Design zählt je Klasse einmal) ===")
        bt = pd.DataFrame({"wahrer Anteil": [r["mid"] for r in bias["all"]], "Mittel in der Klasse": [r["true"] for r in bias["all"]],
                           "Designs": [r["designs"] for r in bias["all"]],
                           "Anteile": [r["shares"] for r in bias["all"]], "alle": [r["err"] for r in bias["all"]]})
        for n in S6.NS:
            bt[str(n)] = pd.Series([r["err"] for r in bias[n]])
        say(bt.round(2).to_string(index=False))
        bt.round(3).to_csv(TABLES / "gesamtfehler_fehler_nach_wahrem_anteil.csv", index=False)
        say(f"Klassen: wahrer Anteil plus/minus 2,5 Punkte; außerhalb (unter 2,5 % oder über 72,5 %): {100 * bias['left_out']:.1f} % der Anteile")
        say(f"wahrer Anteil {S6.BIAS_R_RANGE[0]} bis {S6.BIAS_R_RANGE[1]} % nach r: "
            + " | ".join(f"r {x['r']}: {x['err']:+.2f} ({x['designs']} Designs, {x['shares']} Anteile)" for x in bias["by_r"]))

    # ---- 8. Prüfläufe ------------------------------------------------------------------------
    check = validation(par)
    seed_out = seed_noise(df)

    # ---- Plots und Blog-Grafiken ---------------------------------------------------------------
    plots(sop, par, decomp, win, top, page_cov, zr)
    blog = dict(par=par, exact=exact, factors={k: list(v) for k, v in factors.items()}, decomp=decomp.to_dict("records"),
                win=win.reset_index().to_dict("records"), by_rank=by_rank, page_coverage=page_cov, holdout=hold_out,
                by_products=by_products, diff=diff, thumb=thumb, example={"b": b_of(par, EXAMPLE_R), "v": v_of(par, EXAMPLE_R),
                "n7_90": needed(par, EXAMPLE_R, 7, par["k90"]), "n7_95": needed(par, EXAMPLE_R, 7, par["k95"]),
                "n5_90": needed(par, EXAMPLE_R, 5, par["k90"]), "limit_90": lim90, "limit_95": lim95,
                "m400_90": margin(par, EXAMPLE_R, 400, par["k90"])}, n_runs=n_runs, n_designs=n_designs, validation=check, seeds=seed_out,
                six=six, bias=bias)
    blog_figures(blog, t)
    (TABLES / "formel_gesamtfehler.json").write_text(json.dumps(blog, indent=1, default=str), encoding="utf-8")
    (OUT / "ergebnisse_gesamtfehler.txt").write_text("\n".join(_report) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
def validation(par):
    chk = load_folder("results_check")
    if not len(chk):
        say("\n=== 8. Prüfläufe: noch keine Ergebnisse in results_check/ ===")
        return None
    chk = chk[(chk["rule"] == "share_of_preference") & (chk["market"] == "std") & (~chk["is_none"])].copy()
    say("\n=== 8. Prüfläufe vom 17.11.: neue Populationen, neue Stichprobengrößen, neue Designs (Formel vorher festgelegt) ===")
    out = {"runs": int(chk["run_id"].nunique())}
    pre = dict(b=PREREG["b"], be=PREREG["be"], v=PREREG["v"], ve=PREREG["ve"])
    rng = np.random.default_rng(17)
    for label, sel in (("Beispieldesign, 12 neue Populationen", chk["design"] == "med"), ("3 neue Designs, je 8 Populationen", chk["design"] != "med"),
                       ("alle Prüfläufe", chk["design"] == chk["design"])):
        rows = chk[sel]
        if not len(rows):
            continue
        say(f"-- {label}: {rows['run_id'].nunique()} Läufe, {rows.groupby(['design', 'pop']).ngroups} Populationen, {len(rows)} Anteile")
        for name, p, k90, k95 in (("vorher festgelegt", pre, PREREG["k90"], PREREG["k95"]), ("gerundet wie im Vorschlag", par, par["k90"], par["k95"])):
            zz = zscore(p, rows)
            units = rows.assign(z=zz).groupby(["design", "pop"])["z"].apply(np.array)
            for level, k in (("90", k90), ("95", k95)):
                boots = []
                for _ in range(2000):
                    pick = units.iloc[rng.integers(0, len(units), len(units))]
                    boots.append(100 * float(np.mean(np.concatenate(pick.to_list()) <= k)))
                lo, hi = np.quantile(boots, [0.05, 0.95])
                by = rows.assign(c=zz <= k).groupby(["design", "n"])["c"].mean()
                say(f"   {name}, {level}-%-Spanne: {100 * float((zz <= k).mean()):.1f} % innerhalb (90-%-Bereich bei neu gezogenen Populationen "
                    f"{lo:.1f} bis {hi:.1f}) | je Design und N: " + ", ".join(f"{d} N{n}: {100 * v:.1f}" for (d, n), v in by.items()))
                out[f"{label}|{name}|{level}"] = {"coverage": 100 * float((zz <= k).mean()), "lo": float(lo), "hi": float(hi)}
            say(f"   {name}: gemessen/Formel (Wurzel aus mittlerem quadriertem Fehler) {rel_rms(p, rows):.3f} | je Design und N: "
                + ", ".join(f"{d} N{n}: {rel_rms(p, g):.2f}" for (d, n), g in rows.groupby(["design", "n"])))
    chk["rank"] = chk.groupby(["design", "pop", "n", "rep", "scenario"])["pop_share"].rank(ascending=False, method="first")
    zz = zscore(par, chk)
    say(f"-- größtes Produkt innerhalb der 90-%-Spanne: {100 * float((zz <= par['k90'])[chk['rank'] == 1].mean()):.1f} % | andere: "
        f"{100 * float((zz <= par['k90'])[chk['rank'] > 1].mean()):.1f} %")
    return out


def seed_noise(df):
    long_new = load_folder("results_seedlong")
    if not len(long_new):
        say("\n=== 9. Startwerte bei fünffacher Kettenlänge: noch keine Ergebnisse in results_seedlong/ ===")
        return None
    say("\n=== 9. Streuung der Anteile zwischen HB-Startwerten: Standardlänge gegen fünffache Kettenlänge (dieselben Antworten) ===")
    base = df[(df["method"] == "cbc") & (df["hetero"] == "realistic") & (df["pop"] == 0) & (df["n"] == 400) & (df["rep"] == 0)
              & (df["rule"] == "share_of_preference") & (df["market"] == "std") & (~df["is_none"])]
    short = base[(base["n_burn"] == 10000) & base["folder"].isin(["results", "results_seed"])]
    long_old = base[(base["n_burn"] == 50000) & (base["folder"] == "results_long")]
    long_new = long_new[(long_new["rule"] == "share_of_preference") & (long_new["market"] == "std") & (~long_new["is_none"])]
    cols = ["design", "hb_seed", "scenario", "alt", "est", "ratio"]
    long_all = pd.concat([long_old[cols], long_new[cols]])
    rows = []
    for dsg in sorted(long_all["design"].unique()):
        s_, l_ = short[short["design"] == dsg], long_all[long_all["design"] == dsg]
        if s_["hb_seed"].nunique() < 3 or l_["hb_seed"].nunique() < 3:
            continue
        sd = lambda g: 100 * math.sqrt(float(g.groupby(["scenario", "alt"])["est"].var().mean()))
        rows.append({"design": dsg, "r": float(l_["ratio"].iloc[0]), "startwerte": int(l_["hb_seed"].nunique()), "sd_standard": sd(s_),
                     "sd_fuenffach": sd(l_)})
    if not rows:
        say("noch nicht für ein Design vollständig (drei Startwerte bei fünffacher Länge)")
        return None
    tab = pd.DataFrame(rows).sort_values("r")
    tab["verhaeltnis"] = tab["sd_fuenffach"] / tab["sd_standard"]
    say(tab.round(2).to_string(index=False))
    v_s, v_l = float((tab["sd_standard"] ** 2).mean()), float((tab["sd_fuenffach"] ** 2).mean())
    say(f"mittlere Varianz: Standard {v_s:.2f}, fünffach {v_l:.2f} Quadratpunkte -> Verhältnis {v_l / v_s:.2f} "
        f"(reines Kettenrauschen ließe 0,20 erwarten); als Streuung {math.sqrt(v_s):.2f} gegen {math.sqrt(v_l):.2f} Punkte")
    tab.round(3).to_csv(TABLES / "gesamtfehler_startwerte_kettenlaenge.csv", index=False)
    return {"designs": len(tab), "var_standard": v_s, "var_fuenffach": v_l, "ratio": v_l / v_s}


# ---------------------------------------------------------------------------------------------
def plots(sop, par, decomp, win, top, page_cov, zr):
    S = A.SERIES
    # 14: Abdeckung nach N, Formel der Seite gegen neue Formel
    fig, ax = A.new_fig()
    a = ax[0, 0]
    ns = [150, 400, 1000]
    new90 = pct((zr <= par["k90"]).groupby(sop["n"]).mean())
    new95 = pct((zr <= par["k95"]).groupby(sop["n"]).mean())
    for vals, col, lab in (([page_cov[n] for n in ns], S[1], "Formel der Seite heute (nennt 95 %)"),
                           ([new95[n] for n in ns], S[0], "neue Formel, 95 %"), ([new90[n] for n in ns], S[2], "neue Formel, 90 %")):
        a.plot(range(3), vals, color=col, linewidth=2, marker="o", markersize=8, markeredgecolor=A.SURFACE, markeredgewidth=2, label=lab)
        for i, v in enumerate(vals):
            a.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, -17 if col == S[1] else 9), ha="center", fontsize=10,
                       color=A.BODY)
    for y in (90, 95):
        a.axhline(y, color=A.AXIS_C, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    A.style(a, "Personen", "Anteile innerhalb der Spanne (%)", logx=False)
    a.set_xticks(range(3))
    a.set_xticklabels(["150", "400", "1.000"])
    a.set_xlim(-0.3, 2.3)
    a.set_ylim(86, 97.5)
    a.legend(frameon=False, fontsize=10, labelcolor=A.BODY, loc="lower right")
    A.save(fig, "14_abdeckung_formeln.png", "Die Formel der Seite hält 95 % nicht, die neue Formel hält ihr Niveau",
           "Anteil der simulierten Anteile, deren Fehler gegen den wahren Populationsanteil innerhalb der Spanne liegt. 1.608 Läufe.")

    # 15: gemessener Gesamtfehler je Design nach N mit den Kurven der Formel
    fig, ax = A.new_fig()
    a = ax[0, 0]
    d = cells(sop)
    x = np.geomspace(0.33, 6.8, 80)
    for n, col in zip(ns, S):
        g = d[d["n"] == n]
        a.scatter(np.exp(g["x"]), np.sqrt(g["mse"]), s=30, color=col, alpha=0.75, edgecolors=A.SURFACE, linewidths=1, zorder=3,
                  label=f"{n:,} Personen".replace(",", "."))
        a.plot(x, np.sqrt(model_ms(par, np.log(x), n)), color=col, linewidth=2, zorder=2)
    a.plot(x, par["b"] * x ** par["be"], color=A.INK, linewidth=2, linestyle=(0, (4, 3)), zorder=2, label="fester Teil b (bleibt)")
    A.style(a, "Vergleiche je Nutzenwert (log)", "Fehler gegen den wahren Anteil (Punkte)", ymax=13)
    a.legend(frameon=False, fontsize=10, labelcolor=A.BODY, loc="upper right")
    A.save(fig, "15_gesamtfehler_nach_n.png", "Der gesamte Fehler: ein Teil schrumpft mit mehr Personen, ein Teil bleibt",
           "Punkte: ein Design bei einer Stichprobengröße (Wurzel aus dem mittleren quadrierten Fehler). Linien: neue Formel.")

    # 16: richtiger Gewinner
    fig, ax = A.new_fig()
    a = ax[0, 0]
    w = 0.26
    for j, (n, col) in enumerate(zip(ns, S)):
        vals = [float(win.loc[g, n]) for g in win.index]
        a.bar([i + (j - 1) * (w + 0.02) for i in range(len(vals))], vals, width=w, color=col, label=f"{n:,} Personen".replace(",", "."), zorder=3)
        for i, v in enumerate(vals):
            a.annotate(f"{v:.0f}", (i + (j - 1) * (w + 0.02), v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, color=A.BODY)
    A.style(a, "Wahrer Abstand zwischen den zwei größten Produkten", "Richtiger Gewinner (% der Märkte)", logx=False)
    a.set_xticks(range(len(win.index)))
    a.set_xticklabels(list(win.index))
    a.set_ylim(0, 112)
    a.grid(axis="x", visible=False)
    a.legend(frameon=False, fontsize=10, labelcolor=A.BODY, loc="upper left", ncols=3)
    A.save(fig, "16_richtiger_gewinner.png", "Der richtige Gewinner wird fast immer gefunden, außer bei knappem Rennen",
           "Anteil der Märkte, in denen das Produkt mit dem größten wahren Anteil auch in der Simulation vorn liegt. 67 Designs.")

    # 17: Abstand der zwei größten Produkte
    fig, ax = A.new_fig()
    a = ax[0, 0]
    top = top.copy()
    top["gbin"] = pd.cut(top["d_pop"], [0, 5, 10, 20, 40, 100])
    g = top.groupby("gbin", observed=True)["d_err"]
    labels = ["bis 5", "5 bis 10", "10 bis 20", "20 bis 40", "über 40"]
    means = g.mean().to_list()
    a.bar(range(len(means)), means, width=0.6, color=A.TEAL, zorder=3)
    for i, v in enumerate(means):
        a.annotate(f"{v:+.1f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, color=A.BODY)
    A.style(a, "Wahrer Abstand zwischen den zwei größten Produkten (Punkte)", "Simulierter minus wahrer Abstand (Punkte)", logx=False)
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels)
    a.set_ylim(0, max(means) * 1.2)
    a.grid(axis="x", visible=False)
    A.save(fig, "17_abstand_wird_uebertrieben.png", "Der Simulator übertreibt den Abstand, umso mehr, je größer er wirklich ist",
           "Mittel über alle Stichprobengrößen und Designs. Die Zahl der Personen ändert daran nichts.")

    # 18: Zerlegung nach N
    fig, ax = A.new_fig()
    a = ax[0, 0]
    k = par["k90"]
    xs = np.arange(3)
    parts = (("schrumpft: Rauschen der Schätzung", decomp["davon_schaetzung"], S[1]), ("schrumpft: wer teilnimmt", decomp["davon_stichprobe"], S[0]),
             ("bleibt (Design)", decomp["bleibt"], S[2]))
    bottom = np.zeros(3)
    for lab, vals, col in reversed(parts):
        a.bar(xs, vals, bottom=bottom, width=0.55, color=col, label=lab, zorder=3, edgecolor=A.SURFACE, linewidth=2)
        bottom = bottom + np.array(vals)
    for i, v in enumerate(decomp["gesamt"]):
        a.annotate(f"±{k * math.sqrt(v):.1f} Punkte (90 %)", (i, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10, color=A.BODY)
    A.style(a, "Personen", "Mittlerer quadrierter Fehler (Quadratpunkte)", logx=False)
    a.set_xticks(xs)
    a.set_xticklabels(["150", "400", "1.000"])
    a.set_ylim(0, float(decomp["gesamt"].max()) * 1.15)
    a.grid(axis="x", visible=False)
    handles, labs = a.get_legend_handles_labels()
    a.legend(handles[::-1], labs[::-1], frameon=False, fontsize=10, labelcolor=A.BODY, loc="upper right")
    A.save(fig, "18_woraus_der_fehler_besteht.png", "Woraus der Fehler besteht",
           "Mittel über 67 Designs. Der größte schrumpfende Teil ist das Rauschen der Schätzung; die Formel der Seite kannte ihn bisher nicht.")


# ---------------------------------------------------------------------------------------------
def blog_figures(blog, t):
    par = blog["par"]
    old_dir = BLOG / "alt_vor_review"
    old_dir.mkdir(exist_ok=True)
    for name in ("grafik1_neu.svg.html", "grafik2_neu.svg.html", "grafik4_neu.svg.html", "vorschau_alle_grafiken.html",
                 "vorschau_alle_grafiken.png", "vorschau_grafik1.html"):
        if (BLOG / name).exists() and not (old_dir / name).exists():
            shutil.copy2(BLOG / name, old_dir / name)

    # ---- figure 1: fixed part by design, curve of the formula ----
    ok = t[t["b"] > 0]
    yl, pred = np.log(ok["b"]), np.log(par["b"] * ok["ratio"] ** par["be"])
    r2 = 1 - float(((yl - pred) ** 2).sum() / ((yl - yl.mean()) ** 2).sum())
    A.blog_figure1(t, par["b"], par["be"], r2, blog["n_runs"], float(t["b"].max()) * 1.08)
    f1 = (BLOG / "grafik1_neu.svg.html").read_text(encoding="utf-8")
    f1 = f1.replace("<h3>More comparisons per utility, less estimation error</h3>", "<h3>More comparisons per utility, smaller fixed error</h3>")
    f1 = f1.replace("Estimation error (points)", "Fixed error b (percentage points)")
    f1 = f1.replace("Estimation error against comparisons per utility", "Fixed part of the error against comparisons per utility")
    f1 = f1.replace("Share of preference, markets with three or four products.",
                    "The part of the error that more respondents do not remove. Share of preference, markets with three or four products.")
    (BLOG / "grafik1_neu.svg.html").write_text(f1, encoding="utf-8")

    # ---- figure 2: what shrinks and what stays, by N (measured) ----
    k = par["k90"]
    X = lambda v: 103.3 + v * 43.333
    rows, desc = "", []
    for i, x in enumerate(blog["decomp"]):
        y0 = 56 + 100 * i
        s_, f_, b_ = k * math.sqrt(x["schrumpft"]), k * math.sqrt(x["bleibt"]), k * math.sqrt(x["gesamt"])
        desc.append((s_, f_, b_))
        rows += (f'<rect fill="#c9d6d1" height="24" rx="4" width="{X(s_) - 103.3:.1f}" x="103.3" y="{y0}"></rect>'
                 f'<rect fill="#1a9c8d" height="24" rx="4" width="{X(f_) - 103.3:.1f}" x="103.3" y="{y0 + 29}"></rect>'
                 f'<line stroke="#0d1412" stroke-width="3" x1="103.3" x2="{X(b_):.1f}" y1="{y0 + 63}" y2="{y0 + 63}"></line>'
                 f'<text fill="#33403c" font-size="17" x="{X(s_) + 6:.0f}" y="{y0 + 18}">{s_:.1f}</text>'
                 f'<text fill="#33403c" font-size="17" x="{X(f_) + 6:.0f}" y="{y0 + 47}">{f_:.1f}</text>'
                 f'<text fill="#0d1412" font-size="17" font-weight="600" x="{X(b_) + 6:.0f}" y="{y0 + 69}">{b_:.1f}</text>\n')
    vals = lambda j: ", ".join(f"{d[j]:.1f}" for d in desc[:-1]) + f" and {desc[-1][j]:.1f}"
    grid2 = "".join(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="40" y2="340"></line>' for v in (2, 4, 6, 8, 10, 12))
    labs2 = "".join(f'<text x="{X(v):.1f}" y="364">{v}</text>' for v in (0, 2, 4, 6, 8, 10, 12))
    svg2 = f'''<svg aria-labelledby="sim2-title" role="img" viewBox="0 0 700 420">
<title id="sim2-title">For 150, 400 and 1,000 respondents: the part of the error that shrinks is {vals(0)} points; the fixed part is {vals(1)} points; together {vals(2)} points.</title>
<g fill="#33403c" font-size="18"><rect fill="#c9d6d1" height="16" rx="3" width="16" x="103" y="5"></rect><text x="126" y="19">Shrinking part</text><rect fill="#1a9c8d" height="16" rx="3" width="16" x="268" y="5"></rect><text x="291" y="19">Fixed part</text><line stroke="#0d1412" stroke-width="3" x1="400" x2="418" y1="13" y2="13"></line><text x="424" y="19">Both together</text></g>
<g stroke="#edf1ee" stroke-width="1">{grid2}</g>
<line stroke="#9aa8a3" x1="103.3" x2="103.3" y1="40" y2="340"></line>
<g fill="#55635f" font-size="18" text-anchor="middle">{labs2}</g>
<text fill="#33403c" font-size="18" text-anchor="middle" x="363" y="396">90% margin, points</text>
<g fill="#0d1412" font-size="20" font-weight="600" text-anchor="end"><text x="92" y="92">150</text><text x="92" y="192">400</text><text x="92" y="292">1,000</text></g>
<text fill="#55635f" font-size="16" text-anchor="end" x="92" y="112">people</text>
{rows}</svg>'''
    share_est = 100 * np.mean([x["davon_schaetzung"] / x["schrumpft"] for x in blog["decomp"]])
    f2 = A._fig("More people shrink only part of the error", svg2,
                f"Mean over the {blog['n_designs']} designs, measured against the true share in the population. From 150 to 1,000 people "
                f"the shrinking part falls by more than half; the fixed part changes little. About {share_est:.0f}% of the shrinking part is "
                "noise in the estimate, the rest is who happens to take part.")

    # ---- figure 4: simulator methods, example design with 400 people ----
    bars = []
    for rule, col in (("share_of_preference", "#1a9c8d"), ("rfc", "#7fc4b9"), ("first_choice", "#0d1412")):
        fb, fv = blog["factors"][rule]
        bars.append((RULE_NAMES[rule], margin(par, EXAMPLE_R, 400, k, fb, fv), k * b_of(par, EXAMPLE_R, fb), col))
    unit = 24
    body = '<g fill="#0d1412" font-size="20" font-weight="600">' + "".join(
        f'<text x="0" y="{84 + 64 * i}">{lab}</text>' for i, (lab, _, _, _) in enumerate(bars)) + "</g>\n"
    body += "".join(f'<rect fill="{c}" height="36" rx="6" width="{unit * m:.0f}" x="262" y="{58 + 64 * i}"></rect>'
                    for i, (_, m, _, c) in enumerate(bars)) + "\n"
    body += '<g fill="#0d1412" font-size="24" font-weight="800">' + "".join(
        f'<text x="{262 + unit * m + 10:.0f}" y="{85 + 64 * i}">±{m:.1f}</text>' for i, (_, m, _, _) in enumerate(bars)) + "</g>\n"
    body += '<g fill="#55635f" font-size="18" text-anchor="end">' + "".join(
        f'<text x="760" y="{85 + 64 * i}">±{lim:.1f}</text>' for i, (_, _, lim, _) in enumerate(bars)) + "</g>"
    svg4 = f'''<svg aria-labelledby="sim4-title" role="img" viewBox="0 0 760 246">
<title id="sim4-title">90% margin of the example design with 400 respondents: share of preference {bars[0][1]:.1f}, randomized first choice {bars[1][1]:.1f}, first choice {bars[2][1]:.1f} points. Best reachable: {bars[0][2]:.1f}, {bars[1][2]:.1f} and {bars[2][2]:.1f} points.</title>
<text fill="#55635f" font-size="17" x="262" y="34">90% margin with 400 people, percentage points</text>
<text fill="#55635f" font-size="17" text-anchor="end" x="760" y="34">Best reachable</text>
{body}
</svg>'''
    fbf, fvf = blog["factors"]["first_choice"]
    f4 = A._fig("First choice makes shares less precise", svg4,
                f"Example design: levels 3, 4, 4, 3, 5, with 10 tasks and 3 options. Measured against the choices the simulated people "
                f"would really make. First choice gives each person's whole choice to one product, so small errors in individual "
                f"utilities flip whole choices: the fixed error is {fbf:.2f} times that of share of preference, over {blog['n_designs']} "
                "designs and three sample sizes.")

    # ---- new figure 7: the right winner ----
    cols = [("150", "#7fc4b9"), ("400", "#1a9c8d"), ("1,000", "#0d1412")]
    gaps = [("bis 5 Punkte", "up to 5 points"), ("5 bis 10 Punkte", "5 to 10 points"), ("über 10 Punkte", "more than 10 points")]
    win = {r["gap"]: r for r in blog["win"]}
    body7, desc7 = "", []
    for i, (key, lab) in enumerate(gaps):
        y0 = 62 + 92 * i
        body7 += f'<text fill="#0d1412" font-size="19" font-weight="600" x="0" y="{y0 + 34}">{lab}</text>'
        vals7 = []
        for j, (n, (nlab, col)) in enumerate(zip((150, 400, 1000), cols)):
            v = float(win[key][n])
            vals7.append(v)
            wdt = 3.9 * v
            body7 += (f'<rect fill="{col}" height="20" rx="4" width="{wdt:.0f}" x="252" y="{y0 + 24 * j}"></rect>'
                      f'<text fill="#0d1412" font-size="17" font-weight="700" x="{252 + wdt + 8:.0f}" y="{y0 + 24 * j + 16}">{v:.0f}%</text>')
        body7 += "\n"
        desc7.append(f"{lab}: {vals7[0]:.0f}, {vals7[1]:.0f} and {vals7[2]:.0f} percent")
    legend7 = "".join(f'<rect fill="{col}" height="16" rx="3" width="16" x="{252 + 130 * j}" y="8"></rect>'
                      f'<text x="{252 + 130 * j + 23}" y="22">{nlab} people</text>' for j, (nlab, col) in enumerate(cols))
    svg7 = f'''<svg aria-labelledby="sim7-title" role="img" viewBox="0 0 760 340">
<title id="sim7-title">Share of simulated markets in which the product with the largest true share also came first in the simulation, for 150, 400 and 1,000 respondents, by the true gap between the two largest products. {"; ".join(desc7)}.</title>
<g fill="#33403c" font-size="18">{legend7}</g>
<text fill="#55635f" font-size="16" x="0" y="22">True gap between</text><text fill="#55635f" font-size="16" x="0" y="42">the two largest products</text>
<line stroke="#9aa8a3" x1="252" x2="252" y1="52" y2="326"></line>
{body7}</svg>'''
    d400 = next(x for x in blog["diff"] if x["n"] == 400)
    f7 = A._fig("The right winner is found, unless the race is close", svg7,
                f"Share of simulated markets in which the product with the largest true share also came first in the simulation. "
                f"{blog['n_designs']} designs, share of preference, markets with three or four products. The gap itself is less reliable: "
                f"the simulated lead of the largest product was on average {d400['mittel_abstand']:.0f} points too large, whatever the sample size.")

    if blog.get("six"):  # six sample sizes: line charts instead of the three-size bar charts
        f2 = S6.figure2(blog["six"], k, blog["n_designs"], A._fig)
        f7 = S6.figure7(blog["six"], blog["n_designs"], A._fig)

    f5b = f5c = ""
    if blog.get("bias"):  # figure 5 as a line by true share: one line, and one line per sample size
        f5b, info_b = S6.figure5(blog["bias"], blog["n_designs"], A._fig)
        f5c, info_c = S6.figure5(blog["bias"], blog["n_designs"], A._fig, by_n=True)
        f5c = f5c.replace("sim5-title", "sim5n-title")
        say(f"Grafik 5 als Linie: Nulldurchgang bei {info_b['cross']:.1f} % wahrem Anteil | größter Abstand einer der drei Linien "
            f"(150, 400, 1000) zur Linie aller Läufe, Klassen ab 200 Anteilen: {info_c['spread']:.2f} Punkte")
        (BLOG / "grafik5_variante_linie.svg.html").write_text(f5b, encoding="utf-8")
        (BLOG / "grafik5_variante_linie_je_stichprobe.svg.html").write_text(f5c, encoding="utf-8")
    f6 = (BLOG / "grafik6_neu_zahl_der_produkte.svg.html").read_text(encoding="utf-8")
    f6 = f6.replace("Example design, estimation error in points", "Example design, fixed error in points").replace(
        "Estimation error of the example design by number", "Fixed error of the example design by number")
    # user decision 17.11.: "percentage points" everywhere (not "points"); polyline attributes points="..." stay untouched
    pp = lambda html: re.sub(r"(?<!percentage )\bpoints\b(?!=)", "percentage points", html)
    f1, f2, f4, f6, f7 = pp(f1), pp(f2), pp(f4), pp(f6), pp(f7)
    (BLOG / "grafik1_neu.svg.html").write_text(f1, encoding="utf-8")
    files = {"grafik2_neu.svg.html": f2, "grafik4_neu.svg.html": f4, "grafik6_neu_zahl_der_produkte.svg.html": f6,
             "grafik7_neu_richtiger_gewinner.svg.html": f7}
    for name, html in files.items():
        (BLOG / name).write_text(html, encoding="utf-8")
    css = (OUT.parents[1] / "Choicery/app/static/blog/conjoint-sample-size-calculator/sample-size.css")
    note = lambda txt: f'<p class="vorschau-note">{txt}</p>'
    page = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Vorschau Blog-Grafiken</title>
<link rel="stylesheet" href="{css.as_uri()}">
<style>body{{margin:0;padding:32px;background:#fafbf9;font-family:Inter,Arial,sans-serif}} main{{max-width:1180px;margin:0 auto}}
.vorschau-note{{grid-column:1/-1;margin:26px 0 0;font-size:15px;font-weight:700;color:#0d1412}} h1{{font-size:24px}}</style></head>
<body><main class="ssc"><h1>Vorschau: Grafiken für "What the simulation showed" (Stand 17.11.2026 nach dem Review, Niveau 90 %, nicht eingebaut)</h1>
<section class="block sim"><div class="sim-grid">
{note("Grafik 1 und 2: gleiche Plätze wie heute, neue Daten und neue Formel")}{f1}{f2}
{note("Grafik 4: gleicher Platz wie heute, jetzt als Spanne bei 400 Personen (Grafik 3 zu Best-worst und Dual-response bleibt unverändert)")}{f4}<div></div>
{note("Grafik 5a und 5b (beide kommen rein): links alle Läufe als eine Linie, rechts dieselbe Linie je Stichprobengröße")}{f5b}{f5c}
{note("Weitere neue Grafiken")}{f6}{f7}
</div></section></main></body></html>'''
    (BLOG / "vorschau_alle_grafiken.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
