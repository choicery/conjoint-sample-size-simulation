"""Auswertung aller Simulationsläufe zur Grafik 1 der Sample-Size-Seite (Stand 17.11.2026).

Liest nur: _scratch/2026-11-09_sample_size_sim/grid/results*/ und die grid_specs*.json.
Schreibt: ergebnisse.txt, tabellen/*.csv, plots/*.png, blog_vorschlag/*.svg (+ vorschau.html).

Schätzer für den Schätzfehler b (in Prozentpunkten): Für jedes Produkt eines Marktes ist
e = simulierter Anteil minus wahrer Anteil derselben Personen. Zwei Läufe mit derselben Population und
demselben Markt, aber anderen Personen, haben denselben systematischen Fehler und unabhängiges Rauschen.
Das Produkt e_i * e_j ist deshalb unverzerrt für (systematischer Fehler)^2. b^2 = Mittel dieser Produkte
über alle Paare von Läufen und alle Produkte. Nichts wird auf null gekappt.

Kurve: Die Formel des Rechners addiert b^2 zum Stichprobenfehler. Die Kurve wird deshalb so gelegt, dass b^2 im
Mittel stimmt (Quasi-Likelihood mit Log-Link auf b^2 je Design, fit_mean). Eine Gerade durch log(b) (fit_curve,
so wurde die veröffentlichte Kurve gelegt) liegt etwa 3 % tiefer, weil der Logarithmus verrauschter Werte im
Mittel zu klein ist; sie wird nur zum Vergleich ausgegeben.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

OUT = Path(__file__).resolve().parent
GRID = OUT.parent / "2026-11-09_sample_size_sim" / "grid"
PLOTS, TABLES, BLOG = OUT / "plots", OUT / "tabellen", OUT / "blog_vorschlag"
for d in (PLOTS, TABLES, BLOG):
    d.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)

INK, MUTED, BODY, GRID_C, AXIS_C, TEAL, SURFACE = "#0d1412", "#55635f", "#33403c", "#edf1ee", "#9aa8a3", "#1a9c8d", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated for all pairs (dataviz validator, light surface)
PUBLISHED = (3.7, -0.70)  # curve on the page today

_report = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    _report.append(line)
    print(line)


# ---------------------------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------------------------
def load_specs():
    specs = {}
    for f, folder in (("grid_specs.json", "results"), ("grid_specs_fill.json", "results"),
                      ("grid_specs_more.json", "results_more"), ("grid_specs_night.json", None)):
        p = GRID / f
        if not p.exists():
            continue
        for s in json.loads(p.read_text(encoding="utf-8")):
            specs[(folder or s["_results"], s["run_id"])] = s
    return specs


def load_runs():
    specs = load_specs()
    rows, runs = [], []
    for folder in ("results", "results_more", "results_pop", "results_long", "results_seed", "results_repro"):
        for p in sorted((GRID / folder).glob("*.json")) if (GRID / folder).exists() else []:
            r = json.loads(p.read_text(encoding="utf-8"))
            s = specs.get((folder, r["run_id"])) or specs.get(("results", r["run_id"])) or r
            if r.get("status") != "ok" or not r.get("shares"):
                runs.append({"folder": folder, "run_id": r["run_id"], "status": r.get("status")})
                continue
            levels = list(s.get("levels") or r["levels"])
            tasks, options = int(s.get("tasks") or r["tasks"]), int(s.get("options") or r["options"])
            K = sum(l - 1 for l in levels)
            info = {"folder": folder, "run_id": r["run_id"], "design": r["design"], "method": r["method"],
                    "hetero": s.get("hetero_label", r.get("hetero_label")), "pop": int(r.get("pop", 0)),
                    "n": int(r["n"]), "rep": int(r["rep"]), "hb_seed": int(r.get("hb_seed", 42)),
                    "n_burn": int(r.get("n_burn", 10000)), "A": len(levels), "Lmax": max(levels),
                    "mixed": len(set(levels)) > 1, "K": K, "T": tasks, "O": options,
                    "ratio": tasks * (options - 1) / K}
            diag = r.get("diagnostics") or {}
            runs.append({**info, "status": "ok", "hb_s": r.get("hb_seconds"), "rhat_max": diag.get("rhat_max"),
                         "ess_min": diag.get("ess_min"), "corr": r.get("indiv_corr")})
            for x in r["shares"]:
                rows.append({**info, "market": "std", "scenario": str(x["scenario"]), "n_products": x["n_products"],
                             "alt": x["alt"], "rule": x["rule"], "est": x["est"], "truth": x["insample"],
                             "pop_share": x["pop"], "is_none": bool(x.get("is_none"))})
            for x in r.get("shares_extra") or []:
                rows.append({**info, "market": "extra", "scenario": str(x["scenario"]), "n_products": x["n_products"],
                             "alt": x["alt"], "rule": x["rule"], "est": x["est"], "truth": x["insample"],
                             "pop_share": x["pop"], "is_none": False})
    df = pd.DataFrame(rows)
    df["err"] = df["est"] - df["truth"]
    return df, pd.DataFrame(runs)


# ---------------------------------------------------------------------------------------------
# Schätzer
# ---------------------------------------------------------------------------------------------
UNIT = ["design", "pop", "rule", "scenario", "alt"]


def b2_per_product(df, extra_keys=()):
    """One row per product of a market: unbiased b^2 from all pairs of independent runs."""
    keys = UNIT + list(extra_keys)
    g = df.groupby(keys, sort=False)["err"]
    k = g.size()
    s1, s2 = g.sum(), g.apply(lambda e: float((e ** 2).sum()))
    out = pd.DataFrame({"k": k, "b2": (s1 ** 2 - s2) / (k * (k - 1)).where(k > 1), "signed": s1 / k}).reset_index()
    meta = df.groupby(keys, sort=False)[["ratio", "A", "Lmax", "mixed", "K", "T", "O", "n_products", "pop_share",
                                         "market"]].first().reset_index()
    out = out.merge(meta, on=keys)
    return out[out["k"] > 1]


def to_b(b2):
    return 100 * math.sqrt(max(float(b2), 0.0))


def design_table(prod, by=("design",)):
    g = prod.groupby(list(by))
    t = pd.DataFrame({"ratio": g["ratio"].first(), "A": g["A"].first(), "Lmax": g["Lmax"].first(),
                      "mixed": g["mixed"].first(), "T": g["T"].first(), "O": g["O"].first(),
                      "products": g.size(), "pops": g["pop"].nunique(), "b2": g["b2"].mean()}).reset_index()
    t["b"] = [to_b(v) for v in t["b2"]]
    return t.sort_values("ratio")


def fit_curve(t, col="b"):
    ok = t[t[col] > 0]
    c = np.polyfit(np.log(ok["ratio"]), np.log(ok[col]), 1)
    pred = np.polyval(c, np.log(ok["ratio"]))
    y = np.log(ok[col])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(np.exp(c[1])), float(c[0]), float(r2), len(ok)


def fit_mean(t):
    """E[b^2] = (a * r^e)^2 by quasi-likelihood with log link: the mean of measured b^2 over curve b^2 is 1,
    also along log r. Returns a, e, R2 of log b around this curve, number of designs."""
    x, y = np.log(t["ratio"].to_numpy(float)), t["b2"].to_numpy(float) * 1e4
    obj = lambda q: float(np.sum(y / np.exp(q[0] + q[1] * x) + q[0] + q[1] * x))
    res = minimize(obj, x0=[2 * math.log(3.5), -1.3], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 5000})
    a, e = math.exp(res.x[0] / 2), res.x[1] / 2
    ok = t[t["b"] > 0]
    yl, pred = np.log(ok["b"]), np.log(a * ok["ratio"] ** e)
    r2 = 1 - ((yl - pred) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
    return float(a), float(e), float(r2), len(t)


def boot_curve(t, reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    ok = t.reset_index(drop=True)
    out = []
    for _ in range(reps):
        pick = ok.iloc[rng.integers(0, len(ok), len(ok))]
        a, e, _, _ = fit_mean(pick)
        out.append((a, e, a * 1.4286 ** e))
    q = np.quantile(np.array(out), [0.05, 0.95], axis=0)
    return q


def scale_given_exponent(t, exponent):
    """Scale a of b = a * r^exponent with the exponent held fixed, so that b^2 is right on average."""
    return float(math.sqrt(max(float((t["b2"] * 1e4 / t["ratio"] ** (2 * exponent)).mean()), 0.0)))


def needed_n(m, b):
    room = (m / 100 / 1.959963984540054) ** 2 - (b / 100) ** 2
    return None if room <= 0 else max(300, math.ceil(0.25 / room - 1e-9))


# ---------------------------------------------------------------------------------------------
# Plot-Helfer
# ---------------------------------------------------------------------------------------------
def style(ax, xlabel, ylabel, logx=True, ymax=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_C)
    ax.grid(True, color=GRID_C, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_xlabel(xlabel, color=BODY, fontsize=11)
    ax.set_ylabel(ylabel, color=BODY, fontsize=11)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.set_xlim(0.33, 6.8)
        ax.minorticks_off()
    if ymax:
        ax.set_ylim(0, ymax)


def curve_xy(a, e):
    x = np.geomspace(0.33, 6.8, 80)
    return x, a * x ** e


def dots(ax, t, color=INK, label=None, size=46, alpha=0.8):
    ax.scatter(t["ratio"], t["b"], s=size, color=color, alpha=alpha, edgecolors=SURFACE, linewidths=1.5,
               label=label, zorder=3)


def save(fig, name, title, subtitle=None):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.93, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED, wrap=True)
    fig.savefig(PLOTS / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def new_fig(w=8.2, h=5.2, ncols=1, nrows=1, sharey=True):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), sharey=sharey, squeeze=False)
    fig.subplots_adjust(top=0.84 if ncols == 1 else 0.80, left=max(0.05, 0.75 / w), right=0.98, bottom=0.16 if ncols > 1 else 0.12, wspace=0.08, hspace=0.35)
    return fig, ax


# ---------------------------------------------------------------------------------------------
def main():
    df, runs = load_runs()
    ok_runs = runs[runs["status"] == "ok"]
    say("Läufe geladen:", len(ok_runs), "ok,", int((runs["status"] != "ok").sum()), "nicht ok")
    say(ok_runs.groupby("folder").size().to_string())
    base = df[(df["method"] == "cbc") & (df["hetero"] == "realistic") & (df["hb_seed"] == 42)
              & (df["n_burn"] == 10000) & df["folder"].isin(["results", "results_more", "results_pop"])]
    n_runs_fig = base.groupby(["folder", "run_id"]).ngroups
    sop = base[base["rule"] == "share_of_preference"]
    std = sop[sop["market"] == "std"]
    prod_std = b2_per_product(std)
    prod_all = b2_per_product(sop)

    # ---- 1. Grafik 1: b je Design -----------------------------------------------------------
    t = design_table(prod_std)
    old_names = set(df[df["folder"] == "results"]["design"])
    t["quelle"] = np.where(t["design"].isin(old_names), "veröffentlicht (31)", "neu (36)")
    a, e, r2, nd = fit_mean(t)
    la, le, lr2, _ = fit_curve(t)
    q = boot_curve(t)
    say("\n=== 1. Schätzfehler je Design, Märkte mit 3 oder 4 Produkten (wie auf der Seite) ===")
    say(f"Designs: {len(t)} | Läufe in der Grafik: {n_runs_fig} | Populationen je Design: "
        f"{t['pops'].min()} bis {t['pops'].max()}")
    say(f"Kurve heute auf der Seite:        b = {PUBLISHED[0]:.2f} * r^{PUBLISHED[1]:.2f}  (R2 log 0.85, 31 Designs, gekappter Schätzer)")
    say(f"Kurve neu (unverzerrt, alle Daten): b = {a:.2f} * r^{e:.2f}  R2(log) {r2:.3f}  Designs {nd}  "
        f"(so gelegt, dass b^2 im Mittel stimmt)")
    say(f"   zum Vergleich Gerade durch log(b), wie die veröffentlichte Kurve gelegt wurde: b = {la:.2f} * r^{le:.2f}  "
        f"R2(log) {lr2:.3f}")
    t["_bin"] = pd.cut(t["ratio"], [0.3, 0.6, 1.0, 1.6, 2.6, 7])
    for tag, (aa, ee) in (("neu genau", (a, e)), ("neu gerundet", (round(a, 1), round(e, 2))), ("Gerade durch log(b)", (la, le)),
                          ("heute auf der Seite", PUBLISHED)):
        c2 = (aa * t["ratio"] ** ee / 100) ** 2
        gb = (t["b2"].groupby(t["_bin"], observed=True).sum() / c2.groupby(t["_bin"], observed=True).sum()) ** 0.5
        say(f"   gemessener Fehler geteilt durch Kurve ({tag}: {aa:.2f}, {ee:.2f}), nach r 0.3-0.6 | 0.6-1 | 1-1.6 | 1.6-2.6 | 2.6-7: "
            + " | ".join(f"{v:.2f}" for v in gb) + f" | gesamt {math.sqrt(float((t['b2'] / c2).mean())):.2f}")
    t = t.drop(columns="_bin")
    say(f"   90%-Intervall (Bootstrap über Designs): Faktor {q[0][0]:.2f} bis {q[1][0]:.2f}, Exponent "
        f"{q[0][1]:.2f} bis {q[1][1]:.2f}, b beim Beispieldesign (r=1.43) {q[0][2]:.2f} bis {q[1][2]:.2f}")
    for label, sub in (("nur veröffentlichte 31 Designs", t[t["quelle"].str.startswith("ver")]),
                       ("nur neue 36 Designs", t[t["quelle"].str.startswith("neu")]),
                       ("nur gleiche Stufenzahl", t[~t["mixed"]]), ("nur gemischte Stufenzahl", t[t["mixed"]])):
        if len(sub) > 3:
            a2, e2, r22, n2 = fit_mean(sub)
            say(f"   {label:<34} b = {a2:.2f} * r^{e2:.2f}  R2 {r22:.3f}  Designs {n2}  "
                f"(Faktor bei festem Exponenten {e:.2f}: {scale_given_exponent(sub, e):.2f})")
    per_pop = design_table(prod_std, by=("design", "pop"))
    if per_pop["pop"].nunique() > 1:
        wide = per_pop.pivot(index="design", columns="pop", values="b")
        sd_between = float(np.nanmean(wide.std(axis=1, ddof=1) / wide.mean(axis=1)))
        say(f"   Streuung zwischen Populationen desselben Designs: im Mittel {100 * sd_between:.0f} % von b")
    t.round(4).to_csv(TABLES / "b_je_design.csv", index=False)
    per_pop.round(4).to_csv(TABLES / "b_je_design_und_population.csv", index=False)
    say(t[["design", "ratio", "pops", "products", "b", "quelle"]].round(2).to_string(index=False))

    ymax = max(9, math.ceil(t["b"].max() + 0.6))
    pops_txt = (f"{t['pops'].min()} Population" + ("en" if t["pops"].min() > 1 else "") if t["pops"].min() == t["pops"].max()
                else f"Mittel über {t['pops'].min()} bis {t['pops'].max()} Populationen")
    fig, ax = new_fig()
    dots(ax[0, 0], t)
    ax[0, 0].plot(*curve_xy(a, e), color=TEAL, linewidth=2.5, zorder=2,
                  label=f"neu: b = {a:.1f} × r^{e:.2f}".replace("-", "−"))
    ax[0, 0].plot(*curve_xy(*PUBLISHED), color=AXIS_C, linewidth=1.5, zorder=1, label="heute auf der Seite: b = 3.7 × r^−0.70")
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=ymax)
    save(fig, "01_grafik1_alle_designs.png", f"Schätzfehler gegen Vergleiche je Nutzenwert, {len(t)} Designs",
         f"Ein Punkt je Design ({pops_txt}), unverzerrter Schätzer, Märkte mit 3 oder 4 Produkten.")

    fig, ax = new_fig()
    for (name, sub), col in zip(t.groupby("quelle", sort=False), SERIES):
        dots(ax[0, 0], sub, color=col, label=name)
    ax[0, 0].plot(*curve_xy(a, e), color=TEAL, linewidth=2.5, zorder=2)
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=ymax)
    save(fig, "02_alte_und_neue_designs.png", "Veröffentlichte und neue Designs liegen auf derselben Kurve?",
         "Gleicher Schätzer für beide Gruppen.")

    # ---- 2. nach Zahl der Produkte im Markt ----------------------------------------------------
    say("\n=== 2. Schätzfehler nach Zahl der Produkte im simulierten Markt ===")
    counts = sorted(prod_all["n_products"].unique())
    rows_m = []
    fig, ax = new_fig(w=3.6 * len(counts), h=4.6, ncols=len(counts))
    curves_m = {}
    for i, m in enumerate(counts):
        tm = design_table(prod_all[prod_all["n_products"] == m])
        am, em, r2m, nm = fit_mean(tm)
        curves_m[m] = (am, em)
        rows_m.append({"Produkte im Markt": m, "Designs": nm, "Produkte gesamt": int(tm["products"].sum()),
                       "mittlerer wahrer Anteil": round(1 / m, 3), "Faktor": round(am, 2), "Exponent": round(em, 2),
                       "R2": round(r2m, 2), "Faktor bei Exponent der Hauptkurve": round(scale_given_exponent(tm, e), 2),
                       "b bei r=1.43": round(am * 1.4286 ** em, 2)})
        dots(ax[0, i], tm, size=34)
        ax[0, i].plot(*curve_xy(am, em), color=TEAL, linewidth=2.5)
        ax[0, i].plot(*curve_xy(a, e), color=AXIS_C, linewidth=1.5, zorder=1)
        ax[0, i].set_title(f"{m} Produkte", fontsize=11, color=INK, loc="left")
        style(ax[0, i], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)" if i == 0 else "",
              ymax=max(12, ymax))
        tm.assign(n_products=m).round(4).to_csv(TABLES / f"b_je_design_{m}_produkte.csv", index=False)
    save(fig, "03_nach_zahl_der_produkte.png", "Schätzfehler nach Zahl der Produkte im Markt",
         "Türkis: Kurve für diese Produktzahl. Grau: Hauptkurve (3 oder 4 Produkte). Märkte mit 2 und 5 Produkten nur in den Läufen ab 16.11.")
    tab_m = pd.DataFrame(rows_m)
    say(tab_m.to_string(index=False))
    tab_m.to_csv(TABLES / "nach_zahl_der_produkte.csv", index=False)

    fig, ax = new_fig()
    line_cols = {2: SERIES[0], 3: SERIES[1], 4: SERIES[2], 5: INK}
    for m, (am, em) in curves_m.items():
        x, y = curve_xy(am, em)
        ax[0, 0].plot(x, y, color=line_cols.get(m, MUTED), linewidth=2, label=f"{m} Produkte")
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=ymax)
    save(fig, "04_kurven_nach_zahl_der_produkte.png", "Je mehr Produkte im Markt, desto kleiner der Fehler je Produkt",
         "Angepasste Kurve je Produktzahl. Bei mehr Produkten sind die Anteile kleiner, die Fehler in Prozentpunkten auch.")

    # ---- 3. nach Größe des Anteils und Rang ---------------------------------------------------
    say("\n=== 3. Schätzfehler nach Größe des wahren Anteils (relativ zur Hauptkurve des Designs) ===")
    p = prod_all.copy()
    p["curve"] = (a * p["ratio"] ** e) / 100
    p["rel"] = p["b2"] / p["curve"] ** 2
    p["rank"] = p.groupby(["design", "pop", "scenario"])["pop_share"].rank(ascending=False, method="first").astype(int)
    p["bucket"] = pd.cut(p["pop_share"], [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0])

    def rel_table(frame, by):
        g = frame.groupby(by, observed=True)
        return pd.DataFrame({"Produkte": g.size(), "mittlerer wahrer Anteil": g["pop_share"].mean().round(3),
                             "b relativ zur Kurve": np.sqrt(g["rel"].mean().clip(lower=0)).round(2),
                             "Fehler mit Vorzeichen (pp)": (100 * g["signed"].mean()).round(2)})

    for m in counts:
        say(f"-- Märkte mit {m} Produkten")
        say(rel_table(p[p["n_products"] == m], "bucket").to_string())
    say("-- nach Rang im Markt (1 = größter wahrer Anteil), alle Märkte")
    rank_tab = rel_table(p, ["n_products", "rank"])
    say(rank_tab.to_string())
    rank_tab.to_csv(TABLES / "nach_rang.csv")
    rel_table(p, ["n_products", "bucket"]).to_csv(TABLES / "nach_anteilsgroesse.csv")

    fig, ax = new_fig(w=3.6 * len(counts), h=4.4, ncols=len(counts))
    for i, m in enumerate(counts):
        tb = rel_table(p[p["n_products"] == m], "bucket")
        cats = list(p["bucket"].cat.categories)
        tb = tb[tb["Produkte"] >= 15]
        x = np.array([cats.index(iv) for iv in tb.index])
        ax[0, i].bar(x, tb["b relativ zur Kurve"], width=0.5, color=TEAL)
        ax[0, i].axhline(1, color=AXIS_C, linewidth=1.5)
        ax[0, i].set_xlim(-0.7, len(cats) - 0.3)
        ax[0, i].set_xticks(range(len(cats)))
        ax[0, i].set_xticklabels([f"{int(100 * iv.left)}-{int(100 * iv.right)}" for iv in cats], fontsize=8,
                                 rotation=45, ha="right")
        ax[0, i].set_title(f"{m} Produkte", fontsize=11, color=INK, loc="left")
        style(ax[0, i], "Wahrer Anteil (%)", "Fehler relativ zur Hauptkurve" if i == 0 else "", logx=False)
    save(fig, "05_nach_anteilsgroesse.png", "Der Schätzfehler hängt vom Anteil ab: am größten bei Produkten, die den Markt dominieren",
         "Balken: Schätzfehler der Produkte dieser Anteilsklasse, geteilt durch den Kurvenwert ihres Designs (1 = Kurve). Bei 2 Produkten ist er bei 50 zu 50 am kleinsten. Klassen mit unter 15 Produkten fehlen.")

    fig, ax = new_fig(w=3.6 * len(counts), h=4.4, ncols=len(counts))
    for i, m in enumerate(counts):
        tb = rel_table(p[p["n_products"] == m], "rank")
        x = np.arange(len(tb))
        ax[0, i].bar(x, tb["Fehler mit Vorzeichen (pp)"], width=0.42, color=TEAL)
        ax[0, i].axhline(0, color=AXIS_C, linewidth=1.5)
        ax[0, i].set_xlim(-0.7, 4.7)
        ax[0, i].margins(y=0.18)
        ax[0, i].set_xticks(x)
        ax[0, i].set_xticklabels([f"Rang {r}" for r in tb.index], fontsize=9)
        for xi, v in zip(x, tb["Fehler mit Vorzeichen (pp)"]):
            ax[0, i].annotate(f"{v:+.1f}", (xi, v), xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                              ha="center", color=BODY, fontsize=9)
        ax[0, i].set_title(f"{m} Produkte", fontsize=11, color=INK, loc="left")
        style(ax[0, i], "", "Mittlerer Fehler (Prozentpunkte)" if i == 0 else "", logx=False)
    save(fig, "06_marktfuehrer_wird_ueberschaetzt.png", "Das führende Produkt wird überschätzt, die anderen unterschätzt",
         "Mittel über alle Designs. Rang 1 = Produkt mit dem größten wahren Anteil im Markt.")

    # ---- 4. nach Stichprobengröße --------------------------------------------------------------
    say("\n=== 4. Schätzfehler nach Stichprobengröße (nur Paare von Läufen mit gleichem N) ===")
    prod_n = b2_per_product(std, extra_keys=("n",))
    rows_n = []
    fig, ax = new_fig()
    for n_val, col in zip(sorted(prod_n["n"].unique()), SERIES):
        tn = design_table(prod_n[prod_n["n"] == n_val])
        an, en, r2n, nn = fit_mean(tn)
        rows_n.append({"N": n_val, "Designs": nn, "Faktor": round(an, 2), "Exponent": round(en, 2),
                       "Faktor bei Exponent der Hauptkurve": round(scale_given_exponent(tn, e), 2),
                       "b gepoolt (pp)": round(to_b(prod_n[prod_n["n"] == n_val]["b2"].mean()), 2)})
        x, y = curve_xy(scale_given_exponent(tn, e), e)
        ax[0, 0].plot(x, y, color=col, linewidth=2, label=f"N = {n_val:,}".replace(",", "."))
        dots(ax[0, 0], tn[tn["b"] > 0], color=col, size=20, alpha=0.45)
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=max(12, ymax))
    save(fig, "07_nach_stichprobengroesse.png", "Mehr Personen ändern den Schätzfehler kaum",
         "Je N eine Kurve mit dem Exponenten der Hauptkurve; kleine Punkte = einzelne Designs.")
    tab_n = pd.DataFrame(rows_n)
    say(tab_n.to_string(index=False))
    tab_n.to_csv(TABLES / "nach_stichprobengroesse.csv", index=False)

    # ---- 5. Was erklärt die Abweichung von der Kurve? ------------------------------------------
    say("\n=== 5. Abweichung von der Kurve nach Designmerkmal (b geteilt durch Kurvenwert, 1 = wie Kurve) ===")
    t["rel"] = t["b"] / (a * t["ratio"] ** e)
    fig, ax = new_fig(w=13, h=4.2, ncols=4)
    for i, (col, label) in enumerate((("A", "Attribute"), ("Lmax", "Stufen (Maximum)"), ("T", "Aufgaben"), ("O", "Optionen je Aufgabe"))):
        g = t.groupby(col)["rel"].agg(["mean", "size"])
        say(f"-- {label}: " + ", ".join(f"{k}: {v['mean']:.2f} (n={int(v['size'])})" for k, v in g.iterrows()))
        ax[0, i].scatter(t[col], t["rel"], s=30, color=INK, alpha=0.6, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax[0, i].plot(g.index, g["mean"], color=TEAL, linewidth=2, marker="o", markersize=6, zorder=4)
        ax[0, i].axhline(1, color=AXIS_C, linewidth=1.5)
        style(ax[0, i], label, "b relativ zur Kurve" if i == 0 else "", logx=False)
        ax[0, i].set_xticks(sorted(t[col].unique()))
    save(fig, "08_abweichung_nach_designmerkmal.png", "Was die Kurve nicht erklärt",
         "Punkte: Designs. Türkis: Mittel je Ausprägung. Über 1 = mehr Fehler als die Kurve sagt.")
    fig, ax = new_fig()
    for (o_val, sub), col in zip(t.groupby("O"), SERIES):
        dots(ax[0, 0], sub, color=col, label=f"{o_val} Optionen je Aufgabe")
    ax[0, 0].plot(*curve_xy(a, e), color=TEAL, linewidth=2.5, zorder=2)
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=ymax)
    save(fig, "12_nach_optionen_je_aufgabe.png", "Designs mit 2, 3 und 4 Optionen liegen auf derselben Kurve",
         "Ein Punkt je Design, gefärbt nach der Zahl der Optionen je Aufgabe. Türkis: Hauptkurve.")

    fig, ax = new_fig()
    for _, g in per_pop.groupby("design"):
        ax[0, 0].plot([g["ratio"].iloc[0]] * 2, [g["b"].min(), g["b"].max()], color=AXIS_C, linewidth=1, zorder=1)
    ax[0, 0].scatter(per_pop["ratio"], per_pop["b"], s=14, color=SERIES[0], alpha=0.55, edgecolors="none", zorder=2,
                     label="eine Population (6 Läufe)")
    dots(ax[0, 0], t, label="Design (Mittel über 4 Populationen)")
    ax[0, 0].plot(*curve_xy(a, e), color=TEAL, linewidth=2.5, zorder=2)
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=max(12, ymax))
    save(fig, "13_streuung_zwischen_populationen.png", "So stark streut ein Design von Population zu Population",
         "Kleine Punkte: je eine simulierte Population. Deshalb war die Grafik mit nur einer Population je Design unruhiger.")

    X = np.column_stack([np.ones(len(t)), np.log(t["ratio"]), np.log(t["A"]), np.log(t["O"])])
    okm = t["b"] > 0
    coef, *_ = np.linalg.lstsq(X[okm], np.log(t["b"][okm]), rcond=None)
    pred = X[okm] @ coef
    yv = np.log(t["b"][okm])
    say(f"Modell mit Attributen und Optionen: log b = {coef[0]:.2f} + {coef[1]:.2f} log r + {coef[2]:.2f} log A + "
        f"{coef[3]:.2f} log O, R2(log) {1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum():.3f}")
    say(f"gemischte Stufenzahl: b relativ zur Kurve {t[t['mixed']]['rel'].mean():.2f} | gleiche Stufenzahl "
        f"{t[~t['mixed']]['rel'].mean():.2f}")

    # ---- 6. Simulator-Methoden --------------------------------------------------------------------
    say("\n=== 6. Simulator-Methoden relativ zu Share of Preference (Märkte mit 3 oder 4 Produkten) ===")
    rules = b2_per_product(base[base["market"] == "std"])
    fig, ax = new_fig()
    names = {"share_of_preference": "Share of preference", "rfc": "Randomized first choice", "first_choice": "First choice"}
    rule_tabs = {}
    for (rule, label), col in zip(names.items(), SERIES):
        tr = design_table(rules[rules["rule"] == rule])
        rule_tabs[rule] = tr.set_index("design")["b"]
        ar = scale_given_exponent(tr, e)
        ax[0, 0].plot(*curve_xy(ar, e), color=col, linewidth=2, label=label)
        dots(ax[0, 0], tr, color=col, size=18, alpha=0.4)
        say(f"{label:<26} Faktor bei Exponent der Hauptkurve {ar:.2f} -> relativ zu Share of preference "
            f"{ar / scale_given_exponent(design_table(rules[rules['rule'] == 'share_of_preference']), e):.2f}")
    for rule in ("rfc", "first_choice"):
        f = (rule_tabs[rule] / rule_tabs["share_of_preference"]).replace([np.inf, -np.inf], np.nan).dropna()
        ao, eo, _, _ = fit_mean(design_table(rules[rules["rule"] == rule]))
        say(f"   {names[rule]}: eigene Kurve b = {ao:.2f} * r^{eo:.2f}; Median des Verhältnisses je Design {f.median():.2f} "
            f"(Quartile {f.quantile(.25):.2f} bis {f.quantile(.75):.2f}); Seite heute: " + ("1.15" if rule == "rfc" else "1.62"))
    ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
    style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=max(14, ymax))
    save(fig, "09_simulator_methoden.png", "Schätzfehler nach Simulator-Methode")

    # ---- 7. Kettenlänge ---------------------------------------------------------------------------
    long = df[(df["folder"] == "results_long") & (df["rule"] == "share_of_preference") & (df["market"] == "std")]
    if len(long):
        say("\n=== 7. Kettenlänge: gleiche Antworten, mehr Iterationen (N = 400, 2 Datensätze je Design) ===")
        ref = df[df["folder"].isin(["results", "results_more"]) & (df["rule"] == "share_of_preference")
                 & (df["market"] == "std") & (df["n"] == 400) & (df["hb_seed"] == 42) & (df["method"] == "cbc")
                 & (df["hetero"] == "realistic") & df["design"].isin(long["design"].unique())]
        rows_l = []
        for (d, burn), g in pd.concat([ref, long]).groupby(["design", "n_burn"]):
            pr = b2_per_product(g)
            lead = g[g["pop_share"] == g.groupby(["run_id", "scenario"])["pop_share"].transform("max")]
            rr = runs[(runs["design"] == d) & (runs["n_burn"] == burn) & (runs["n"] == 400)
                      & runs["folder"].isin(g["folder"].unique())]
            rows_l.append({"design": d, "ratio": g["ratio"].iloc[0], "Iterationen": f"{burn // 1000}k+{burn // 1000}k",
                           "burn": burn, "b": to_b(pr["b2"].mean()) if len(pr) else np.nan,
                           "Fehler Marktführer (pp)": 100 * lead["err"].mean(), "R-hat max": rr["rhat_max"].mean(),
                           "ESS min": rr["ess_min"].mean()})
        lt = pd.DataFrame(rows_l).sort_values(["ratio", "burn"])
        say(lt.drop(columns="burn").round(2).to_string(index=False))
        lt.to_csv(TABLES / "kettenlaenge.csv", index=False)
        piv = lt.pivot_table(index=["design", "ratio"], columns="burn", values="b").reset_index().sort_values("ratio")
        say("gepoolt (quadratisches Mittel über Designs mit beiden Längen): "
            + " | ".join(f"{c // 1000}k: {np.sqrt((piv.dropna(subset=[10000, c])[c] ** 2).mean()):.2f} "
                         f"(Standard dort {np.sqrt((piv.dropna(subset=[10000, c])[10000] ** 2).mean()):.2f})"
                         for c in piv.columns if isinstance(c, (int, np.integer)) and c != 10000))
        fig, ax = new_fig()
        burns = [c for c in piv.columns if isinstance(c, (int, np.integer))]
        for c, col in zip(burns, [INK, SERIES[0], SERIES[1]]):
            sub = piv.dropna(subset=[c])
            ax[0, 0].scatter(sub["ratio"], sub[c], s=52, color=col, alpha=0.85, edgecolors=SURFACE, linewidths=1.5,
                             label=f"{c // 1000}.000 + {c // 1000}.000 Iterationen" + (" (Standard)" if c == 10000 else ""), zorder=3)
        for _, row in piv.iterrows():
            vals = [row[c] for c in burns if not np.isnan(row[c])]
            ax[0, 0].plot([row["ratio"]] * len(vals), vals, color=AXIS_C, linewidth=1, zorder=1)
        ax[0, 0].plot(*curve_xy(a, e), color=TEAL, linewidth=2, zorder=2)
        ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
        style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Schätzfehler b (Prozentpunkte)", ymax=max(14, ymax))
        save(fig, "10_kettenlaenge.png", "Längere Ketten machen den Schätzfehler nicht kleiner",
             "Gleiche Antworten, andere Kettenlänge. Je Design nur 2 Datensätze, die Punkte streuen deshalb stark. Türkis: Hauptkurve.")

    # ---- 8. Rauschen der Schätzung: gleiche Antworten, anderer Startwert ----------------------------
    seedset = df[(df["rule"] == "share_of_preference") & (df["market"] == "std") & (df["method"] == "cbc")
                 & (df["hetero"] == "realistic") & (df["n_burn"] == 10000) & (df["pop"] == 0) & (df["n"] == 400)
                 & (df["rep"] == 0) & df["folder"].isin(["results", "results_seed"])]
    multi = seedset.groupby("design").filter(lambda g: g["hb_seed"].nunique() >= 3)
    if len(multi):
        say("\n=== 8. Gleiche Antworten, drei Startwerte (N = 400): Streuung der Anteile ===")
        rows_s = []
        for d, g in multi.groupby("design"):
            sd = 100 * math.sqrt(g.groupby(["scenario", "alt"])["est"].var(ddof=1).mean())
            rows_s.append({"design": d, "ratio": g["ratio"].iloc[0], "Streuung je Lauf (pp)": sd,
                           "b laut Kurve (pp)": a * g["ratio"].iloc[0] ** e})
        st = pd.DataFrame(rows_s).sort_values("ratio")
        say(st.round(2).to_string(index=False))
        say(f"Spanne {st['Streuung je Lauf (pp)'].min():.1f} bis {st['Streuung je Lauf (pp)'].max():.1f} pp, Median "
            f"{st['Streuung je Lauf (pp)'].median():.1f}; Seite heute: '1 to 3.7 points' (3 Designs)")
        st.to_csv(TABLES / "rauschen_startwerte.csv", index=False)
        fig, ax = new_fig()
        ax[0, 0].scatter(st["ratio"], st["Streuung je Lauf (pp)"], s=52, color=SERIES[1], edgecolors=SURFACE,
                         linewidths=1.5, label="Rauschen der Schätzung (Streuung je Lauf)", zorder=3)
        ax[0, 0].plot(*curve_xy(a, e), color=TEAL, linewidth=2, label="Schätzfehler b (Hauptkurve)")
        ax[0, 0].legend(frameon=False, fontsize=10, labelcolor=BODY, loc="upper right")
        style(ax[0, 0], "Vergleiche je Nutzenwert (log)", "Prozentpunkte", ymax=max(9, ymax))
        save(fig, "11_rauschen_der_schaetzung.png", "Dieselben Antworten, anderer Startwert: so stark schwanken die Anteile",
             "Dieses Rauschen steckt nicht in der Formel des Rechners. Mehr Iterationen verringern es.")

    # ---- 9. Folgen für den Rechner ------------------------------------------------------------------
    say("\n=== 9. Folgen für den Rechner (Share of preference, CBC) ===")
    examples = [("Beispiel der Seite: 3,4,4,3,5 / 10 Aufgaben / 3 Optionen", [3, 4, 4, 3, 5], 10, 3),
                ("4 Attribute x 3 Stufen / 12 / 4", [3] * 4, 12, 4), ("6 x 4 / 12 / 3", [4] * 6, 12, 3),
                ("6 x 4 / 8 / 3", [4] * 6, 8, 3), ("8 x 5 / 10 / 3", [5] * 8, 10, 3), ("5 x 3 / 15 / 4", [3] * 5, 15, 4)]
    rows_e = []
    for label, lv, T, O in examples:
        r = T * (O - 1) / sum(l - 1 for l in lv)
        row = {"Design": label, "r": round(r, 2)}
        for tag, (aa, ee) in (("heute", PUBLISHED), ("neu", (round(a, 1), round(e, 2)))):
            b = aa * r ** ee
            row[f"b {tag}"] = round(b, 2)
            row[f"beste Genauigkeit {tag}"] = round(1.96 * b, 1)
            for m in (10, 7, 5):
                nn = needed_n(m, b)
                row[f"N ±{m} {tag}"] = "nicht erreichbar" if nn is None else nn
        rows_e.append(row)
    ex = pd.DataFrame(rows_e)
    say(ex.to_string(index=False))
    ex.to_csv(TABLES / "folgen_fuer_den_rechner.csv", index=False)
    say(f"\nNeue Konstanten für sample-size-math.js: B_SCALE = {a:.1f}, B_EXPONENT = {e:.2f} "
        f"(genau: {a:.3f}, {e:.3f}); Designs {len(t)}, Läufe {n_runs_fig}")
    say(f"   (Gerade durch log(b) zum Vergleich: {la:.3f}, {le:.3f} -> Beispiel b {la * 1.4286 ** le:.2f}, N für ±7: "
        f"{needed_n(7, la * 1.4286 ** le)})")
    say("Diagnose aller Läufe der Grafik: R-hat max Median",
        round(float(ok_runs[ok_runs['folder'].isin(['results', 'results_more', 'results_pop'])]['rhat_max'].median()), 2),
        "| ESS min Median", round(float(ok_runs[ok_runs['folder'].isin(['results', 'results_more', 'results_pop'])]['ess_min'].median()), 1))

    blog = blog_numbers(df, base, prod_std, prod_all, p, t, a, e, rules, rule_tabs)
    blog_figure1(t, round(a, 1), round(e, 2), r2, n_runs_fig, ymax)
    blog_figures_more(blog, t, r2, n_runs_fig)
    (OUT / "ergebnisse.txt").write_text("\n".join(_report), encoding="utf-8")
    json.dump({"a": a, "e": e, "r2": r2, "designs": int(len(t)), "runs": int(n_runs_fig), "blog": blog},
              open(TABLES / "kurve.json", "w", encoding="utf-8"), indent=1)


# ---------------------------------------------------------------------------------------------
# Blog-Vorschlag: Grafik 1 im Stil der Seite (gleiches SVG-Gerüst wie heute)
# ---------------------------------------------------------------------------------------------
def blog_figure1(t, a, e, r2, n_runs, ymax):
    top = 10 if ymax > 9 else 8
    X = lambda r: 315 + 135 * math.log2(r)
    Y = lambda b: 420 - b * (420 - 127) / 8 * (8 / top)
    ticks = [0, 2, 4, 6, 8] if top == 8 else [0, 2, 4, 6, 8, 10]
    grid = "".join(f'<line x1="80" x2="720" y1="{Y(v):.0f}" y2="{Y(v):.0f}"></line>' for v in ticks[1:])
    grid += "".join(f'<line x1="{X(v):.0f}" x2="{X(v):.0f}" y1="90" y2="420"></line>' for v in (0.5, 1, 2, 5))
    ylab = "".join(f'<text x="68" y="{Y(v) + 7:.0f}">{v}</text>' for v in ticks)
    xlab = "".join(f'<text x="{X(v):.0f}" y="448">{lab}</text>' for v, lab in ((0.5, "0.5"), (1, "1"), (2, "2"), (5, "5")))
    xs = [80 + 16 * i for i in range(41)]
    pts = " ".join(f"{x:.1f},{max(Y(a * (2 ** ((x - 315) / 135)) ** e), 92):.1f}" for x in xs)
    circles = "".join(f'<circle cx="{X(r):.0f}" cy="{Y(b):.0f}" r="6"></circle>' for r, b in zip(t["ratio"], t["b"]))
    exp_txt = f"{e:.2f}".replace("-", "−")
    svg = f'''<figure class="sim-figure">
<h3>More comparisons per utility, less estimation error</h3>
<svg aria-labelledby="sim1-title" role="img" viewBox="0 80 760 410">
<title id="sim1-title">Estimation error against comparisons per utility for {len(t)} simulated CBC designs, with the fitted curve b = {a:.1f} times r to the power of {e:.2f}.</title>
<g stroke="#edf1ee" stroke-width="1">{grid}</g>
<line stroke="#9aa8a3" x1="80" x2="720" y1="420" y2="420"></line><line stroke="#9aa8a3" x1="80" x2="80" y1="90" y2="420"></line>
<g fill="#55635f" font-size="19" text-anchor="end">{ylab}</g>
<g fill="#55635f" font-size="19" text-anchor="middle">{xlab}</g>
<text fill="#33403c" font-size="19" text-anchor="middle" x="400" y="482">Comparisons per utility (log scale)</text>
<text fill="#33403c" font-size="19" text-anchor="middle" transform="rotate(-90 20 255)" x="20" y="255">Estimation error (points)</text>
<polyline fill="none" points="{pts}" stroke="#1a9c8d" stroke-width="3"></polyline>
<g fill="#0d1412" fill-opacity=".75" stroke="#ffffff" stroke-width="1.5">{circles}</g>
<g font-size="19"><rect fill="#ffffff" height="70" rx="8" stroke="#d9e0dc" width="290" x="426" y="92"></rect><circle cx="446" cy="115" fill="#0d1412" fill-opacity=".8" r="6"></circle><text fill="#33403c" x="462" y="121">One simulated design</text><line stroke="#1a9c8d" stroke-width="3" x1="437" x2="455" y1="143" y2="143"></line><text fill="#33403c" x="462" y="150">b ≈ {a:.1f} × r<tspan baseline-shift="super" font-size="13">{exp_txt}</tspan></text></g>
</svg>
<figcaption>Share of preference, markets with three or four products. Each dot averages {t["pops"].min()} simulated populations and samples of 150, 400 and 1,000. The curve explains {100 * r2:.0f}% of the variation on the log scale.</figcaption>
</figure>'''
    (BLOG / "grafik1_neu.svg.html").write_text(svg, encoding="utf-8")
    css = (OUT.parents[1] / "Choicery/app/static/blog/conjoint-sample-size-calculator/sample-size.css")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Vorschau Grafik 1</title>
<link rel="stylesheet" href="{css.as_uri()}">
<style>body{{margin:0;padding:32px;background:#fafbf9;font-family:Inter,Arial,sans-serif}} main{{max-width:620px}}</style></head>
<body><main class="ssc"><section class="block sim">{svg}</section></main></body></html>'''
    (BLOG / "vorschau_grafik1.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Zahlen für die Texte und Grafiken der Seite
# ---------------------------------------------------------------------------------------------
Z = 1.959963984540054


def blog_numbers(df, base, prod_std, prod_all, p, t, a, e, rules, rule_tabs):
    say("\n=== 10. Zahlen für Texte und Grafiken der Seite ===")
    A, E = round(a, 1), round(e, 2)
    out = {"B_SCALE": A, "B_EXPONENT": E}
    r_ex = 10 * 2 / 14
    b_ex = A * r_ex ** E
    out["example_b"] = b_ex
    out["example_n7"] = needed_n(7, b_ex)
    out["example_limit"] = Z * b_ex
    say(f"Beispieldesign (r = {r_ex:.2f}): b = {b_ex:.2f} | bestmögliche Genauigkeit ±{Z * b_ex:.1f} | N für ±7: {out['example_n7']}"
        f" | heute: b 2.88, ±5.6, N 563")

    # Grafik 2: Fehler nach N, als 95%-Spanne, Mittel über die Designs
    std = base[(base["rule"] == "share_of_preference") & (base["market"] == "std")]
    prod_n = b2_per_product(std, extra_keys=("n",))
    out["by_n"] = []
    say("-- Grafik 2: 95%-Spanne, Mittel über die Designs (quadratisch gemittelt)")
    for n_val in sorted(prod_n["n"].unique()):
        tn = design_table(prod_n[prod_n["n"] == n_val])
        b_rms = math.sqrt((tn["b2"].clip(lower=0)).mean()) * 100
        b_mean = tn["b"].mean()
        samp = Z * math.sqrt(0.25 / n_val) * 100
        est = Z * b_rms
        out["by_n"].append({"n": int(n_val), "sampling": samp, "estimation": est, "both": math.hypot(samp, est),
                            "estimation_simple_mean": Z * b_mean})
        say(f"   N {n_val:>5}: Stichprobe {samp:.1f} | Schätzung {est:.1f} (einfaches Mittel der Designs {Z * b_mean:.1f}) | zusammen "
            f"{math.hypot(samp, est):.1f}")
    say("   Seite heute: Stichprobe 8.0 / 4.9 / 3.1, Schätzung 7.8 / 6.7 / 6.6, zusammen 11.2 / 8.3 / 7.3 (31 Designs)")

    # Grafik 4: Simulator-Methoden, Faktor = geometrisches Mittel der Verhältnisse je Design, mit Bootstrap
    rng = np.random.default_rng(3)
    out["rules"] = {}
    say("-- Grafik 4: Faktor je Simulator-Methode (so dass b^2 im Mittel stimmt, 90%-Intervall Bootstrap über Designs)")
    curve2 = (a * t.set_index("design")["ratio"] ** e) ** 2
    rel_sop = (rule_tabs["share_of_preference"] ** 2 / curve2)
    for rule, label in (("rfc", "Randomized first choice"), ("first_choice", "First choice")):
        f = (rule_tabs[rule] / rule_tabs["share_of_preference"]).replace([np.inf, -np.inf], np.nan).dropna()
        rel_rule = (rule_tabs[rule] ** 2 / curve2)
        both = pd.DataFrame({"rule": rel_rule, "sop": rel_sop}).dropna().to_numpy()
        idx = [rng.integers(0, len(both), len(both)) for _ in range(4000)]
        boots = [math.sqrt(both[i, 0].mean() / both[i, 1].mean()) for i in idx]
        gm = float(math.sqrt(both[:, 0].mean() / both[:, 1].mean()))
        lo, hi = np.quantile(boots, [0.05, 0.95])
        out["rules"][rule] = {"geo_mean": gm, "lo": float(lo), "hi": float(hi), "median": float(f.median()),
                              "q25": float(f.quantile(.25)), "q75": float(f.quantile(.75))}
        say(f"   {label}: {gm:.2f} ({lo:.2f} bis {hi:.2f}); Median {f.median():.2f}, mittlere Hälfte {f.quantile(.25):.2f} bis "
            f"{f.quantile(.75):.2f}")
    for rule in ("share_of_preference", "rfc", "first_choice"):
        say(f"   Beispieldesign 'med' gemessen ({rule}): b = {rule_tabs[rule].get('med', float('nan')):.2f}")

    # neue Grafik: Fehler mit Vorzeichen nach Rang, auch relativ zum Kurvenwert
    p = p.copy()
    p["signed_rel"] = p["signed"] / p["curve"]
    g = p.groupby(["n_products", "rank"])
    rk = pd.DataFrame({"pp": 100 * g["signed"].mean(), "rel": g["signed_rel"].mean(), "share": 100 * g["pop_share"].mean(),
                       "n": g.size()}).reset_index()
    out["rank"] = rk.round(3).to_dict("records")
    say("-- neue Grafik: Fehler mit Vorzeichen nach Rang (pp = Mittel über alle Designs; rel = in Einheiten des Kurvenwerts b)")
    say(rk.round(2).to_string(index=False))
    lead = p[p["rank"] == 1]
    say("   Marktführer, Fehler in Einheiten von b nach Vergleichen je Nutzenwert: "
        + " | ".join(f"r {lo_} bis {hi_}: {lead[(lead['ratio'] >= lo_) & (lead['ratio'] < hi_)]['signed_rel'].mean():+.2f}"
                     for lo_, hi_ in ((0.3, 0.75), (0.75, 1.5), (1.5, 3), (3, 7))))
    std_lead = lead[lead["n_products"].isin([3, 4])]
    out["leader_rel_std"] = float(std_lead["signed_rel"].mean())
    out["leader_pp_std"] = float(100 * std_lead["signed"].mean())
    out["leader_share_overestimated"] = float((std_lead["signed"] > 0).mean())
    say(f"   Märkte mit 3 oder 4 Produkten: Marktführer im Mittel {out['leader_pp_std']:+.1f} pp = {out['leader_rel_std']:+.2f} × b; "
        f"überschätzt in {100 * out['leader_share_overestimated']:.0f} % der Märkte")
    dom = p[(p["pop_share"] > 0.5) & p["n_products"].isin([3, 4])]
    mid = p[(p["pop_share"] > 0.4) & (p["pop_share"] <= 0.6) & p["n_products"].isin([3, 4])]
    out["rel_above_half_std"] = float(math.sqrt(dom["rel"].mean()))
    out["rel_40_60_std"] = float(math.sqrt(mid["rel"].mean()))
    say(f"   Märkte mit 3 oder 4 Produkten: Fehler relativ zur Kurve bei Anteil 40 bis 60 %: {out['rel_40_60_std']:.2f}; über 50 %: "
        f"{out['rel_above_half_std']:.2f}")
    two = p[p["n_products"] == 2]
    near = two[(two["pop_share"] > 0.4) & (two["pop_share"] <= 0.6)]
    out["rel_two_near_half"] = float(math.sqrt(near["rel"].mean()))
    say(f"   Märkte mit 2 Produkten, Anteil 40 bis 60 %: {out['rel_two_near_half']:.2f} × Kurve")

    # Zahl der Produkte: b für das Beispieldesign
    out["by_products"] = []
    for m in sorted(prod_all["n_products"].unique()):
        tm = design_table(prod_all[prod_all["n_products"] == m])
        am = scale_given_exponent(tm, e)
        out["by_products"].append({"products": int(m), "scale": am, "example_b": am * r_ex ** e, "rel": am / a})
    say("-- Zahl der Produkte: b beim Beispieldesign " + " | ".join(
        f"{x['products']} Produkte {x['example_b']:.1f} ({x['rel']:.2f} × Kurve)" for x in out["by_products"]))

    # Streuung zwischen Populationen in Punkten
    per_pop = design_table(prod_std, by=("design", "pop"))
    wide = per_pop.pivot(index="design", columns="pop", values="b")
    out["sd_between_pops_pp"] = float(wide.std(axis=1, ddof=1).median())
    out["sd_between_pops_rel"] = float((wide.std(axis=1, ddof=1) / wide.mean(axis=1)).median())
    say(f"-- Streuung von b zwischen Populationen desselben Designs: Median {out['sd_between_pops_pp']:.2f} pp "
        f"({100 * out['sd_between_pops_rel']:.0f} % von b)")
    all_main = df[df["folder"].isin(["results", "results_more", "results_pop"])]
    out["runs_main_folders"] = int(all_main.groupby(["folder", "run_id"]).ngroups)
    out["runs_all"] = int(df.groupby(["folder", "run_id"]).ngroups)
    say(f"-- Läufe: Grafik 1 {base.groupby(['folder', 'run_id']).ngroups} | alle Hauptläufe inkl. Methoden, None, Heterogenität "
        f"{out['runs_main_folders']} | mit Prüfläufen (Kettenlänge, Startwerte, Wiederholung) {out['runs_all']}")
    return out


def _fig(title, svg, caption):
    return f'<figure class="sim-figure">\n<h3>{title}</h3>\n{svg}\n<figcaption>{caption}</figcaption>\n</figure>'


def _signed(v):
    return f"{v:+.1f}".replace("-", "−")


def blog_figures_more(blog, t, r2, n_runs):
    """Figures 2 and 4 with the new numbers and two new figures, same SVG skeleton and colours as the page."""
    f1 = (BLOG / "grafik1_neu.svg.html").read_text(encoding="utf-8")
    # ---- figure 2: sampling vs estimation by N ----
    X = lambda v: 103.3 + v * 43.333
    rows = ""
    for i, x in enumerate(blog["by_n"]):
        y0 = 56 + 100 * i
        s_, e_, b_ = x["sampling"], x["estimation"], x["both"]
        rows += (f'<rect fill="#c9d6d1" height="24" rx="4" width="{X(s_) - 103.3:.1f}" x="103.3" y="{y0}"></rect>'
                 f'<rect fill="#1a9c8d" height="24" rx="4" width="{X(e_) - 103.3:.1f}" x="103.3" y="{y0 + 29}"></rect>'
                 f'<line stroke="#0d1412" stroke-width="3" x1="103.3" x2="{X(b_):.1f}" y1="{y0 + 63}" y2="{y0 + 63}"></line>'
                 f'<text fill="#33403c" font-size="17" x="{X(s_) + 6:.0f}" y="{y0 + 18}">{s_:.1f}</text>'
                 f'<text fill="#33403c" font-size="17" x="{X(e_) + 6:.0f}" y="{y0 + 47}">{e_:.1f}</text>'
                 f'<text fill="#0d1412" font-size="17" font-weight="600" x="{X(b_) + 6:.0f}" y="{y0 + 69}">{b_:.1f}</text>\n')
    vals = lambda k: ", ".join(f"{x[k]:.1f}" for x in blog["by_n"][:-1]) + f" and {blog['by_n'][-1][k]:.1f}"
    grid2 = "".join(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="40" y2="340"></line>' for v in (2, 4, 6, 8, 10, 12))
    labs2 = "".join(f'<text x="{X(v):.1f}" y="364">{v}</text>' for v in (0, 2, 4, 6, 8, 10, 12))
    svg2 = f'''<svg aria-labelledby="sim2-title" role="img" viewBox="0 0 700 420">
<title id="sim2-title">For 150, 400 and 1,000 respondents: sampling error {vals("sampling")} points; estimation error {vals("estimation")} points; combined {vals("both")} points.</title>
<g fill="#33403c" font-size="18"><rect fill="#c9d6d1" height="16" rx="3" width="16" x="103" y="5"></rect><text x="126" y="19">Sampling</text><rect fill="#1a9c8d" height="16" rx="3" width="16" x="250" y="5"></rect><text x="273" y="19">Estimation</text><line stroke="#0d1412" stroke-width="3" x1="410" x2="428" y1="13" y2="13"></line><text x="434" y="19">Both together</text></g>
<g stroke="#edf1ee" stroke-width="1">{grid2}</g>
<line stroke="#9aa8a3" x1="103.3" x2="103.3" y1="40" y2="340"></line>
<g fill="#55635f" font-size="18" text-anchor="middle">{labs2}</g>
<text fill="#33403c" font-size="18" text-anchor="middle" x="363" y="396">95% margin, points</text>
<g fill="#0d1412" font-size="20" font-weight="600" text-anchor="end"><text x="92" y="92">150</text><text x="92" y="192">400</text><text x="92" y="292">1,000</text></g>
<text fill="#55635f" font-size="16" text-anchor="end" x="92" y="112">people</text>
{rows}</svg>'''
    f2 = _fig("More people shrink sampling error, not estimation error", svg2,
              f"Mean over the {len(t)} designs. Sampling error for a 50% share, estimation error for an average product. From 150 to "
              "1,000 people the sampling error falls by more than half; the estimation error barely moves. Beyond a few hundred "
              "people, the design decides the precision.")

    # ---- figure 4: simulator methods ----
    rfc, fc = round(blog["rules"]["rfc"]["geo_mean"], 2), round(blog["rules"]["first_choice"]["geo_mean"], 2)
    b0 = blog["example_b"]
    bars = [("Share of preference", 1.0, "#1a9c8d"), ("Randomized first choice", rfc, "#7fc4b9"), ("First choice", fc, "#0d1412")]
    body = '<g fill="#0d1412" font-size="20" font-weight="600">' + "".join(
        f'<text x="0" y="{68 + 64 * i}">{lab}</text>' for i, (lab, _, _) in enumerate(bars)) + "</g>\n"
    body += "".join(f'<rect fill="{c}" height="36" rx="6" width="{175 * f:.0f}" x="262" y="{42 + 64 * i}"></rect>'
                    for i, (_, f, c) in enumerate(bars)) + "\n"
    body += '<g fill="#0d1412" font-size="24" font-weight="800">' + "".join(
        f'<text x="{262 + 175 * f + 10:.0f}" y="{69 + 64 * i}">{f:.2f}×</text>'.replace("1.00×", "1.0×")
        for i, (_, f, _) in enumerate(bars)) + "</g>\n"
    body += '<g fill="#55635f" font-size="18" text-anchor="end">' + "".join(
        f'<text x="760" y="{69 + 64 * i}">{b0 * f:.1f} points</text>' for i, (_, f, _) in enumerate(bars)) + "</g>"
    svg4 = f'''<svg aria-labelledby="sim4-title" role="img" viewBox="0 0 760 230">
<title id="sim4-title">Estimation error relative to share of preference: randomized first choice {rfc:.2f} times, first choice {fc:.2f} times; in the example design {b0:.1f}, {b0 * rfc:.1f} and {b0 * fc:.1f} points.</title>
<text fill="#55635f" font-size="17" text-anchor="end" x="760" y="18">Example design</text>
{body}
</svg>'''
    q = blog["rules"]["first_choice"]
    f4 = _fig("First choice almost doubles the estimation error", svg4,
              f"Factors over {len(t)} designs and three sample sizes; for first choice the middle half lay between {q['q25']:.1f} and "
              f"{q['q75']:.1f}. Example design: levels 3, 4, 4, 3, 5, with 10 tasks and 3 options. First choice gives each person's "
              "whole choice to one product, so small errors in individual utilities flip whole choices.")

    # ---- new figure 5: the leading product is overestimated ----
    rk = pd.DataFrame(blog["rank"])
    groups = [(3, "three products"), (4, "four products")]
    zero, unit = 420, 46  # x of zero, pixels per point
    y = 58
    body5 = ""
    names = {1: "Largest", 2: "Second", 3: "Third", 4: "Fourth"}
    for m, glabel in groups:
        sub = rk[rk["n_products"] == m].sort_values("rank")
        body5 += f'<text fill="#55635f" font-size="17" x="0" y="{y - 10}">Market with {glabel}</text>'
        for _, row in sub.iterrows():
            v = row["pp"]
            w = abs(v) * unit
            x0 = zero if v >= 0 else zero - w
            col = "#1a9c8d" if v >= 0 else "#0d1412"
            xl = zero + w + 8 if v >= 0 else zero - w - 8
            body5 += (f'<text fill="#0d1412" font-size="19" font-weight="600" x="0" y="{y + 22}">{names[int(row["rank"])]}</text>'
                      f'<text fill="#55635f" font-size="16" x="92" y="{y + 22}">true share {row["share"]:.0f}%</text>'
                      f'<rect fill="{col}" height="26" rx="5" width="{w:.0f}" x="{x0:.0f}" y="{y + 3}"></rect>'
                      f'<text fill="#0d1412" font-size="19" font-weight="700" text-anchor="{"start" if v >= 0 else "end"}" '
                      f'x="{xl:.0f}" y="{y + 23}">{_signed(v)}</text>\n')
            y += 38
        y += 38
    bottom = y - 34
    ticks5 = "".join(f'<line x1="{zero + k * unit}" x2="{zero + k * unit}" y1="30" y2="{bottom}"></line>' for k in (-4, -2, 2, 4, 6))
    labs5 = "".join(f'<text x="{zero + k * unit}" y="{bottom + 24}">{("+" if k > 0 else "") + str(k).replace("-", "−")}</text>'
                    for k in (-4, -2, 0, 2, 4, 6))
    svg5 = f'''<svg aria-labelledby="sim5-title" role="img" viewBox="0 0 760 {bottom + 60}">
<title id="sim5-title">Mean error of the simulated share by rank of the product: the product with the largest true share is overestimated by about 4 points, the others are underestimated.</title>
<g stroke="#edf1ee" stroke-width="1">{ticks5}</g>
<line stroke="#9aa8a3" x1="{zero}" x2="{zero}" y1="30" y2="{bottom}"></line>
<g fill="#55635f" font-size="18" text-anchor="middle">{labs5}</g>
<text fill="#33403c" font-size="18" text-anchor="middle" x="{zero + unit}" y="{bottom + 52}">Simulated minus true share, points</text>
{body5}</svg>'''
    f5 = _fig("The leading product is overestimated", svg5,
              f"Mean over the {len(t)} designs, share of preference. When the model pulls everyone towards the average, the product "
              "that wins on average wins for too many people. The calculator plans with the error of an average product; for the "
              "leading product, expect more.")

    # ---- new figure 6: number of products in the market ----
    bp = blog["by_products"]
    mx = max(x["example_b"] for x in bp)
    body6 = ""
    for i, x in enumerate(bp):
        w = 380 * x["example_b"] / mx
        col = "#1a9c8d" if x["products"] in (3, 4) else "#7fc4b9"
        body6 += (f'<text fill="#0d1412" font-size="20" font-weight="600" x="0" y="{58 + 52 * i}">{x["products"]} products</text>'
                  f'<rect fill="{col}" height="32" rx="6" width="{w:.0f}" x="150" y="{36 + 52 * i}"></rect>'
                  f'<text fill="#0d1412" font-size="22" font-weight="800" x="{150 + w + 10:.0f}" y="{60 + 52 * i}">{x["example_b"]:.1f}</text>'
                  f'<text fill="#55635f" font-size="18" text-anchor="end" x="760" y="{59 + 52 * i}">average share {100 / x["products"]:.0f}%</text>\n')
    desc6 = ", ".join(f"{x['products']} products {x['example_b']:.1f} points" for x in bp)
    svg6 = f'''<svg aria-labelledby="sim6-title" role="img" viewBox="0 0 760 {36 + 52 * len(bp)}">
<title id="sim6-title">Estimation error of the example design by number of products in the simulated market: {desc6}.</title>
<text fill="#55635f" font-size="17" text-anchor="end" x="760" y="18">Example design, estimation error in points</text>
{body6}</svg>'''
    f6 = _fig("More products in the market, smaller error per product", svg6,
              "Share of preference. With more products, each share is smaller and so is its error in points. The calculator uses "
              "markets with three or four products (darker bars).")

    files = {"grafik2_neu.svg.html": f2, "grafik4_neu.svg.html": f4, "grafik5_neu_marktfuehrer.svg.html": f5,
             "grafik6_neu_zahl_der_produkte.svg.html": f6}
    for name, html in files.items():
        (BLOG / name).write_text(html, encoding="utf-8")
    css = (OUT.parents[1] / "Choicery/app/static/blog/conjoint-sample-size-calculator/sample-size.css")
    note = lambda txt: f'<p class="vorschau-note">{txt}</p>'
    page = f'''<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Vorschau Blog-Grafiken</title>
<link rel="stylesheet" href="{css.as_uri()}">
<style>body{{margin:0;padding:32px;background:#fafbf9;font-family:Inter,Arial,sans-serif}} main{{max-width:1180px;margin:0 auto}}
.vorschau-note{{grid-column:1/-1;margin:26px 0 0;font-size:15px;font-weight:700;color:#0d1412}} h1{{font-size:24px}}</style></head>
<body><main class="ssc"><h1>Vorschau: Grafiken für "What the simulation showed" (Stand 17.11.2026, nicht eingebaut)</h1>
<section class="block sim"><div class="sim-grid">
{note("Grafik 1 und 2: gleiche Grafiken wie heute, neue Daten")}{f1}{f2}
{note("Grafik 4: gleiche Grafik wie heute, neue Faktoren (Grafik 3 zu Best-worst und Dual-response bleibt unverändert)")}{f4}<div></div>
{note("Neue Grafiken")}{f5}{f6}
</div></section></main></body></html>'''
    (BLOG / "vorschau_alle_grafiken.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
