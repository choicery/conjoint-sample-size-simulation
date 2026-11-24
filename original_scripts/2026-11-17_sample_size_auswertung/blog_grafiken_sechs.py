"""Blog figures 2 and 7 for six sample sizes (50, 100, 150, 250, 400, 1,000 respondents), 17.11.2026.

six_sizes(): the same decomposition and winner statistics as in auswertung_gesamtfehler.main, per sample size, for the
stored main runs (150, 400, 1,000) and the small-sample runs (50, 100, 250).
figure2(), figure7(): line charts over the number of respondents (log scale on x), in the style of the page.
"""
import math

import numpy as np
import pandas as pd

INK, BODY, MUTED, GRID, AXIS = "#0d1412", "#33403c", "#55635f", "#edf1ee", "#9aa8a3"
TEAL, LIGHT, GREY = "#1a9c8d", "#7fc4b9", "#8a9a95"
NS = (50, 100, 150, 250, 400, 1000)
GAPS = [("bis 5", "up to 5"), ("5 bis 10", "5 to 10"), ("über 10", "more than 10")]


def six_sizes(sop, small):
    """sop: main rows (share of preference, standard markets); small: rows of results_small in the same layout."""
    keys = ["design", "pop", "n", "rep", "scenario"]
    small = small[(small["rule"] == "share_of_preference") & (small["market"] == "std") & (~small["is_none"])].copy()
    small["rank"] = small.groupby(keys)["pop_share"].rank(ascending=False, method="first")
    out = []
    for src in (small, sop):
        for n, g in src.groupby("n"):
            g = g.assign(err=g["est"] - g["truth"], samp=(g["truth"] - g["pop_share"]) * 100)
            u = g.groupby(["design", "pop", "scenario", "alt"])["err"]
            k, s1, s2 = u.size(), u.sum(), u.apply(lambda e: float((e ** 2).sum()))
            b2 = ((s1 ** 2 - s2) / (k * (k - 1)))[k > 1]  # unbiased squared fixed error from pairs of independent runs
            fixed = 1e4 * float(b2.groupby("design").mean().mean())
            tot = float(g.groupby("design")["total"].apply(lambda v: (v ** 2).mean()).mean())
            samp = float(g.groupby("design")["samp"].apply(lambda v: (v ** 2).mean()).mean())
            top = g[g["rank"] <= 2].pivot_table(index=keys, columns="rank", values=["est", "pop_share"])
            d_est = 100 * (top[("est", 1.0)] - top[("est", 2.0)])
            d_pop = 100 * (top[("pop_share", 1.0)] - top[("pop_share", 2.0)])
            gap = pd.cut(d_pop, [0, 5, 10, 100], labels=[a for a, _ in GAPS])
            first = g.loc[g.groupby(keys)["est"].idxmax()].set_index(keys)["rank"].reindex(top.index) == 1
            row = {"n": int(n), "runs": int(g.groupby(["pop", "rep", "design"]).ngroups), "pops": int(g["pop"].nunique()),
                   "gesamt": tot, "bleibt": fixed, "schrumpft": tot - fixed, "davon_stichprobe": samp,
                   "davon_schaetzung": tot - fixed - samp, "abstand_mittel": float((d_est - d_pop).mean()),
                   "vor_dem_zweiten": 100 * float((d_est > 0).mean()), "gewinner": 100 * float(first.mean())}
            for a, _ in GAPS:
                row[f"vor_dem_zweiten|{a}"] = 100 * float((d_est > 0)[gap == a].mean())
                row[f"gewinner|{a}"] = 100 * float(first[gap == a].mean())
            out.append(row)
    return sorted(out, key=lambda r: r["n"])


def _x(n, x0, x1):
    return x0 + (math.log(n) - math.log(NS[0])) / (math.log(NS[-1]) - math.log(NS[0])) * (x1 - x0)


def _line(pts, color, width=3):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"></circle>' for x, y in pts)
    return f'<polyline fill="none" points="{d}" stroke="{color}" stroke-linejoin="round" stroke-width="{width}"></polyline>{dots}'


def _n_label(n):
    return f"{n:,}"


def figure2(six, k, n_designs, fig):
    x0, x1, y0, y1, top = 78, 668, 352, 52, 18.0  # plot area; y1 is the pixel of the value "top"
    X = lambda n: _x(n, x0 + 30, x1)
    Y = lambda v: y0 - v / top * (y0 - y1)
    rows = {r["n"]: r for r in six}
    s = [k * math.sqrt(rows[n]["schrumpft"]) for n in NS]
    f = [k * math.sqrt(rows[n]["bleibt"]) for n in NS]
    b = [k * math.sqrt(rows[n]["gesamt"]) for n in NS]
    grid = "".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"></line>' for v in (4, 8, 12, 16))
    ylab = "".join(f'<text x="{x0 - 10}" y="{Y(v) + 6:.1f}">{v}</text>' for v in (0, 4, 8, 12, 16))
    xlab = "".join(f'<text x="{X(n):.1f}" y="{y0 + 26}">{_n_label(n)}</text>' for n in NS)
    ticks = "".join(f'<line x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{y0}" y2="{y0 + 6}"></line>' for n in NS)
    lines = (_line([(X(n), Y(v)) for n, v in zip(NS, s)], GREY) + _line([(X(n), Y(v)) for n, v in zip(NS, f)], TEAL)
             + _line([(X(n), Y(v)) for n, v in zip(NS, b)], INK))
    # value labels: every point of "both together"; the two parts at both ends only
    lab = "".join(f'<text fill="{INK}" font-size="16" font-weight="700" text-anchor="middle" x="{X(n):.1f}" y="{Y(v) - 12:.1f}">{v:.1f}</text>'
                  for n, v in zip(NS, b))
    ends = ""
    for i in (0, len(NS) - 1):
        low, high = sorted([(s[i], "s"), (f[i], "f")])
        for (v, _), dy in ((low, 23), (high, 23 if b[i] - high[0] < 2.4 else -12)):
            ends += f'<text fill="{BODY}" font-size="16" text-anchor="middle" x="{X(NS[i]):.1f}" y="{Y(v) + dy:.1f}">{v:.1f}</text>'
    vals = lambda a: ", ".join(f"{v:.1f}" for v in a[:-1]) + f" and {a[-1]:.1f}"
    svg = f'''<svg aria-labelledby="sim2-title" role="img" viewBox="0 0 700 430">
<title id="sim2-title">90% margin in percentage points for 50, 100, 150, 250, 400 and 1,000 respondents. Both parts together: {vals(b)}. Shrinking part: {vals(s)}. Fixed part: {vals(f)}.</title>
<g fill="{BODY}" font-size="18"><line stroke="{INK}" stroke-width="3" x1="{x0}" x2="{x0 + 20}" y1="14" y2="14"></line><text x="{x0 + 27}" y="20">Both together</text><line stroke="{GREY}" stroke-width="3" x1="{x0 + 190}" x2="{x0 + 210}" y1="14" y2="14"></line><text x="{x0 + 217}" y="20">Shrinking part</text><line stroke="{TEAL}" stroke-width="3" x1="{x0 + 385}" x2="{x0 + 405}" y1="14" y2="14"></line><text x="{x0 + 412}" y="20">Fixed part</text></g>
<g stroke="{GRID}" stroke-width="1">{grid}</g>
<line stroke="{AXIS}" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"></line>
<g stroke="{AXIS}">{ticks}</g>
<g fill="{MUTED}" font-size="17" text-anchor="end">{ylab}</g>
<g fill="{MUTED}" font-size="17" text-anchor="middle">{xlab}</g>
<text fill="{BODY}" font-size="17" text-anchor="middle" x="{(x0 + x1) / 2:.0f}" y="{y0 + 56}">Respondents</text>
<text fill="{BODY}" font-size="17" x="0" y="42">90% margin, percentage points</text>
{lines}
{lab}{ends}
</svg>'''
    share_est = 100 * np.mean([rows[n]["davon_schaetzung"] / rows[n]["schrumpft"] for n in NS])
    caption = (f"Mean over the {n_designs} designs, measured against the true share in the population. From 50 to 1,000 people the "
               f"shrinking part falls from {s[0]:.1f} to {s[-1]:.1f} percentage points; the fixed part stays between {min(f):.1f} and {max(f):.1f}. "
               f"About {share_est:.0f}% of the shrinking part is noise in the estimate, the rest is who happens to take part.")
    return fig("More people shrink only part of the error", svg, caption)


def figure7(six, n_designs, fig, key="gewinner"):
    x0, x1, y0, y1, lo, hi = 78, 668, 372, 84, 40.0, 100.0
    X = lambda n: _x(n, x0 + 30, x1)
    Y = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    rows = {r["n"]: r for r in six}
    series = [(a, lab, [rows[n][f"{key}|{a}"] for n in NS], col) for (a, lab), col in zip(GAPS, (LIGHT, TEAL, INK))]
    grid = "".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"></line>' for v in (60, 80, 100))
    ylab = "".join(f'<text x="{x0 - 10}" y="{Y(v) + 6:.1f}">{v}%</text>' for v in (40, 60, 80, 100))
    xlab = "".join(f'<text x="{X(n):.1f}" y="{y0 + 26}">{_n_label(n)}</text>' for n in NS)
    ticks = "".join(f'<line x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{y0}" y2="{y0 + 6}"></line>' for n in NS)
    lines, labels, ends, desc = "", "", "", []
    legend_x = (262, 388, 520)
    for i, (a, lab, vals, col) in enumerate(series):
        lines += _line([(X(n), Y(v)) for n, v in zip(NS, vals)], col)
        dy = -12 if i == 2 else 23  # top line: numbers above; the two lower lines: numbers below
        labels += "".join(f'<text x="{X(n):.1f}" y="{Y(v) + dy:.1f}">{v:.0f}</text>' for n, v in zip(NS, vals))
        ends += (f'<line stroke="{col}" stroke-width="3" x1="{legend_x[i]}" x2="{legend_x[i] + 20}" y1="14" y2="14"></line>'
                 f'<text x="{legend_x[i] + 27}" y="20">{lab}</text>')
        desc.append(f"true gap {lab} percentage points: " + ", ".join(f"{v:.0f}" for v in vals[:-1]) + f" and {vals[-1]:.0f} percent")
    svg = f'''<svg aria-labelledby="sim7-title" role="img" viewBox="0 0 700 450">
<title id="sim7-title">Share of simulated markets in which the product with the largest true share also came first in the simulation, for 50, 100, 150, 250, 400 and 1,000 respondents, by the true gap between the two largest products. {"; ".join(desc)}.</title>
<text fill="{BODY}" font-size="18" x="0" y="20">True gap in percentage points:</text>
<text fill="{BODY}" font-size="17" x="0" y="52">Right winner, share of markets</text>
<g stroke="{GRID}" stroke-width="1">{grid}</g>
<line stroke="{AXIS}" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"></line>
<g stroke="{AXIS}">{ticks}</g>
<g fill="{MUTED}" font-size="17" text-anchor="end">{ylab}</g>
<g fill="{MUTED}" font-size="17" text-anchor="middle">{xlab}</g>
<text fill="{BODY}" font-size="17" text-anchor="middle" x="{(x0 + x1) / 2:.0f}" y="{y0 + 56}">Respondents</text>
{lines}
<g fill="{INK}" font-size="16" font-weight="600" text-anchor="middle">{labels}</g>
<g fill="{BODY}" font-size="18">{ends}</g>
</svg>'''
    lead = [rows[n]["abstand_mittel"] for n in NS]
    caption = (f"Share of simulated markets in which the product with the largest true share also came first in the simulation. "
               f"{n_designs} designs, share of preference, markets with three or four products. The gap itself is less reliable: the "
               f"simulated lead of the largest product was on average {min(lead):.0f} to {max(lead):.0f} percentage points too large, at every sample size.")
    return fig("The right winner is found, unless the race is close", svg, caption)


# ---------------------------------------------------------------------------------------------
# Figure 5 as a line: signed error of a simulated share by the true share of the product
# ---------------------------------------------------------------------------------------------
# Classes centred on round true shares: 5 % stands for 2.5 to 7.5 %, ..., 70 % for 67.5 to 72.5 %.
# Products below 2.5 % or above 72.5 % are left out (about 3 % of all simulated products).
BIAS_CENTRES = list(range(5, 75, 5))
BIAS_EDGES = [c - 2.5 for c in BIAS_CENTRES] + [BIAS_CENTRES[-1] + 2.5]
BIAS_R_EDGES = [0.3, 0.75, 1.5, 3, 7]
BIAS_R_RANGE = (57.5, 72.5)  # classes 60, 65 and 70 % together, for the comparison of thin and rich designs


def bias_table(sop, small):
    """Mean of (simulated minus true share) per class of the true share; every design counts once per class.

    Returns {"all": rows, 50: rows, ..., 1000: rows, "left_out": share of products outside the classes, "by_r": rows};
    rows = list of dicts with mid (centre of the class, %), true (mean true share in the class, %), err (points).
    """
    cols = ["design", "n", "pop_share", "total", "ratio"]
    small = small[(small["rule"] == "share_of_preference") & (small["market"] == "std") & (~small["is_none"])]
    rows = pd.concat([sop[cols], small[cols]], ignore_index=True)
    rows["true"] = 100 * rows["pop_share"]
    rows["bin"] = pd.cut(rows["true"], BIAS_EDGES)

    def table(q):
        per = q.groupby(["bin", "design"], observed=True).agg(err=("total", "mean"), true=("true", "mean"), k=("total", "size"))
        g = per.groupby("bin", observed=True)
        t = pd.DataFrame({"true": g["true"].mean(), "err": g["err"].mean(), "designs": g.size(), "shares": g["k"].sum()}).reset_index()
        t["lo"], t["hi"] = [float(b.left) for b in t["bin"]], [float(b.right) for b in t["bin"]]
        t["mid"] = (t["lo"] + t["hi"]) / 2
        return t.drop(columns="bin").to_dict("records")

    out = {"all": table(rows), "left_out": float(rows["bin"].isna().mean())}
    for n, q in rows.groupby("n"):
        out[int(n)] = table(q)
    # thin against rich designs for large true shares: one wide class, every design counts once
    wide = rows[(rows["true"] > BIAS_R_RANGE[0]) & (rows["true"] <= BIAS_R_RANGE[1])]
    per = wide.groupby([pd.cut(wide["ratio"], BIAS_R_EDGES), "design"], observed=True)["total"].agg(["mean", "size"])
    g = per.groupby(level=0, observed=True)
    out["by_r"] = [{"r": str(k), "err": float(v), "designs": int(d), "shares": int(n)}
                   for k, v, d, n in zip(g["mean"].mean().index, g["mean"].mean(), g.size(), g["size"].sum())]
    return out


def _signed(v, digits=1):
    return f"{v:+.{digits}f}".replace("-", "−")


def figure5(bias, n_designs, fig, by_n=False):
    x0, x1, y0, y1, lo, hi = 78, 668, 352, 62, -4.0, 8.0
    X = lambda v: x0 + v / 80.0 * (x1 - x0)
    Y = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    grid = "".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"></line>' for v in (-4, 4, 8))
    ylab = "".join(f'<text x="{x0 - 10}" y="{Y(v) + 6:.1f}">{_signed(v, 0) if v else "0"}</text>' for v in (-4, 0, 4, 8))
    xlab = "".join(f'<text x="{X(v):.1f}" y="{y0 + 26}">{v}%</text>' for v in range(0, 90, 10))
    ticks = "".join(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y0}" y2="{y0 + 6}"></line>' for v in range(0, 90, 10))
    pts = lambda rows: [(X(r["mid"]), Y(r["err"])) for r in rows]  # every dot exactly above its round true share
    allr = bias["all"]
    cross = next(a["mid"] + (0 - a["err"]) * (b["mid"] - a["mid"]) / (b["err"] - a["err"]) for a, b in zip(allr, allr[1:])
                 if a["err"] < 0 <= b["err"])
    low = min(allr, key=lambda r: r["err"])
    first = allr[0]
    span = f"{allr[0]['mid']:.0f}%, {allr[1]['mid']:.0f}% and so on up to {allr[-1]['mid']:.0f}%"
    why = (f"The error is in percentage points, and that is why the line bends back towards zero at the far left: a product with a "
           f"true share of {first['mid']:.0f}% cannot lose more than {first['mid']:.0f} percentage points. Measured against their own size, "
           f"these smallest products lose the most, about {abs(100 * first['err'] / first['true']):.0f}% of their share. ")
    if by_n:
        series = [(150, "150 people", LIGHT), (400, "400 people", TEAL), (1000, "1,000 people", INK)]
        lines = "".join(_line(pts(bias[n]), col, 2.5) for n, _, col in series)
        legend = "".join(f'<line stroke="{col}" stroke-width="3" x1="{x0 + 170 * i}" x2="{x0 + 170 * i + 20}" y1="14" y2="14"></line>'
                         f'<text x="{x0 + 170 * i + 27}" y="20">{lab}</text>' for i, (_, lab, col) in enumerate(series))
        labels = ""
        spread = max(abs(a["err"] - b["err"]) for n, _, _ in series for a, b in zip(bias[n], allr) if a["shares"] >= 200)
        desc = "; ".join(f"{lab}: " + ", ".join(_signed(r["err"]) for r in bias[n]) for n, lab, _ in series)
        title = (f"Mean of simulated minus true share in percentage points, for true shares of {span}, for three sample sizes. {desc}.")
        caption = (f"The line of the figure before, drawn separately for 150, 400 and 1,000 people; {n_designs} designs, share of "
                   f"preference, markets with three or four products. The three lines lie almost on top of each other: more respondents "
                   f"shrink the random scatter around this line, but not the line itself. A design with more comparisons per utility "
                   f"makes the line flatter.")
    else:
        lines = _line(pts(allr), TEAL)
        legend = ""
        pick = {0, allr.index(low), len(allr) - 1} | {i for i, r in enumerate(allr) if r["mid"] in (40.0, 50.0, 60.0)}
        def place(i, r):
            x, y = X(r["true"]), Y(r["err"])
            if i == 0:  # first point: to the upper right, the line falls away below
                return f'text-anchor="start" x="{x + 10:.1f}" y="{y - 5:.1f}"'
            if r["err"] < 0:  # lowest point: below
                return f'x="{x:.1f}" y="{y + 23:.1f}"'
            if i >= len(allr) - 2:  # flat end: above
                return f'x="{x:.1f}" y="{y - 12:.1f}"'
            return f'text-anchor="end" x="{x - 10:.1f}" y="{y - 6:.1f}"'  # rising part: to the upper left

        labels = "".join(f'<text {place(i, r)}>{_signed(r["err"])} pp</text>' for i, r in enumerate(allr) if i in pick)
        spread = None
        title = (f"Mean of simulated minus true share in percentage points, for true shares of {span}: "
                 + ", ".join(_signed(r["err"]) for r in allr) + ".")
        caption = (f"Each dot is the mean over all simulated products whose true share lies within 2.5 percentage points of the value "
                   f"below it, {n_designs} designs, share of preference, markets with three or four products, 50 to 1,000 people. "
                   f"Products with a true share below 2.5% or above 72.5% are left out ({100 * bias['left_out']:.0f}% of all). "
                   f"When the model pulls everyone towards the "
                   f"average, the product that wins on average wins for too many people: products with a true share below about "
                   f"{cross:.0f}% come out too small, larger ones too large. {why}The calculator plans with the error of an average "
                   f"product; for a leading product, expect more.")
    svg = f'''<svg aria-labelledby="sim5-title" role="img" viewBox="0 0 700 430">
<title id="sim5-title">{title}</title>
<g fill="{BODY}" font-size="18">{legend}</g>
<text fill="{BODY}" font-size="17" x="0" y="{46 if by_n else 26}">Simulated minus true share, percentage points</text>
<g stroke="{GRID}" stroke-width="1">{grid}</g>
<line stroke="{AXIS}" stroke-width="1.5" x1="{x0}" x2="{x1}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"></line>
<g stroke="{AXIS}">{ticks}</g>
<g fill="{MUTED}" font-size="17" text-anchor="end">{ylab}</g>
<g fill="{MUTED}" font-size="17" text-anchor="middle">{xlab}</g>
<text fill="{BODY}" font-size="17" text-anchor="middle" x="{(x0 + x1) / 2:.0f}" y="{y0 + 56}">True share of the product</text>
<g fill="{MUTED}" font-size="16"><text x="{x0 + 10}" y="{Y(0) - 12:.1f}">simulated share too large</text><text text-anchor="end" x="{x1}" y="{Y(0) + 24:.1f}">simulated share too small</text></g>
{lines}
<g fill="{INK}" font-size="16" font-weight="600" text-anchor="middle">{labels}</g>
</svg>'''
    heading = "The shift is the same at every sample size" if by_n else "Large shares are overestimated, small ones underestimated"
    return fig(heading, svg, caption), {"cross": cross, "spread": spread}
