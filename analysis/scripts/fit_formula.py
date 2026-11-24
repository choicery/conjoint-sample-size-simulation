# Packaged copy of the script fit_formel2.py as it ran for the article. Changed for this archive: it reads the files
# through _data.py (folder or ZIP file in ../../data) and writes no file.
"""The formula for the precision of the price premium, fitted on all blocks of studies.

The studies cross tasks and options (8, 12 and 16 tasks with 2, 3 and 4 options) and include lean designs with few
tasks and many utilities, so that the weights of tasks, options and utilities can be told apart. The options enter
with an exponent of their own; the fit is compared with counting every option fully (exponent 1).

Data: the reference case of the calculator (typical reaction to price, a step worth about 8% of the tested price
range), CBC, default settings of the estimation; priced attributes with 2 to 12 levels; three settings for how much
people agree on the level: nearly everybody prefers it (spread of the gain 0.10 or 0.30), about 3 in 10 prefer the
other level (0.57), almost half do (1.90). The studies with 38% (0.98) have no setting in the calculator; they are
left out of the fit and shown at the end.
One value = one study x one of three base products; error as a share of the tested price range.

Model for the mean absolute error:
    MAE = g x sqrt(B^2 + V / n) x (levels / 2)^c
    B = b0 x r^-pb,  V = v0 x r^-pv,  r = tasks x (options - 1)^a / utilities        (the price counts as one utility)
    "better for most" and "matter of taste": one factor g each; "matter of taste" also an own exponent c for the
    levels when that pays. Two factors for "matter of taste" (on B and on sqrt(V)) are fitted as a comparison only:
    the data do not tell them apart.
Laplace likelihood with the MAE as its scale. The margin for 9 out of 10 is K x MAE; K = the 90% point of
|error| / MAE over all values, times a safety factor so that every design and every combination of setting and
number of levels reaches 90% (the factor is set on these same data; the check with a design left out is printed).

Usage: python fit_formula.py        (takes a few minutes)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from _data import studies

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
SKIP = {"agree_long", "agree_pv1", "agree_pv025", "thin_agree_pv025", "split_pv025"}
SETTING = {0.10: "agree", 0.30: "agree", 0.57: "most", 0.98: "none", 1.90: "split"}
NAME = {"agree": "clearly better", "most": "better for most", "split": "matter of taste", "none": "38% prefer the other (no setting)"}
rows = []
for r in list(studies("premium_studies", "addon")) + list(studies("premium_studies", "ext1")):
    a = r.get("addon")
    if not a or r.get("status") != "ok" or r["arm"] in SKIP or (r.get("hb_prior_variance") or 4.0) != 4.0:
        continue
    span = max(r["price"]["values"]) - min(r["price"]["values"])
    P = sum(l - 1 for l in r["levels"]) + 1
    for row in r["rows"]:
        if row["app_median"] is None or row["true_median"] is None or (row.get("app_n_successful") or 0) < 150:
            continue
        rows.append(dict(study=r["family"] + "__" + r["run_id"], night=r["family"] == "ext1", T=r["tasks"], O=r["options"], P=P, L=r["levels"][a["attr"]], n=r["n"], pop=r["pop"],
                         setting=SETTING[round(float(a["gain_sd"]), 2)], weak=r["price"]["span_mean"] < 3, big=a["gain_mean"] > 0.5,
                         e=(row["app_median"] - row["true_median"]) / span))
everything = pd.DataFrame(rows)
everything["a"] = everything["e"].abs()
everything["design"] = everything["T"].astype(str) + "x" + everything["O"].astype(str) + ", " + (everything["P"] - everything["L"] + 2).astype(str) + " utilities"
ref = everything[~everything["weak"] & ~everything["big"]]
d = ref[ref["setting"] != "none"].reset_index(drop=True)
print(f"reference case: {d['study'].nunique()} studies, {len(d)} premiums | designs {d['design'].nunique()} | levels {sorted(d['L'].unique())} | respondents {sorted(d['n'].unique())}")
print("   by setting: " + " | ".join(f"{NAME[k]}: {v['study'].nunique()} studies" for k, v in d.groupby("setting")))
print("   (the design is named by tasks x options and by the utilities it would have with a priced attribute of two levels)")


def columns(x):
    return dict(t=np.log(x["T"].to_numpy(float)), o=np.log(x["O"].to_numpy(float) - 1), p=np.log(x["P"].to_numpy(float)), n=x["n"].to_numpy(float),
                lv=np.log(x["L"].to_numpy(float) / 2), most=(x["setting"] == "most").to_numpy(float), split=(x["setting"] == "split").to_numpy(float))


# theta: lb, pb, lv, pv, a, c, log g most, log g split (on B when two), log factor on sqrt(V) for split (when two), c for split
def mae(th, c, own_c, two=False):
    m = c["t"] + th[4] * c["o"] - c["p"]
    b = np.exp(th[0] - th[1] * m + th[7] * c["split"])
    v = np.exp(th[2] - th[3] * m + 2 * (th[8] if two else th[7]) * c["split"])
    expo = th[5] + ((th[9] - th[5]) * c["split"] if own_c else 0)
    return np.exp(th[6] * c["most"]) * np.sqrt(b * b + v / c["n"]) * np.exp(expo * c["lv"])


def fit(x, own_c, starts=5, seed=3, first=None, two=False, fixed_a=None):
    c, A = columns(x), x["a"].to_numpy()
    k = 10

    def nll(th):
        if fixed_a is not None:
            th = np.concatenate([th[:4], [fixed_a], th[5:]])
        s = mae(th, c, own_c, two)
        return float(np.sum(np.log(s) + A / s))

    base = np.array([math.log(0.0035), 1.09, math.log(0.135), 1.05, 0.2, 0.44, math.log(1.12), math.log(1.7), math.log(1.7), 0.14]) if first is None else np.array(first)
    best, rng = None, np.random.default_rng(seed)
    for i in range(starts):
        res = minimize(nll, base + (rng.normal(0, 0.2, k) if i else 0), method="Nelder-Mead", options=dict(maxiter=80000, maxfev=80000, xatol=1e-7, fatol=1e-9))
        res = minimize(nll, res.x, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
    if fixed_a is not None:
        best.x[4] = fixed_a
    return best


print("\n== 1 Fit")
common, own, two = fit(d, False), fit(d, True), fit(d, True, two=True)
print(f"   one exponent for the levels                              -2 log L {2 * common.fun:10.1f}   parameters 8    AIC {2 * common.fun + 16:10.1f}")
print(f"   own exponent for 'matter of taste'                       -2 log L {2 * own.fun:10.1f}   parameters 9    AIC {2 * own.fun + 18:10.1f}")
print(f"   and two factors for 'matter of taste' (comparison only)  -2 log L {2 * two.fun:10.1f}   parameters 10   AIC {2 * two.fun + 20:10.1f}"
      f"   (B x {math.exp(two.x[7]):.2f}, sqrt(V) x {math.exp(two.x[8]):.2f})")
OWN_C = (2 * own.fun + 18) < (2 * common.fun + 16) - 4          # the extra exponent only when it clearly pays
th = (own if OWN_C else common).x
print(f"   used: one factor for matter of taste, {'own exponent for its levels' if OWN_C else 'one exponent for the levels'}")
full = fit(d, OWN_C, starts=3, first=th, fixed_a=1.0)
z_fit, z_full = d["a"].to_numpy() / mae(th, columns(d), OWN_C), d["a"].to_numpy() / mae(full.x, columns(d), OWN_C)
lowest = lambda z: min(float((z[(d["design"] == des).to_numpy()] <= np.quantile(z, .9)).mean()) for des in d["design"].unique())
print(f"   comparison, every option counted fully (exponent 1): -2 log L {2 * full.fun:10.1f}, that is {2 * (full.fun - (own if OWN_C else common).fun):.1f} worse; "
      f"lowest design inside the 90% point: {100 * lowest(z_full):.0f}%, with the fitted exponent {100 * lowest(z_fit):.0f}%")
print(f"   r = tasks x (options - 1)^{th[4]:.3f} / utilities")
print(f"   B = {math.exp(th[0]):.5f} x r^-{th[1]:.3f}      V = {math.exp(th[2]):.4f} x r^-{th[3]:.3f}")
print(f"   levels: (levels / 2)^{th[5]:.3f}" + (f", for matter of taste (levels / 2)^{th[9]:.3f}" if OWN_C else ""))
print(f"   better for most: x {math.exp(th[6]):.3f}      matter of taste: x {math.exp(th[7]):.3f}")

# rounded constants, as the calculator will carry them
R = dict(a=round(float(th[4]), 2), b0=float(f"{math.exp(th[0]):.2g}"), pb=round(float(th[1]), 2), v0=float(f"{math.exp(th[2]):.3g}"), pv=round(float(th[3]), 2),
         c=round(float(th[5]), 2), c_split=round(float(th[9]), 2) if OWN_C else round(float(th[5]), 2), most=round(math.exp(th[6]), 2),
         split=round(math.exp(th[7]), 2))
print(f"   rounded for the calculator: {R}")


def mae_rounded(x):
    r = x["T"] * (x["O"] - 1) ** R["a"] / x["P"]
    split, most = (x["setting"] == "split"), (x["setting"] == "most")
    b = R["b0"] * r ** -R["pb"]
    v = R["v0"] * r ** -R["pv"]
    return np.where(most, R["most"], np.where(split, R["split"], 1.0)) * np.sqrt(b * b + v / x["n"]) * (x["L"] / 2) ** np.where(split, R["c_split"], R["c"])


d["z"] = d["a"] / mae_rounded(d)
d["zs"] = d["e"] / mae_rounded(d)
k90, k95 = float(np.quantile(d["z"], .90)), float(np.quantile(d["z"], .95))
print(f"\n== 2 The 90% point of |error| / MAE with the rounded constants: {k90:.3f} (95% point {k95:.3f}); mean of |error| / MAE {d['z'].mean():.3f}")
groups = {"design": d.groupby("design"), "setting x levels": d.groupby(["setting", "L"])}
need = {}
for name, g in groups.items():
    q = {key: float(np.quantile(x["z"], .9)) for key, x in g if len(x) >= 90}
    worst = max(q, key=q.get)
    need[name] = q[worst] / k90
    print(f"   factor that brings the 90% point of every {name} under K: {need[name]:.3f} (set by {worst}, {len(g.get_group(worst))} premiums)")
SAFETY = max(need.values())
K = math.ceil(k90 * SAFETY * 100 - 1e-9) / 100          # rounded up, so that the design that sets it reaches 90%
SAFETY = K / k90
print(f"   K = {k90:.3f} x {SAFETY:.3f} = {K:.2f}")
d["hit"] = d["z"] <= K


def line(x):
    together = x.groupby("study")["hit"].all()
    return f"{100 * x['hit'].mean():5.1f}% of {len(x):5d} premiums ({x['study'].nunique():4d} studies; all three of a study {100 * together.mean():5.1f}%) | mean error / MAE {x['z'].mean():4.2f}"


print(f"\n== 3 Inside the margin K x MAE (K = {K:.2f}), reference case")
print(f"   all: {line(d)}")
for name, col in (("setting", "setting"), ("levels of the priced attribute", "L"), ("respondents", "n"), ("options per task", "O"), ("tasks", "T")):
    print(f"   by {name}")
    for key, x in d.groupby(col):
        print(f"      {NAME.get(key, str(key)):18s} {line(x)}")
print("   by design (ratio r of the new formula for a priced attribute with two levels)")
order = d.groupby("design").agg(T=("T", "first"), O=("O", "first"), P2=("P", "min"), night=("night", "max")).assign(r=lambda x: x["T"] * (x["O"] - 1) ** R["a"] / x["P2"]).sort_values("r")
for des, info in order.iterrows():
    print(f"      {des:26s} r {info['r']:4.2f}  {line(d[d['design'] == des])}")
print("   by setting and levels")
for (setting, lv), x in d.groupby(["setting", "L"]):
    print(f"      {NAME[setting]:16s} {lv:2d} levels  {line(x)}")
print("   cells of design x respondents x setting x levels with at least 90 premiums and under 85% inside:")
low = [(key, x) for key, x in d.groupby(["design", "n", "setting", "L"]) if len(x) >= 90 and x["hit"].mean() < 0.85]
for key, x in low:
    print(f"      {key[0]}, N {key[1]}, {NAME[key[2]]}, {key[3]} levels: {100 * x['hit'].mean():5.1f}% of {len(x)}")
cells = [x["hit"].mean() for _, x in d.groupby(["design", "n", "setting", "L"]) if len(x) >= 90]
print(f"      {len(low)} of {len(cells)} such cells; cells under 90%: {sum(c < .9 for c in cells)}; lowest {100 * min(cells):.1f}%")
shares = d.groupby("design")["hit"].mean()
print(f"   by design, exactly: lowest {100 * shares.min():.2f}% ({shares.idxmin()}, {int(d[d['design'] == shares.idxmin()]['hit'].sum())} of {int((d['design'] == shares.idxmin()).sum())}), highest {100 * shares.max():.2f}%")
print(f"   sides: true premium below the lower end {100 * (d['zs'] > K).mean():4.1f}% | above the upper end {100 * (d['zs'] < -K).mean():4.1f}%   by setting: "
      + " | ".join(f"{NAME[s]} {100 * (x['zs'] > K).mean():3.1f}% / {100 * (x['zs'] < -K).mean():3.1f}%" for s, x in d.groupby("setting")))

print("\n== 4 With one design left out of the fit")
print("   a) constants and 90% point from the other designs, the safety factor as above | b) the safety factor from the other designs as well")
out = []
for des in order.index:
    own_rows = (d["design"] == des).to_numpy()
    rest = d[~own_rows]
    res = fit(rest, OWN_C, starts=1, first=th)
    rest_z = rest["a"].to_numpy() / mae(res.x, columns(rest), OWN_C)
    own_z = d[own_rows]["a"].to_numpy() / mae(res.x, columns(d[own_rows]), OWN_C)
    k_rest = float(np.quantile(rest_z, .9))
    need_rest = max([float(np.quantile(rest_z[(rest["design"] == other).to_numpy()], .9)) for other in rest["design"].unique() if (rest["design"] == other).sum() >= 90]
                    + [float(np.quantile(rest_z[((rest["setting"] == st) & (rest["L"] == lv)).to_numpy()], .9)) for (st, lv), g in rest.groupby(["setting", "L"]) if len(g) >= 90])
    k_strict = math.ceil(need_rest * 100 - 1e-9) / 100
    out.append((des, float((own_z <= k_rest * SAFETY).mean()), float((own_z <= k_strict).mean()), res.x[4], k_strict, int(own_rows.sum())))
for des, share_a, share_b, a_opt, k_strict, count in out:
    print(f"      {des:26s} a) {100 * share_a:5.1f}%   b) {100 * share_b:5.1f}% (K {k_strict:.2f})   of {count:4d} premiums | exponent of the options {a_opt:.2f}")
print(f"   lowest: a) {100 * min(x[1] for x in out):.1f}%, b) {100 * min(x[2] for x in out):.1f}% | designs under 90%: a) {sum(x[1] < .9 for x in out)}, b) {sum(x[2] < .9 for x in out)} of {len(out)}"
      f" | exponent of the options between {min(x[3] for x in out):.2f} and {max(x[3] for x in out):.2f}")

print("\n== 5 Outside the reference case, with the margin of the new formula (the calculator does not know these conditions)")
other = everything[(everything["weak"] | everything["big"]) & (everything["setting"] != "none")].copy()
other["hit"] = other["a"] / mae_rounded(other) <= K
other["z"] = other["a"] / mae_rounded(other)
for (big, weak), x in other.groupby(["big", "weak"]):
    label = ("step worth about 28% of the price range" if big else "step worth about 8%") + ", " + ("weak reaction to price" if weak else "typical reaction to price")
    print(f"   {label:62s} {line(x)}")
mid = ref[ref["setting"] == "none"].assign(setting="most")
mid = mid.assign(z=mid["a"] / mae_rounded(mid))
print(f"   38% prefer the other level, with the setting 'better for most': {100 * (mid['z'] <= K).mean():.1f}% of {len(mid)} premiums inside"
      f" | with 'matter of taste': {100 * (mid['a'] / mae_rounded(mid.assign(setting='split')) <= K).mean():.1f}%")

print("\n== 6 Examples: margin at 400 respondents | respondents for ±5% of the price range (attribute with the levels shown)")


def one(tasks, options, utilities, levels, setting, n):
    x = pd.DataFrame([dict(T=tasks, O=options, P=utilities, L=levels, n=n, setting=setting)])
    return float(100 * K * mae_rounded(x)[0])


def people(tasks, options, utilities, levels, setting):
    return next((n for n in range(50, 20001) if one(tasks, options, utilities, levels, setting, n) <= 5), None)


for label, (t, o, p, lv) in (("example of the calculator: 3, 4, 4, 3 + price, 10 tasks of 3; the attribute with 4 levels", (10, 3, 11, 4)),
                             ("the same design, a feature with 2 levels", (10, 3, 11, 2)), ("12 tasks of 2, same attributes, 4 levels", (12, 2, 11, 4)),
                             ("16 tasks of 4, same attributes, 4 levels", (16, 4, 11, 4)), ("6 tasks of 2, 30 utilities, 2 levels", (6, 2, 30, 2))):
    print(f"   {label}: " + " | ".join(f"{NAME[s]} ±{one(t, o, p, lv, s, 400):4.1f}, {people(t, o, p, lv, s)}" for s in ("agree", "most", "split")))

