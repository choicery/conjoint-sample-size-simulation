"""All numbers of the draft's simulation section, from the result files. Writes zahlen.json and zahlen.txt.

Run after auswertung_neu.py and kleine_stichproben_neu.py. Numbers that need the small-sample or check runs are
None until those runs are complete; build_draft.py refuses to build while a number it needs is None.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import auswertung_neu as N

A, G, HET, HERE = N.A, N.G, N.HET, N.HERE
sys.path.insert(0, str(HET))
import auswertung_sim as SIM  # noqa: E402

PAR = dict(N.FORMULA)
K90, K95 = PAR["k90"], PAR["k95"]
R = G.EXAMPLE_R
Z = {}


def de(x, nd=1):
    return f"{x:.{nd}f}"


def num(n):
    return None if n is None else f"{n:,}"


def cov(rows, k, tf=1.0):
    return 100 * float((G.zscore(PAR, rows, task_factor=tf) <= k).mean())


def sim_rows(q):
    """Rows of auswertung_sim.load in the column layout G.zscore expects."""
    return q.assign(total=q["err"], x=np.log(q["ratio"]))


blog = json.loads((G.TABLES / "formel_gesamtfehler.json").read_text(encoding="utf-8"))
assert all(abs(float(blog["par"][k]) - PAR[k]) < 1e-9 for k in PAR)

# ---- main runs -------------------------------------------------------------------------------
df, _ = A.load_runs()
df = G.prepare(df)
main_rows = df[~df["is_none"]]
sop = main_rows[(main_rows["market"] == "std") & (main_rows["rule"] == "share_of_preference")].copy()
assert sop["run_id"].nunique() == 1608 and sop["design"].nunique() == 67
sop["z"] = G.zscore(PAR, sop)
key = ["design", "pop", "n", "rep", "scenario"]
sop["rank"] = sop.groupby(key)["pop_share"].rank(ascending=False, method="first")
Z["RUNS_FIT"] = num(1608)
Z["SHARES_FIT"] = num(len(sop))   # simulated choice shares in the runs behind the formula
Z["COV90_N"] = {int(n): de(100 * float((g["z"] <= K90).mean())) for n, g in sop.groupby("n")}
Z["COV95_N"] = {int(n): de(100 * float((g["z"] <= K95).mean())) for n, g in sop.groupby("n")}
Z["RUNS_N"] = {int(n): int(g["run_id"].nunique()) for n, g in sop.groupby("n")}
per_design = (sop["z"] <= K90).groupby(sop["design"]).mean() * 100
Z["DESIGN_MIN"], Z["DESIGN_MAX"] = de(per_design.min(), 0), de(per_design.max(), 0)
Z["LEAD90"] = de(100 * float((sop["z"] <= K90)[sop["rank"] == 1].mean()), 0)
Z["OTHER90"] = de(100 * float((sop["z"] <= K90)[sop["rank"] > 1].mean()), 0)
Z["JOINT90"] = de(100 * float((sop["z"] <= K90).groupby([sop[c] for c in key]).all().mean()), 0)
t = A.design_table(A.b2_per_product(sop))
ok = t[t["b"] > 0]
yl, pred = np.log(ok["b"]), np.log(PAR["b"] * ok["ratio"] ** PAR["be"])
Z["FIG1_R2"] = de(100 * (1 - float(((yl - pred) ** 2).sum() / ((yl - yl.mean()) ** 2).sum())), 0)
thin, rich = t.sort_values("ratio").head(5)["b"].mean(), t.sort_values("ratio").tail(5)["b"].mean()
Z["FIG1_B_THIN"], Z["FIG1_B_RICH"] = de(thin, 0), de(rich, 1)
Z["FIG1_B_THIN_EXACT"], Z["FIG1_B_RICH_EXACT"], Z["FIG1_B_MAX"] = de(thin, 2), de(rich, 2), de(float(t["b"].max()), 2)
o4 = sop[sop["O"] == 4]
Z["FOUR_OPTIONS_REL"] = de(G.rel_rms(PAR, o4), 2)
Z["FOUR_OPTIONS_COV"] = de(100 * float((o4["z"] <= K90).mean()), 1)
Z["FOUR_OPTIONS_DESIGNS"] = int(o4["design"].nunique())
for m, x in blog["by_products"].items():
    Z[f"PRODUCTS_{m}_REL"] = de(x["rel"], 1)
top = sop[sop["rank"] <= 2].pivot_table(index=key, columns="rank", values="pop_share")
Z["GAP_OVER10"] = de(100 * float((100 * (top[1.0] - top[2.0]) > 10).mean()), 0)
real = pd.read_csv(HET / "tabellen" / "echte_studien.csv")
import summarize_real as SR  # noqa: E402
Z["REAL_STUDIES"] = int(real[real["study"].map(SR.group_of) == "market"]["family"].nunique())

# ---- formula and example design ----------------------------------------------------------------
fac = blog["factors"]
Z["RFC_B"], Z["RFC_V"] = de(fac["rfc"][0], 2), de(fac["rfc"][1], 2)
Z["FC_B"], Z["FC_V"] = de(fac["first_choice"][0], 2), de(fac["first_choice"][1], 2)
Z["LIMIT90"], Z["LIMIT95"] = de(G.margin(PAR, R, 0, K90)), de(G.margin(PAR, R, 0, K95))
for target in (6, 7, 8, 10, 12):
    Z[f"N{target}_90"], Z[f"N{target}_95"] = num(G.needed(PAR, R, target, K90)), num(G.needed(PAR, R, target, K95))
for n in (300, 400):
    Z[f"M{n}_90"], Z[f"M{n}_95"] = de(G.margin(PAR, R, n, K90)), de(G.margin(PAR, R, n, K95))
r15 = 15 * 2 / 14
Z["N7_15TASKS"], Z["N6_15TASKS"], Z["LIMIT90_15TASKS"] = num(G.needed(PAR, r15, 7, K90)), num(G.needed(PAR, r15, 6, K90)), de(G.margin(PAR, r15, 0, K90))
for rule, short in (("rfc", "RFC"), ("first_choice", "FC")):
    fb, fv = fac[rule]
    Z[f"M400_{short}"], Z[f"LIMIT_{short}"] = de(G.margin(PAR, R, 400, K90, fb, fv)), de(G.margin(PAR, R, 0, K90, fb, fv))
    Z[f"N12_{short}"] = num(G.needed(PAR, R, 12, K90, fb, fv))
Z["N12_SOP"] = num(G.needed(PAR, R, 12, K90))
Z["N8_BW"], Z["N8_DR"] = num(G.needed(PAR, R * 1.6, 8, K90)), num(G.needed(PAR, R * 1.3, 8, K90))

# ---- populations estimated from real studies, and the stored simulation with more varied people -------
emp = SIM.load("results_emp", HET)
emp = sim_rows(emp[(emp["market"] == "std") & (emp["rule"] == "share_of_preference")])
emp["z"] = G.zscore(PAR, emp)
Z["EMP_RUNS"], Z["EMP_STUDIES"] = num(int(emp["run_id"].nunique())), int(emp["study"].nunique())
Z["EMP90"], Z["EMP95"] = de(100 * float((emp["z"] <= K90).mean())), de(100 * float((emp["z"] <= K95).mean()))
per_study = (emp["z"] <= K90).groupby(emp["study"]).mean() * 100
Z["EMP_STUDY_MIN"], Z["EMP_STUDY_MAX"] = de(per_study.min(), 0), de(per_study.max(), 0)
# designs of the real studies: the calculator's range note rests on these numbers (2 levels, 3 attributes, 6 levels)
_lev = {}
for _p in (HET / "results_emp").glob("*.json"):
    _r = json.loads(_p.read_text(encoding="utf-8"))
    if "levels" in _r:
        _lev[_r["run_id"]] = tuple(int(x) for x in _r["levels"])
emp["levels"] = emp["run_id"].map(_lev)
assert emp["levels"].notna().all()
emp["l_min"], emp["l_max"] = emp["levels"].map(min), emp["levels"].map(max)
Z["EMP_A_MIN"], Z["EMP_A_MAX"] = int(emp["levels"].map(len).min()), int(emp["levels"].map(len).max())
Z["EMP_L_MIN"], Z["EMP_L_MAX"] = int(emp["l_min"].min()), int(emp["l_max"].max())
two, three, six, nine = emp[emp["l_min"] == 2], emp[emp["levels"].map(len) == 3], emp[emp["l_max"] == 6], emp[emp["levels"].map(len) > 8]
Z["EMP_L2_STUDIES"], Z["EMP_L2_90"] = int(two["study"].nunique()), de(100 * float((two["z"] <= K90).mean()), 0)
Z["EMP_A3_STUDIES"], Z["EMP_A3_90"] = int(three["study"].nunique()), de(100 * float((three["z"] <= K90).mean()), 0)
Z["EMP_L6_STUDIES"], Z["EMP_L6_90"] = int(six["study"].nunique()), de(100 * float((six["z"] <= K90).mean()), 0)
Z["EMP_A9_STUDIES"], Z["EMP_A9_90"] = int(nine["study"].nunique()), de(100 * float((nine["z"] <= K90).mean()), 0)
e7 = emp[emp["attributes"] >= 7]
Z["EMP_ATTR7_90"], Z["EMP_ATTR7_STUDIES"] = de(100 * float((e7["z"] <= K90).mean()), 0), int(e7["study"].nunique())
Z["EMP_REL"] = de(G.rel_rms(PAR, emp), 2)
lead = emp[emp["pop_share"] == emp.groupby(["run_id", "scenario"])["pop_share"].transform("max")]
Z["EMP_LEAD90"] = de(100 * float((lead["z"] <= K90).mean()), 0)
ref = pd.read_pickle(HET / "reference_sop.pkl")
ref = sim_rows(ref[~ref["run_id"].str.contains("_s7|_s1234")])
assert ref["run_id"].nunique() == 1608
ref["z"] = G.zscore(PAR, ref)
Z["OLD_RUNS"] = num(1608)
Z["OLD90"], Z["OLD95"] = de(100 * float((ref["z"] <= K90).mean())), de(100 * float((ref["z"] <= K95).mean()))
Z["OLD_REL"] = de(G.rel_rms(PAR, ref), 2)

# ---- question formats: measured in the stored simulation (more varied people) ----------------------
old_df, _ = N.ORIG_LOAD_RUNS()
old_df = G.prepare(old_df)
old = old_df[(old_df["folder"] == "results") & (old_df["hb_seed"] == 42) & (old_df["rule"] == "share_of_preference")
             & (old_df["market"] == "std") & (~old_df["is_none"])]
for short, method, tf in (("NONE", "cbc_none", 1.0), ("BW", "bwcbc", 1.6), ("DR", "drcbc", 1.3)):
    rows = old[old["method"] == method]
    Z[f"OLD_{short}_90"] = de(cov(rows, K90, tf), 0)
    Z[f"OLD_{short}_RUNS"] = int(rows["run_id"].nunique())

# ---- six sample sizes (needs results_tuned_small) ---------------------------------------------------
six, bias = blog.get("six"), blog.get("bias")
small_file = G.TABLES / "kleine_stichproben_nach_n.csv"
if six and bias and small_file.exists():
    tab = pd.read_csv(small_file).set_index("n")
    assert {50, 100, 250} <= set(tab.index) and all(int(r["runs"]) == 536 for r in six), "small-sample runs incomplete"
    for n in (50, 100, 250):
        Z["COV90_N"][n], Z["COV95_N"][n], Z["RUNS_N"][n] = de(tab.loc[n, "in_90"]), de(tab.loc[n, "in_95"]), 536
    Z["SMALL_MIN90"] = de(min(tab.loc[n, "in_90"] for n in (50, 100, 250)))
    Z["SMALL_MAX90"] = de(max(tab.loc[n, "in_90"] for n in (50, 100, 250)))
    rows = {r["n"]: r for r in six}
    Z["FIG2_TOTAL_50"], Z["FIG2_TOTAL_1000"] = de(K90 * math.sqrt(rows[50]["gesamt"])), de(K90 * math.sqrt(rows[1000]["gesamt"]))
    Z["FIG2_NOISE_SHARE"] = de(100 * np.mean([rows[n]["davon_schaetzung"] / rows[n]["schrumpft"] for n in rows]), 0)
    Z["FIG2_FIXED_LARGER_FROM"] = min(n for n in sorted(rows) if all(rows[m]["bleibt"] > rows[m]["schrumpft"] for m in rows if m >= n))
    Z["WINNER_N"] = {int(n): de(rows[n]["gewinner"], 0) for n in rows}
    Z["WINNER_GAP"] = {gap: {int(n): de(rows[n][f"gewinner|{gap}"], 0) for n in rows} for gap in ("bis 5", "5 bis 10", "über 10")}
    lead_gap = [rows[n]["abstand_mittel"] for n in rows]
    Z["LEAD_TOO_LARGE_MIN"], Z["LEAD_TOO_LARGE_MAX"] = de(min(lead_gap), 0), de(max(lead_gap), 0)
    Z["LEAD_TOO_LARGE"] = (f"about {de(min(lead_gap), 0)}" if de(min(lead_gap), 0) == de(max(lead_gap), 0)
                           else f"{de(min(lead_gap), 0)} to {de(max(lead_gap), 0)}")
    allr = bias["all"]
    by_mid = {int(r["mid"]): r for r in allr}
    Z["BIAS_CROSS"] = de(next(a["mid"] + (0 - a["err"]) * (b["mid"] - a["mid"]) / (b["err"] - a["err"]) for a, b in zip(allr, allr[1:])
                              if a["err"] < 0 <= b["err"]), 0)
    Z["BIAS_AT"] = {m: de(by_mid[m]["err"], 1) for m in by_mid}
    Z["BIAS_10_ABS"] = de(abs(by_mid[10]["err"]), 1)
    Z["BIAS_65_70"] = de((by_mid[65]["err"] + by_mid[70]["err"]) / 2, 0)
    assert by_mid[10]["err"] < 0 < by_mid[65]["err"] and abs(by_mid[65]["err"] - by_mid[70]["err"]) < 1.5
    Z["BIAS_R_THIN"], Z["BIAS_R_RICH"] = de(bias["by_r"][0]["err"], 0), de(bias["by_r"][-1]["err"], 0)
    small = G.load_folder("results_small")
    small = small[(small["rule"] == "share_of_preference") & (small["market"] == "std") & (~small["is_none"])].copy()
    small["z"] = G.zscore(PAR, small)
    small["rank"] = small.groupby(key)["pop_share"].rank(ascending=False, method="first")
    cols = key + ["z", "rank"]
    six_rows = pd.concat([sop[cols], small[cols]], ignore_index=True)
    inside = six_rows["z"] <= K90
    Z["LEAD90_ALL"] = de(100 * float(inside[six_rows["rank"] == 1].mean()), 0)
    Z["OTHER90_ALL"] = de(100 * float(inside[six_rows["rank"] > 1].mean()), 0)
    Z["JOINT90_ALL"] = de(100 * float(inside.groupby([six_rows[c] for c in key]).all().mean()), 0)
    Z["BIAS_SMALLEST_REL"] = de(abs(100 * allr[0]["err"] / allr[0]["true"]), 0)
    Z["BIAS_LEFT_OUT"] = de(100 * bias["left_out"], 0)
    Z["BIAS_BY_R"] = [(x["r"], de(x["err"], 1), x["designs"]) for x in bias["by_r"]]
    Z["BIAS_N_SPREAD"] = de(max(abs(a["err"] - b["err"]) for n in (150, 400, 1000) for a, b in zip(bias[str(n)] if str(n) in bias else bias[n], allr)
                                if a["shares"] >= 200), 2)
else:
    for k in ("SMALL_MIN90", "SMALL_MAX90", "FIG2_TOTAL_50", "FIG2_TOTAL_1000", "FIG2_NOISE_SHARE", "FIG2_FIXED_LARGER_FROM", "WINNER_N",
              "WINNER_GAP", "LEAD_TOO_LARGE_MIN", "LEAD_TOO_LARGE_MAX", "BIAS_CROSS", "BIAS_AT", "BIAS_SMALLEST_REL", "BIAS_LEFT_OUT",
              "BIAS_BY_R", "BIAS_N_SPREAD", "BIAS_10_ABS", "BIAS_65_70", "BIAS_R_THIN", "BIAS_R_RICH", "LEAD90_ALL", "OTHER90_ALL",
              "JOINT90_ALL", "LEAD_TOO_LARGE"):
        Z[k] = None

# ---- test runs on other populations and designs (results_tuned_check and results_tuned_range) -------
chk = G.load_folder("results_check")
rng = G.load_folder("results_range")
assert chk["run_id"].nunique() == 264 and rng["run_id"].nunique() == 960
chk = pd.concat([chk, rng], ignore_index=True)
chk = chk[(chk["rule"] == "share_of_preference") & (chk["market"] == "std") & (~chk["is_none"])].copy()
chk["z"] = G.zscore(PAR, chk)
_lev = {}
for _folder in ("results_tuned_check", "results_tuned_range"):
    for _p in (HET / _folder).glob("*.json"):
        _r = json.loads(_p.read_text(encoding="utf-8"))
        _lev[_r["run_id"]] = tuple(int(x) for x in _r["levels"])
chk["levels"] = chk["run_id"].map(_lev)
assert chk["levels"].notna().all()
med, new = chk[chk["design"] == "med"], chk[chk["design"] != "med"]
assert med["run_id"].nunique() == 120 and new["run_id"].nunique() == 144 + 960
Z["CHECK_PLAN_90"], Z["CHECK_PLAN_95"] = de(100 * float((med["z"] <= K90).mean())), de(100 * float((med["z"] <= K95).mean()))
Z["CHECK_NEW_90"], Z["CHECK_NEW_95"] = de(100 * float((new["z"] <= K90).mean())), de(100 * float((new["z"] <= K95).mean()))
Z["CHECK_NEW_RUNS"], Z["CHECK_NEW_N"] = num(int(new["run_id"].nunique())), int(new["design"].nunique())
_per_design = (new["z"] <= K90).groupby(new["design"]).mean() * 100
Z["CHECK_NEW_DESIGN_MIN"], Z["CHECK_NEW_DESIGN_MAX"] = de(_per_design.min(), 0), de(_per_design.max(), 0)
Z["CHECK_NEW_DESIGNS"] = {d: de(v, 0) for d, v in _per_design.items()}
Z["CHECK_NEW_A_MIN"], Z["CHECK_NEW_A_MAX"] = int(new["levels"].map(len).min()), int(new["levels"].map(len).max())
Z["CHECK_NEW_L_MIN"], Z["CHECK_NEW_L_MAX"] = int(new["levels"].map(min).min()), int(new["levels"].map(max).max())
_many = new[new["levels"].map(len) > 8]
Z["CHECK_NEW_A9_DESIGNS"], Z["CHECK_NEW_A9_90"] = int(_many["design"].nunique()), de(100 * float((_many["z"] <= K90).mean()), 0)
Z["CHECK_NEW_A9_MIN"], Z["CHECK_NEW_A9_MAX"] = int(_many["levels"].map(len).min()), int(_many["levels"].map(len).max())
Z["CHECK_N_MIN"], Z["CHECK_N_MAX"] = num(int(chk["n"].min())), num(int(chk["n"].max()))
Z["RUNS_CHECK"] = num(1608 + 264 + 960 + 432 + 1608)   # small samples, other populations and designs, real studies, earlier simulation
Z["RUNS_TOTAL"] = num(1608 + 1608 + 264 + 960 + 432 + 1608)

(HERE / "zahlen.json").write_text(json.dumps(Z, indent=1, ensure_ascii=False), encoding="utf-8")
lines = [f"{k}: {v}" for k, v in Z.items()]
(HERE / "zahlen.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("\nnoch offen:", [k for k, v in Z.items() if v is None])
