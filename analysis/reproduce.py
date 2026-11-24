"""Reproduce the formula and the coverage table of the article from the run files in ../data.

    python reproduce.py            prints the fitted formula and the coverage table
    python reproduce.py --check    also compares the output with expected_output.txt

Needs Python 3.10 or newer with numpy, pandas and scipy. Nothing else: the run files already hold the true and
the estimated choice shares of every simulated market, so the Choicery app is not needed for this step.

Every run file is one simulated study: a design, a simulated population, n simulated respondents, a Hierarchical
Bayes estimation and a set of simulated markets. For every product of every market the file holds
    pop        the true choice share in the population
    est        the choice share simulated from the estimated utilities of the n respondents
The error of a share is (est - pop) * 100, in percentage points.

The formula:
    margin = k * sqrt(b^2 + v / n)
    b = 4.3 * r^-0.57      the part of the error that more respondents do not reduce
    v = 5000 * r^-0.81     the part that shrinks with the number of respondents n
    k = 1.65 for the 90% margin, 2.0 for the 95% margin
    r = tasks * (options - 1) / sum(levels - 1)      comparisons per utility
"""
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FORMULA = dict(b=4.3, be=-0.57, v=5000.0, ve=-0.81)   # rounded, fixed before the test runs
K90, K95 = 1.65, 2.0
RULE = "share_of_preference"


def runs(folder):
    """The run files of one folder in ../data, in the order of their names. The folder may also be stored as one ZIP
    file with the same name (data/runs_fit.zip instead of data/runs_fit/); both give the same result."""
    path = DATA / folder
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            yield json.loads(file.read_text(encoding="utf-8"))
    elif (DATA / (folder + ".zip")).is_file():
        with zipfile.ZipFile(DATA / (folder + ".zip")) as archive:
            names = sorted((n for n in archive.namelist() if n.endswith(".json")), key=lambda n: n.rsplit("/", 1)[-1])
            for name in names:
                yield json.loads(archive.read(name).decode("utf-8"))


def load(folder):
    """One row per product of every standard market (three or four products, no None option)."""
    rows = []
    for run in runs(folder):
        if run.get("status") != "ok" or not run.get("shares"):
            continue
        utilities = sum(levels - 1 for levels in run["levels"])
        info = {"run_id": run["run_id"], "design": run["design"], "n": int(run["n"]), "pop": int(run.get("pop", 0)),
                "study": run.get("study", ""), "group": run.get("group", ""), "attributes": len(run["levels"]),
                "ratio": int(run["tasks"]) * (int(run["options"]) - 1) / utilities}
        for share in run["shares"]:
            if share["rule"] != RULE or share.get("is_none"):
                continue
            rows.append({**info, "error": (share["est"] - share["pop"]) * 100})
    return pd.DataFrame(rows)


def predicted_mean_square(par, rows):
    return (par["b"] * rows["ratio"] ** par["be"]) ** 2 + par["v"] * rows["ratio"] ** par["ve"] / rows["n"]


def z(par, rows):
    """Error of a share divided by the error the formula predicts for its design and sample size."""
    return rows["error"].abs() / np.sqrt(predicted_mean_square(par, rows))


def fit(rows):
    """Fit b and v to the mean squared error of every design and sample size (gamma quasi-likelihood)."""
    cells = rows.groupby(["design", "n"]).agg(mse=("error", lambda e: float((e ** 2).mean())), ratio=("ratio", "first")).reset_index()
    x = np.log(cells["ratio"])

    def predicted(q):
        return np.exp(q[0] + q[1] * x) + np.exp(q[2] + q[3] * x) / cells["n"]

    def objective(q):
        return float(np.sum(cells["mse"] / predicted(q) + np.log(predicted(q))))

    options = {"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000, "maxfev": 20000}
    starts = ([2.5, -1.24, 8.0, -1.0], [2.0, -1.0, 8.5, -0.5])
    q = min((minimize(objective, x0=s, method="Nelder-Mead", options=options) for s in starts), key=lambda r: r.fun).x
    return dict(b=math.exp(q[0] / 2), be=q[1] / 2, v=math.exp(q[2]), ve=q[3])


def margin(ratio, n, k):
    return k * math.sqrt((FORMULA["b"] * ratio ** FORMULA["be"]) ** 2 + (FORMULA["v"] * ratio ** FORMULA["ve"] / n if n else 0.0))


def respondents_for(ratio, target, k):
    room = (target / k) ** 2 - (FORMULA["b"] * ratio ** FORMULA["be"]) ** 2
    return None if room <= 0 else math.ceil(FORMULA["v"] * ratio ** FORMULA["ve"] / room - 1e-9)


def main():
    out = []

    def say(text=""):
        out.append(text)
        print(text)

    def row(name, rows):
        zz = z(FORMULA, rows)
        say(f"{name:58s} {rows['run_id'].nunique():6,d} {100 * float((zz <= K90).mean()):8.1f}% {100 * float((zz <= K95).mean()):8.1f}%")

    fit_rows = load("runs_fit")
    exact = fit(fit_rows)
    ze, zr = z(exact, fit_rows), z(FORMULA, fit_rows)
    say(f"Formula fitted to runs_fit ({fit_rows['run_id'].nunique():,} runs, {fit_rows['design'].nunique()} designs)")
    say(f"  fitted:  b = {exact['b']:.3f} * r^{exact['be']:.3f}   v = {exact['v']:.0f} * r^{exact['ve']:.3f}   "
        f"k90 = {float(ze.quantile(0.90)):.3f}   k95 = {float(ze.quantile(0.95)):.3f}")
    say(f"  rounded: b = {FORMULA['b']} * r^{FORMULA['be']}   v = {FORMULA['v']:.0f} * r^{FORMULA['ve']}   k90 = {K90}   k95 = {K95}")
    say(f"  quantiles of the error with the rounded formula: 90% {float(zr.quantile(0.90)):.3f}   95% {float(zr.quantile(0.95)):.3f}")
    r = 10 * 2 / 14
    say(f"  example design (levels 3, 4, 4, 3, 5; 10 tasks; 3 options; r = {r:.2f}): +-8 needs {respondents_for(r, 8, K90)} respondents "
        f"at 90% and {respondents_for(r, 8, K95)} at 95%; limit of the design +-{margin(r, 0, K90):.1f}")
    say()
    say(f"{'Share of choice shares inside the margin of the rounded formula':58s} {'runs':>6s} {'90%':>9s} {'95%':>9s}")
    say("Runs the formula was built from")
    for n, rows in fit_rows.groupby("n"):
        row(f"  {n:,} people", rows)
    say("Test runs the formula has not seen")
    small = load("runs_small_samples")
    for n, rows in small.groupby("n"):
        row(f"  {n:,} people", rows)
    check = load("runs_new_populations_and_designs")
    row("  Example design, 12 new populations", check[check["design"] == "med"])
    new = check[check["design"] != "med"]
    row(f"  {new['design'].nunique()} new designs, 8 populations each", new)
    for design, rows in new.groupby("design"):
        row(f"      {design}", rows)
    real = load("runs_real_study_populations")
    row(f"  Populations estimated from {real['study'].nunique()} real studies", real)
    row("  Earlier simulation with more varied people", load("runs_earlier_simulation"))
    edge = load("runs_range")
    if len(edge):
        say("Test runs at the edges of the range (generator populations)")
        for group, rows in edge.groupby("group"):
            row(f"  {group}, {rows['design'].nunique()} designs", rows)
            for design, sub in rows.groupby("design"):
                row(f"      {design}", sub)
    say()
    say("Real studies, by design (runs_real_study_populations)")
    levels = {}
    for run in runs("runs_real_study_populations"):
        levels[run["run_id"]] = tuple(run["levels"])
    real["levels"] = real["run_id"].map(levels)
    row("  studies with two-level attributes", real[real["levels"].map(min) == 2])
    row("  studies with three attributes", real[real["attributes"] == 3])
    row("  studies with a six-level attribute", real[real["levels"].map(max) == 6])
    row("  studies with more than eight attributes", real[real["attributes"] > 8])
    for study, rows in real.groupby("study"):
        row(f"      {study} (levels {' '.join(str(x) for x in rows['levels'].iloc[0])})", rows)

    expected = HERE / "expected_output.txt"
    if "--write-expected" in sys.argv:
        expected.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    if "--check" in sys.argv:
        same = expected.read_text(encoding="utf-8").splitlines() == out
        print("\nOutput matches expected_output.txt." if same else "\nOutput DIFFERS from expected_output.txt.")
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
