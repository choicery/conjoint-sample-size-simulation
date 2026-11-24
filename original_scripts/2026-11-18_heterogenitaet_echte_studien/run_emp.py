"""Night runs 18.11.2026: does the sample-size formula hold for populations taken from real studies?

Stages:
  emp    one population per real market-research study (largest arm of every study family), two designs each
         (the study's own tasks and options, and a contrast design far away in comparisons per utility),
         two sets of markets, 150 / 400 / 1,000 respondents, two data sets                 -> results_emp/
  tuned  the 67 designs of the stored simulation with the generator re-tuned to the real studies
         (numbers in tuned_gen.json), populations 0 and 1                                  -> results_tuned/

Usage: python run_emp.py emp|tuned <workers> <deadline HH:MM>     (resumable; no run starts after the deadline)
"""
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for k in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emplib  # noqa: E402
import summarize_real as SR  # noqa: E402

NS = (1000, 400, 150)
EXTRA = {"2": 8, "5": 4}
CONTRAST = [(6, 2), (6, 3), (8, 3), (12, 3), (16, 4)]
MIN_N_STUDY = 100
DEADLINE = [None]


def emp_studies():
    """Largest arm of every market-research study family with a finished fit."""
    best = {}
    for f in sorted(emplib.FITS.glob("*.json")):
        if "__boot" in f.name:
            continue
        fit = json.loads(f.read_text(encoding="utf-8"))
        sid = fit.get("study")
        if fit.get("status") != "ok" or SR.group_of(sid) != "market" or fit["n"] < MIN_N_STUDY:
            continue
        fam = SR.family_of(sid)
        if fam not in best or fit["n"] > best[fam]["n"]:
            best[fam] = fit
    return best


def emp_plan():
    out = []
    scale = json.loads((HERE / "emp_scale.json").read_text(encoding="utf-8")) if (HERE / "emp_scale.json").exists() else {}
    for j, (fam, fit) in enumerate(sorted(emp_studies().items())):
        sid = fit["study"]
        meta = json.loads((emplib.HERE / "prepared" / sid / "meta.json").read_text(encoding="utf-8"))
        levels = list(fit["levels"])
        k = sum(l - 1 for l in levels)
        own = (int(min(max(meta["tasks_median"], 4), 20)), int(min(max(meta["options_median"], 2), 5)))
        r_own = own[0] * (own[1] - 1) / k
        cands = [c for c in CONTRAST if 0.3 <= c[0] * (c[1] - 1) / k <= 7 and c != own]
        contrast = max(cands, key=lambda c: abs(math.log(c[0] * (c[1] - 1) / k) - math.log(r_own)))
        short = re.sub(r"[^a-z0-9]+", "", fam.lower())[:24]
        for which, (tasks, options) in (("own", own), ("contrast", contrast)):
            dname = f"{short}_t{tasks}o{options}"
            for p in range(2):
                pop_seed = 9000 + 37 * j + (500 if which == "contrast" else 0) + 100000 * p
                for n in NS:
                    for rep in range(2):
                        out.append(dict(run_id=f"{dname}_emp_p{p}_n{n}_r{rep}", design=dname, levels=levels, tasks=tasks,
                                        options=options, method="cbc", n=n, rep=rep, hetero="empirical",
                                        hetero_label="empirical", study=sid, family=fam, which=which, none_share=None,
                                        hb_seed=42, pop_seed=pop_seed, pop=p, data_seed=pop_seed * 10 + n * 3 + rep * 101,
                                        sd_scale=float(scale.get("sd_scale", 1.0)), mean_scale=float(scale.get("mean_scale", 1.0)),
                                        _results="results_emp", extra_scenarios=EXTRA, stage="emp"))
    return sorted(out, key=lambda s: (s["rep"], s["pop"], -emplib.cost(s)))


def tuned_plan(pops=(0, 1)):
    sys.path.insert(0, str(emplib.SIM / "grid"))
    import run_night

    gen = json.loads((HERE / "tuned_gen.json").read_text(encoding="utf-8"))
    out = []
    for dname, s in run_night.base_designs().items():
        for p in pops:
            pop_seed = s["pop_seed"] + 100000 * p
            for n in NS:
                for rep in range(2):
                    out.append(dict(run_id=f"{dname}_cbc_tuned_p{p}_n{n}_r{rep}", design=dname, levels=list(s["levels"]),
                                    tasks=s["tasks"], options=s["options"], method="cbc", n=n, rep=rep, hetero="tuned",
                                    hetero_label="tuned", gen=gen, none_share=None, hb_seed=42, pop_seed=pop_seed, pop=p,
                                    data_seed=pop_seed * 10 + n * 3 + rep * 101 + 7, _results="results_tuned",
                                    extra_scenarios=EXTRA, stage="tuned"))
    return sorted(out, key=lambda s: (s["pop"], s["rep"], -emplib.cost(s)))


def work(args):
    spec, deadline = args
    if deadline and time.time() > deadline:
        return {"status": "skipped_deadline"}
    return emplib.run(spec)


def parse_deadline(text):
    hh, mm = (int(x) for x in text.split(":"))
    now = time.localtime()
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
    return t + 86400 if t < time.time() else t


def main():
    stage, workers = sys.argv[1], int(sys.argv[2])
    deadline = parse_deadline(sys.argv[3]) if len(sys.argv) > 3 else None
    emplib.pin_to_efficiency_cores()
    todo_all = emp_plan() if stage == "emp" else tuned_plan((2, 3)) if stage == "tuned2" else tuned_plan()
    assert len({s["run_id"] for s in todo_all}) == len(todo_all) == len({s["data_seed"] for s in todo_all})
    (HERE / f"specs_{stage}.json").write_text(json.dumps(todo_all, indent=1), encoding="utf-8")
    res = HERE / todo_all[0]["_results"]
    res.mkdir(exist_ok=True)
    todo = [s for s in todo_all if not (res / f"{s['run_id']}.json").exists()]
    log = HERE / "sim_progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    write(f"{time.strftime('%d.%m %H:%M:%S')} {stage} start {len(todo)} of {len(todo_all)} runs, {workers} workers, "
          f"deadline {sys.argv[3] if deadline else 'none'}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, (s, deadline)): s for s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                if rec.get("status") == "skipped_deadline":
                    continue
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{stage} {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                      f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{stage} {k}/{len(todo)}] {s['run_id']} worker-exception {exc}")
    write(f"{time.strftime('%d.%m %H:%M:%S')} {stage} done")


if __name__ == "__main__":
    main()
