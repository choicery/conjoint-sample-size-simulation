"""Question formats with the tuned generator (19.11.2026): DR-CBC, BW-CBC and BW-CBC with None.

Why: the task factors of the calculator (BW-CBC 1.6, DR-CBC 1.3) come from 4 designs of the old simulation at N = 400.
Here: 20 designs from thin to rich (3 or 4 options per task), the populations 0, 1 and 2 and the markets of the plain CBC
runs in ../2026-11-18_heterogenitaet_echte_studien/results_tuned/, the SAME respondents and task designs as those runs
(same data seed), 150 / 400 / 1,000 respondents, two data sets each, three formats. DR-CBC also on three designs with
2 options per task (BW-CBC needs at least 3). None formats: None utility calibrated to about 30 % None in random tasks.

Order: population 0 data set 0 first, then p1 r0, p0 r1, p1 r1, and population 2 last (r0, r1), so that a stop at any time
leaves a balanced set.

Usage: python run_formate.py <workers> [smoke]      (resumable; pinned to six efficiency cores, logical 26 to 31)
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for k in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fmtlib  # noqa: E402

DESIGNS = ["a8l5t6o3", "a6l5t6o3", "a8l5t10o3", "m5345345t9o3", "a6l4t8o3", "a5l5t10o3", "a6l4t10o3", "a8l3t10o3", "m453453t8o4", "med",
           "a6l5t12o4", "a5l4t12o3", "a4l5t10o4", "a7l3t14o3", "a6l4t14o4", "a4l3t10o3", "a4l3t12o3", "a4l4t14o4", "a5l3t14o4",
           "a4l3t16o4"]
DESIGNS_O2 = ["a5l5t8o2", "a4l4t8o2", "m3345t12o2"]      # DR-CBC only
FORMATS = ("drcbc", "bwcbc", "bwcbc_none")
NS = (1000, 400, 150)
EXTRA = {"2": 8, "5": 4}
NONE_SHARE = 0.30


def base_designs():
    import emplib

    sys.path.insert(0, str(emplib.SIM / "grid"))
    import run_night

    return run_night.base_designs()


def plan():
    gen = json.loads((fmtlib.HET / "tuned_gen.json").read_text(encoding="utf-8"))
    base = base_designs()
    out = []
    for dname in DESIGNS + DESIGNS_O2:
        s = base[dname]
        for fmt in FORMATS:
            if s["options"] < 3 and fmt != "drcbc":
                continue
            for p in (0, 1, 2):
                pop_seed = s["pop_seed"] + 100000 * p
                for n in NS:
                    for rep in range(2):
                        out.append(dict(
                            run_id=f"{dname}_{fmt}_tuned_p{p}_n{n}_r{rep}", design=dname, levels=list(s["levels"]), tasks=s["tasks"],
                            options=s["options"], method=fmt, n=n, rep=rep, hetero="tuned", hetero_label="tuned", gen=gen,
                            none_share=(None if fmt == "bwcbc" else NONE_SHARE), hb_seed=42, pop_seed=pop_seed, pop=p,
                            # the data seed of the plain CBC run of this cell in results_tuned/: same people, same task design
                            data_seed=pop_seed * 10 + n * 3 + rep * 101 + 7, cbc_reference=f"{dname}_cbc_tuned_p{p}_n{n}_r{rep}",
                            _results="results_formate", extra_scenarios=EXTRA, stage="formate"))
    return sorted(out, key=lambda x: (x["pop"] == 2, x["rep"], x["pop"], -fmtlib.cost(x)))


def smoke_plan():
    gen = json.loads((fmtlib.HET / "tuned_gen.json").read_text(encoding="utf-8"))
    s = base_designs()["a4l3t10o3"]
    return [dict(run_id=f"smoke_{fmt}_n{n}", design="a4l3t10o3", levels=list(s["levels"]), tasks=s["tasks"], options=s["options"],
                 method=fmt, n=n, rep=0, hetero="tuned", hetero_label="tuned", gen=gen, none_share=(None if fmt == "bwcbc" else NONE_SHARE),
                 hb_seed=42, pop_seed=s["pop_seed"], pop=0, data_seed=s["pop_seed"] * 10 + n * 3 + 7, _results="results_smoke",
                 extra_scenarios=EXTRA, stage="smoke") for fmt in FORMATS for n in (150, 1000)]


def main():
    workers = int(sys.argv[1])
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    fmtlib.pin()
    todo_all = smoke_plan() if smoke else plan()
    assert len({s["run_id"] for s in todo_all}) == len(todo_all)
    if not smoke:
        (HERE / "specs_formate.json").write_text(json.dumps(todo_all, indent=1), encoding="utf-8")
    res = HERE / todo_all[0]["_results"]
    res.mkdir(exist_ok=True)
    todo = [s for s in todo_all if not (res / f"{s['run_id']}.json").exists()]
    log = HERE / "progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    stage = "smoke" if smoke else "formate"
    write(f"{time.strftime('%d.%m %H:%M:%S')} {stage} start {len(todo)} of {len(todo_all)} runs, {workers} workers, cores "
          f"{fmtlib.CORES[0]} to {fmtlib.CORES[-1]}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for s in todo:   # submitted in plan order
            futs[ex.submit(fmtlib.run, s)] = s
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{stage} {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                      f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s corr={rec.get('indiv_corr')} "
                      f"nobuy={rec.get('observed_no_buy_share')} err={json.dumps(rec.get('error'))[:300]}")
            except Exception as exc:  # noqa: BLE001
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{stage} {k}/{len(todo)}] {s['run_id']} worker-exception {exc}")
    write(f"{time.strftime('%d.%m %H:%M:%S')} {stage} done")


if __name__ == "__main__":
    main()
