"""Check runs for the formula fitted to the tuned-generator runs (18.11.2026, daytime).

Formula under test, fixed BEFORE these runs from the 1,608 runs in results_tuned/ (see FORMEL_18_11.md):
    margin = k * sqrt(b^2 + v / n)   in percentage points of the true population share
    b = 4.3 * r^-0.57,  v = 5000 * r^-0.81,  k = 1.65 for 90 %, 2.0 for 95 %
    r = tasks * (options - 1) / sum(levels - 1)
Same generator as results_tuned/ (tuned_gen.json). Nothing here was used for the fit.

Stages (in this order):
  plan   example design of the page ('med'), 12 new populations, N = 335 and 1,016 (the planning values of the
         formula for +-8 at 90 % and 95 %), 5 data sets each                       -> results_tuned_check/
  new    the 3 designs outside the 67 of run_check.py, 8 populations each, N = 300 and 700,
         3 data sets each                                                           -> results_tuned_check/
  small  all 67 designs with 50, 100 and 250 respondents, populations 0 to 3 (the populations and markets of
         results_tuned/, new people), 2 data sets each                              -> results_tuned_small/

Usage: python run_tuned_more.py <workers>     (resumable; pinned to 8 efficiency cores, logical 24 to 31)
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
import emplib  # noqa: E402

CORES = list(range(24, 32))


def pin_to_eight_efficiency_cores():
    try:
        import psutil

        if psutil.cpu_count() == 32:
            psutil.Process().cpu_affinity(CORES)
    except Exception:  # noqa: BLE001
        pass


# emplib.run pins every worker itself; at module level so the spawned workers get the same patch.
emplib.pin_to_efficiency_cores = pin_to_eight_efficiency_cores

EXTRA = {"2": 8, "5": 4}
PLAN_NS = (335, 1016)
NEW_DESIGNS = [([5, 4, 4, 3, 5, 4], 8, 3), ([3, 4, 3, 5], 11, 3), ([3, 3, 4, 3], 13, 4)]  # r = 0.84, 2.0, 4.33
NEW_NS = (300, 700)
SMALL_NS = (250, 100, 50)


def spec(dname, levels, tasks, options, pop_seed, p, n, rep, stage, results, gen):
    return dict(run_id=f"{dname}_cbc_tuned_p{p}_n{n}_r{rep}", design=dname, levels=list(levels), tasks=tasks,
                options=options, method="cbc", n=n, rep=rep, hetero="tuned", hetero_label="tuned", gen=gen,
                none_share=None, hb_seed=42, pop_seed=pop_seed, pop=p, data_seed=pop_seed * 10 + n * 3 + rep * 101 + 7,
                _results=results, extra_scenarios=EXTRA, stage=stage)


def plan():
    sys.path.insert(0, str(emplib.SIM / "grid"))
    import run_more
    import run_night

    gen = json.loads((HERE / "tuned_gen.json").read_text(encoding="utf-8"))
    base = run_night.base_designs()
    out = []
    s = base["med"]
    for p in range(10, 22):
        for n in PLAN_NS:
            for rep in range(5):
                out.append(spec("med", s["levels"], s["tasks"], s["options"], s["pop_seed"] + 100000 * p, p, n, rep, "plan",
                                "results_tuned_check", gen))
    for j, (levels, tasks, options) in enumerate(NEW_DESIGNS):
        dname = run_more.design_name(levels, tasks, options)
        assert dname not in base, dname
        for p in range(8):
            for n in NEW_NS:
                for rep in range(3):
                    out.append(spec(dname, levels, tasks, options, 9000 + 13 * j + 100000 * p, p, n, rep, "new",
                                    "results_tuned_check", gen))
    small = []
    for dname, s in base.items():
        for p in range(4):
            for n in SMALL_NS:
                for rep in range(2):
                    small.append(spec(dname, s["levels"], s["tasks"], s["options"], s["pop_seed"] + 100000 * p, p, n, rep,
                                      "small", "results_tuned_small", gen))
    # one data set of every design first, so an early look already covers all designs
    return out + sorted(small, key=lambda x: (x["rep"], x["pop"], -emplib.cost(x)))


def work(s):
    return emplib.run(s)


def main():
    workers = int(sys.argv[1])
    pin_to_eight_efficiency_cores()
    todo_all = plan()
    assert len({(s["_results"], s["run_id"]) for s in todo_all}) == len(todo_all) == len({s["data_seed"] for s in todo_all})
    (HERE / "specs_tuned_more.json").write_text(json.dumps(todo_all, indent=1), encoding="utf-8")
    todo = []
    for s in todo_all:
        (HERE / s["_results"]).mkdir(exist_ok=True)
        if not (HERE / s["_results"] / f"{s['run_id']}.json").exists():
            todo.append(s)
    log = HERE / "sim_progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    write(f"{time.strftime('%d.%m %H:%M:%S')} tuned_more start {len(todo)} of {len(todo_all)} runs, {workers} workers, cores {CORES[0]} to {CORES[-1]}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for s in todo:  # submitted in plan order: plan, new, small
            futs[ex.submit(work, s)] = s
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{s['stage']} {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                      f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{s['stage']} {k}/{len(todo)}] {s['run_id']} worker-exception {exc}")
    write(f"{time.strftime('%d.%m %H:%M:%S')} tuned_more done")


if __name__ == "__main__":
    main()
