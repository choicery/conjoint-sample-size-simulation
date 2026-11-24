"""Check runs for the edges of the calculator's range (20.11.2026).

The formula is fixed (FORMEL_18_11.md) and nothing here is used to fit it:
    margin = k * sqrt(b^2 + v / n),  b = 4.3 * r^-0.57,  v = 5000 * r^-0.81,  k = 1.65 / 2.0
The generator runs so far cover 4 to 8 attributes with 3 to 5 levels. Designs with 3 attributes, with two-level
attributes, with six levels and with more than 8 attributes were only tested with populations from real studies
(results_emp/). These runs test them with the tuned generator (tuned_gen.json), the same one as results_tuned/.

Per design: 8 populations, N = 150, 400 and 1,000 (the sample sizes of the main runs), 2 data sets each = 48 runs.
Results -> results_tuned_range/

Usage: python run_tuned_range.py <workers>     (resumable; pinned to P threads 2 to 15 and E cores 20 to 23)
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

# P core 0 (logical 0, 1) and the E cores 16 to 19 and 24 to 31 stay free for the user's own work.
CORES = list(range(2, 16)) + list(range(20, 24))


def pin_to_range_cores():
    try:
        import psutil

        if psutil.cpu_count() == 32:
            psutil.Process().cpu_affinity(CORES)
    except Exception:  # noqa: BLE001
        pass


# emplib.run pins every worker itself; at module level so the spawned workers get the same patch.
emplib.pin_to_efficiency_cores = pin_to_range_cores

EXTRA = {"2": 8, "5": 4}
NS = (1000, 400, 150)
RESULTS = "results_tuned_range"
# (group, levels, tasks, options); r = tasks * (options - 1) / sum(levels - 1), all between 0.375 and 6
DESIGNS = [
    ("three_attributes", [3, 3, 3], 8, 3),                  # r = 2.67
    ("three_attributes", [4, 4, 4], 6, 2),                  # 0.67
    ("three_attributes", [2, 3, 4], 10, 3),                 # 3.33
    ("three_attributes", [3, 4, 5], 12, 3),                 # 2.67
    ("three_attributes", [2, 2, 3], 6, 2),                  # 1.5
    ("three_attributes", [5, 5, 5], 8, 2),                  # 0.67
    ("two_levels", [2, 2, 2, 2], 8, 2),                     # 2.0
    ("two_levels", [2, 2, 2, 2, 2, 2], 8, 3),               # 2.67
    ("two_levels", [2, 3, 2, 4, 2], 10, 3),                 # 2.5
    ("two_levels", [2, 2, 3, 3, 4, 4], 12, 3),              # 2.0
    ("two_levels", [2, 3, 4, 5], 8, 2),                     # 0.8
    ("two_levels", [2, 2, 2, 3, 3, 3, 4, 4], 12, 2),        # 0.86
    ("six_levels", [6, 6, 6, 6], 12, 3),                    # 1.2
    ("six_levels", [3, 4, 6, 6, 2], 10, 4),                 # 1.88
    ("six_levels", [6, 3, 3, 4], 8, 3),                     # 1.33
    ("many_attributes", [3] * 10, 12, 3),                   # 1.2
    ("many_attributes", [2, 2, 2, 2, 3, 3, 3, 4, 4], 16, 3),  # 2.13
    ("many_attributes", [2] * 12, 12, 3),                   # 2.0
    ("many_attributes", [3, 3, 3, 3, 4, 4, 4, 4, 5, 5], 16, 4),  # 1.85
    ("many_attributes", [4] * 9, 10, 3),                    # 0.74
]


def spec(dname, levels, tasks, options, pop_seed, p, n, rep, group, gen):
    return dict(run_id=f"{dname}_cbc_tuned_p{p}_n{n}_r{rep}", design=dname, levels=list(levels), tasks=tasks,
                options=options, method="cbc", n=n, rep=rep, hetero="tuned", hetero_label="tuned", gen=gen,
                none_share=None, hb_seed=42, pop_seed=pop_seed, pop=p, data_seed=pop_seed * 10 + n * 3 + rep * 101 + 7,
                _results=RESULTS, extra_scenarios=EXTRA, stage="range", group=group)


def plan():
    sys.path.insert(0, str(emplib.SIM / "grid"))
    import run_more
    import run_night

    gen = json.loads((HERE / "tuned_gen.json").read_text(encoding="utf-8"))
    base = run_night.base_designs()
    used = set()
    for f in ("specs_tuned.json", "specs_tuned2.json", "specs_tuned_more.json"):
        for s in json.loads((HERE / f).read_text(encoding="utf-8")):
            used.add(s["design"])
    out = []
    for j, (group, levels, tasks, options) in enumerate(DESIGNS):
        r = tasks * (options - 1) / sum(l - 1 for l in levels)
        assert 0.375 <= r <= 6.0, (levels, tasks, options, r)
        dname = run_more.design_name(levels, tasks, options)
        assert dname not in base and dname not in used, dname
        for p in range(8):
            for n in NS:
                for rep in range(2):
                    out.append(spec(dname, levels, tasks, options, 12000 + 13 * j + 100000 * p, p, n, rep, group, gen))
    # one data set of every design and sample size first, large runs first inside a round
    return sorted(out, key=lambda x: (x["rep"], x["pop"], -emplib.cost(x)))


def work(s):
    return emplib.run(s)


def main():
    workers = int(sys.argv[1])
    pin_to_range_cores()
    todo_all = plan()
    assert len({s["run_id"] for s in todo_all}) == len(todo_all) == len({s["data_seed"] for s in todo_all})
    (HERE / "specs_tuned_range.json").write_text(json.dumps(todo_all, indent=1), encoding="utf-8")
    (HERE / RESULTS).mkdir(exist_ok=True)
    todo = [s for s in todo_all if not (HERE / RESULTS / f"{s['run_id']}.json").exists()]
    log = HERE / "range_progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    write(f"{time.strftime('%d.%m %H:%M:%S')} tuned_range start {len(todo)} of {len(todo_all)} runs, {workers} workers, "
          f"cores {CORES}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, s): s for s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{s['group']} {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                      f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{s['group']} {k}/{len(todo)}] {s['run_id']} worker-exception {exc}")
    write(f"{time.strftime('%d.%m %H:%M:%S')} tuned_range done")


if __name__ == "__main__":
    main()
