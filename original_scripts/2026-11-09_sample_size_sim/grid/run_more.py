"""Checks and extra designs for figure 1 of the sample size page (16.11.2026).

Modes (first argument):
  repro  re-run two stored runs unchanged -> results_repro/   (is the pipeline still the same?)
  long   same data as the stored N=400 runs, five times the burn-in and draws -> results_long/
         (is the estimation error a matter of chain length?)
  more   36 new CBC designs inside the simulated range, same settings as the grid -> results_more/

Nothing here writes to results/ (the 238 runs behind the published figure stay untouched).
Resumable: runs with an existing result file are skipped.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for k in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

GRID = Path(__file__).resolve().parent
sys.path.insert(0, str(GRID))

NS = (1000, 400, 150)
LONG_FACTOR = 5
LONG_DESIGNS = ["a8l5t6o3", "a6l5t6o3", "a6l4t8o3", "med", "a4l3t10o3", "a4l3t14o4"]
REPRO_RUNS = ["med_cbc_realistic_n400_r0", "a6l5t6o3_cbc_realistic_n150_r1"]

# New designs: (levels per attribute, tasks, options). All inside the range stated on the page:
# 4 to 8 attributes, 3 to 5 levels, 6 to 16 tasks, 2 to 4 options, 0.375 to 6 comparisons per utility.
MORE_DESIGNS = [
    # mixed numbers of levels, as in real studies
    ([3, 3, 4, 4, 5], 8, 3), ([3, 4, 5, 3, 4, 3], 12, 3), ([4, 4, 3, 3, 5, 3, 4], 12, 4), ([5, 4, 3, 3], 8, 3),
    ([3, 3, 3, 4, 4, 5, 5, 4], 12, 3), ([5, 5, 4, 4, 3, 3], 10, 2), ([4, 3, 5, 4], 16, 3), ([3, 4, 3, 4, 3], 15, 4),
    ([5, 3, 4, 5, 3, 4, 5], 9, 3), ([3, 3, 4, 5], 12, 2), ([4, 5, 3, 4, 5, 3], 8, 4), ([3, 5, 4, 3, 3], 16, 4),
    # equal numbers of levels, spread over comparisons per utility
    ([5] * 5, 8, 2), ([5] * 7, 8, 3), ([5] * 5, 10, 3), ([3] * 7, 9, 3), ([4] * 5, 12, 3), ([4] * 7, 12, 4),
    ([5] * 5, 14, 4), ([3] * 7, 14, 3), ([3] * 8, 16, 4), ([3] * 6, 16, 4), ([3] * 5, 16, 4), ([3] * 4, 15, 4),
    ([4] * 5, 8, 2), ([5] * 7, 12, 2), ([3] * 8, 12, 2), ([5] * 4, 10, 4), ([5] * 6, 12, 4), ([4] * 7, 10, 3),
    ([3] * 5, 6, 3), ([3] * 6, 8, 4), ([4] * 8, 16, 3), ([5] * 8, 16, 4), ([5] * 7, 16, 4), ([4] * 4, 8, 2),
]
MORE_SEED_OFFSET = 40  # population seeds continue after the grid (0..18) and the fill (19..30)


def design_name(levels, tasks, options):
    if len(set(levels)) == 1:
        return f"a{len(levels)}l{levels[0]}t{tasks}o{options}"
    return "m" + "".join(str(l) for l in levels) + f"t{tasks}o{options}"


def stored_specs():
    specs = json.loads((GRID / "grid_specs.json").read_text(encoding="utf-8"))
    specs += json.loads((GRID / "grid_specs_fill.json").read_text(encoding="utf-8"))
    return {s["run_id"]: s for s in specs}


def build(mode):
    if mode == "repro":
        specs = stored_specs()
        return [dict(specs[r], _results="results_repro") for r in REPRO_RUNS]
    if mode == "long":
        out = []
        for s in stored_specs().values():
            if (s["design"] in LONG_DESIGNS and s["method"] == "cbc" and s["hetero_label"] == "realistic"
                    and s["n"] == 400 and s.get("hb_seed", 42) == 42):
                out.append(dict(s, _results="results_long", n_burn=10000 * LONG_FACTOR, n_draws=10000 * LONG_FACTOR))
        return out
    if mode == "more":
        out = []
        for j, (levels, tasks, options) in enumerate(MORE_DESIGNS):
            dname = design_name(levels, tasks, options)
            pop_seed = 7000 + 13 * (MORE_SEED_OFFSET + j)
            for n in NS:
                for rep in range(2):
                    out.append(dict(run_id=f"{dname}_cbc_realistic_n{n}_r{rep}", design=dname, levels=list(levels),
                                    tasks=tasks, options=options, method="cbc", n=n, rep=rep, hetero="realistic",
                                    hetero_label="realistic", none_share=None, hb_seed=42, pop_seed=pop_seed,
                                    data_seed=pop_seed * 10 + n * 3 + rep * 101, _results="results_more",
                                    extra_scenarios={"2": 8, "5": 4}))
        return out
    raise SystemExit("mode: repro | long | more")


def pin_to_efficiency_cores():
    """Leave the 8 performance cores (logical 0 to 15) to other work; child processes (Julia) inherit this."""
    try:
        import psutil

        if psutil.cpu_count() == 32:
            psutil.Process().cpu_affinity(list(range(16, 32)))
    except Exception:  # noqa: BLE001
        pass


def run(spec):
    """Same as gridlib.run, with its own result folder and optional chain length."""
    import shutil

    import gridlib
    import simlib

    pin_to_efficiency_cores()

    simlib.make_population = gridlib.make_population
    simlib.RESULTS = GRID / spec["_results"]
    simlib.WORK = GRID / ("work_" + spec["_results"])
    simlib.DESIGNS[spec["design"]] = dict(levels=list(spec["levels"]), tasks=int(spec["tasks"]),
                                          options=int(spec["options"]))
    simlib.APP_HB["seed"] = int(spec.get("hb_seed", 42))
    simlib.APP_HB["n_burn"] = int(spec.get("n_burn", 10000))
    simlib.APP_HB["n_draws"] = int(spec.get("n_draws", 10000))
    simlib.APP_HB["thinning"] = int(spec.get("thinning", 5))
    rec = simlib.run_one(spec)
    if rec.get("status") == "ok":
        shutil.rmtree(simlib.WORK / spec["run_id"], ignore_errors=True)
    return rec


def cost(s):
    return s["n"] * s["tasks"] * sum(l - 1 for l in s["levels"]) * s.get("n_burn", 10000)


def main():
    mode = sys.argv[1]
    grid = build(mode)
    if mode == "more":
        (GRID / "grid_specs_more.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    res_dir = GRID / grid[0]["_results"]
    res_dir.mkdir(exist_ok=True)
    done = {p.stem for p in res_dir.glob("*.json")}
    todo = sorted((s for s in grid if s["run_id"] not in done), key=lambda s: -cost(s))
    log = GRID / "more_progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    write(f"{time.strftime('%d.%m %H:%M:%S')} {mode} start {len(todo)} of {len(grid)} runs")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("GRID_WORKERS", "8"))) as ex:
        futs = {ex.submit(run, s): s for s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{mode} {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                      f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s "
                      f"err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{mode} {k}/{len(todo)}] {s['run_id']} worker-exception {exc}")
    write(f"{time.strftime('%d.%m %H:%M:%S')} {mode} done")


if __name__ == "__main__":
    main()
