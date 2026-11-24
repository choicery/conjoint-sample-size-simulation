"""Fill the gaps in comparisons per utility (0.5 to 1 and 3 to 6) with 12 more CBC designs.

Same settings as the main grid: realistic heterogeneity, share of preference / RFC / first choice,
N = 150, 400, 1000, two data replications. Resumable: runs with a result file are skipped.
Specs go to grid_specs_fill.json so the main grid_specs.json stays untouched.
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
import gridlib  # noqa: E402
import run_grid  # noqa: E402

# (attributes, levels, tasks, options); ratio = T*(O-1)/(A*(L-1))
FILL_POINTS = [
    (6, 5, 7, 3), (8, 5, 10, 3), (8, 4, 8, 3), (7, 4, 8, 3), (5, 4, 6, 3), (6, 4, 8, 3),   # 0.58 to 0.89
    (4, 3, 12, 3), (4, 4, 14, 4), (5, 3, 12, 4), (5, 3, 14, 4), (4, 3, 12, 4), (4, 3, 16, 4),  # 3.0 to 6.0
]
SEED_OFFSET = 19  # continue the main grid's population seeds (7000 + 13 * i)


def build_fill():
    grid = []
    for j, p in enumerate(FILL_POINTS):
        dname = run_grid.name(p)
        pop_seed = 7000 + 13 * (SEED_OFFSET + j)
        for n in run_grid.NS:
            for rep in range(2):
                grid.append(dict(run_id=f"{dname}_cbc_realistic_n{n}_r{rep}", design=dname, levels=[p[1]] * p[0],
                                 tasks=p[2], options=p[3], method="cbc", n=n, rep=rep, hetero="realistic",
                                 hetero_label="realistic", none_share=None, hb_seed=42, pop_seed=pop_seed,
                                 data_seed=pop_seed * 10 + n * 3 + rep * 101))
    return grid


def main():
    grid = build_fill()
    (GRID / "grid_specs_fill.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    done = {p.stem for p in (GRID / "results").glob("*.json")}
    todo = sorted((s for s in grid if s["run_id"] not in done), key=lambda s: -run_grid.cost(s))
    log = GRID / "grid_progress.log"
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"{time.strftime('%d.%m %H:%M:%S')} fill start {len(todo)} of {len(grid)} runs\n")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("GRID_WORKERS", "8"))) as ex:
        futs = {ex.submit(gridlib.run, s): s for s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                line = (f"{time.strftime('%d.%m %H:%M:%S')} [fill {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                        f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s "
                        f"err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                line = f"{time.strftime('%d.%m %H:%M:%S')} [fill {k}/{len(todo)}] {s['run_id']} worker-exception {exc}"
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(line + "\n")
            print(line, flush=True)
    print("fill done", flush=True)


if __name__ == "__main__":
    main()
