"""Full grid: 2 single-threaded workers. Resumable: runs with an existing result file are skipped."""
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

# (attributes, levels per attribute, tasks, options); ratio = T*(O-1)/(A*(L-1))
POINTS = [
    (4, 3, 14, 4), (4, 3, 10, 3), (4, 4, 10, 4), (4, 5, 6, 2), (4, 4, 6, 3), (4, 5, 14, 3),
    (6, 3, 14, 3), (6, 3, 10, 4), (6, 4, 10, 3), (6, 4, 14, 4), (6, 5, 6, 3), (6, 5, 10, 2), (6, 3, 6, 2),
    (8, 3, 10, 3), (8, 4, 14, 4), (8, 5, 6, 3), (8, 5, 14, 4), (8, 4, 10, 2),
]
MEDIUM = ("med", [3, 4, 4, 3, 5], 10, 3)
NS = (1000, 400, 150)
# Subsets for heterogeneity, None/DR/BW and seed tests (indices into POINTS + medium).
HIGH_POINTS = [1, 4, 8, 10, 13, 15]
VARIANT_POINTS = [1, 8, 10, 13]


def name(p):
    a, l, t, o = p
    return f"a{a}l{l}t{t}o{o}"


def designs():
    out = [(name(p), [p[1]] * p[0], p[2], p[3]) for p in POINTS]
    out.append(MEDIUM)
    return out


def build_grid():
    ds = designs()
    seeds = {d[0]: 7000 + 13 * i for i, d in enumerate(ds)}
    grid = []

    def add(d, method, n, rep, hetero, none_share=None, hb_seed=42, tag=""):
        dname, levels, tasks, options = d
        pop_seed = seeds[dname] + (500 if hetero == "high" else 0)
        run_id = f"{dname}_{method}{tag}_{hetero}_n{n}_r{rep}" + (f"_s{hb_seed}" if hb_seed != 42 else "")
        grid.append(dict(run_id=run_id, design=dname, levels=levels, tasks=tasks, options=options, method=method,
                         n=n, rep=rep, hetero=hetero if hetero == "realistic" else True, hetero_label=hetero,
                         none_share=none_share, hb_seed=hb_seed, pop_seed=pop_seed,
                         data_seed=pop_seed * 10 + n * 3 + rep * 101 + (17 if method != "cbc" else 0)
                         + int(100 * (none_share or 0))))

    for d in ds:
        for n in NS:
            for rep in range(2):
                add(d, "cbc", n, rep, "realistic")
    for i in HIGH_POINTS + [len(ds) - 1]:
        for rep in range(2):
            add(ds[i], "cbc", 400, rep, "high")
    for i in VARIANT_POINTS:
        for rep in range(2):
            add(ds[i], "cbc_none", 400, rep, "realistic", none_share=0.10, tag="10")
            add(ds[i], "cbc_none", 400, rep, "realistic", none_share=0.30, tag="30")
            add(ds[i], "drcbc", 400, rep, "realistic", none_share=0.30)
            add(ds[i], "bwcbc", 400, rep, "realistic")
    # Seed test: same data (rep 0) re-estimated with two other seeds.
    for i in (1, 10, 13):
        for s in (7, 1234):
            add(ds[i], "cbc", 400, 0, "realistic", hb_seed=s)
    return grid


def cost(s):
    return s["n"] * s["tasks"] * sum(l - 1 for l in s["levels"])


def main():
    grid = build_grid()
    (GRID / "grid_specs.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    done = {p.stem for p in (GRID / "results").glob("*.json")} if (GRID / "results").exists() else set()
    todo = [s for s in grid if s["run_id"] not in done]
    todo.sort(key=lambda s: -cost(s))  # longest first
    log = GRID / "grid_progress.log"
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"{time.strftime('%d.%m %H:%M:%S')} start {len(todo)} of {len(grid)} runs\n")
    with ProcessPoolExecutor(max_workers=int(os.environ.get("GRID_WORKERS", "2"))) as ex:
        futs = {ex.submit(gridlib.run, s): s for s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                line = (f"{time.strftime('%d.%m %H:%M:%S')} [{k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                        f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s corr={rec.get('indiv_corr')} "
                        f"err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                line = f"{time.strftime('%d.%m %H:%M:%S')} [{k}/{len(todo)}] {s['run_id']} worker-exception {exc}"
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(line + "\n")
            print(line, flush=True)


if __name__ == "__main__":
    main()
