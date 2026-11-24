"""Runs the robust-premium evaluation (soslib.py) on stored runs. Resumable: finished runs are skipped.

Usage: python run_sos.py <workers> [--families wtp,addon,addon2,addon3] [--arms typ,typ_pv1] [--limit-tasks K] [--one RUN_ID]
No estimation is started; every task only reads stored results. Workers run below normal priority on the free cores.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import soslib  # noqa: E402


def option(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main():
    workers = int(sys.argv[1])
    families = option("--families").split(",") if option("--families") else None
    arms = option("--arms").split(",") if option("--arms") else None
    runs = soslib.list_runs(families)
    if arms:
        runs = [r for r in runs if r["arm"] in arms]
    if option("--one"):
        runs = [r for r in runs if r["run_id"] == option("--one")]
    todo = [r for r in runs if not (soslib.RESULTS / f"{r['family']}__{r['run_id']}.json").exists()]
    tasks = {}
    for r in todo:
        tasks.setdefault(r["key"], []).append(r)
    order = sorted(tasks.values(), key=lambda g: -sum(x["n"] for x in g))
    if option("--limit-tasks"):
        order = order[:int(option("--limit-tasks"))]
    log = HERE / "sos_progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    stamp = lambda: time.strftime("%d.%m %H:%M:%S")
    write(f"{stamp()} start: {sum(len(g) for g in order)} runs in {len(order)} populations ({len(runs) - len(todo)} of {len(runs)} runs done before), {workers} workers")
    k = 0
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futs = {ex.submit(soslib.work, {"key": g[0]["key"], "runs": [(x["family"], x["run_id"]) for x in g]}): g for g in order}
        for fut in as_completed(futs):
            k += 1
            g = futs[fut]
            try:
                res = fut.result()
                write(f"{stamp()} [{k}/{len(order)}] {g[0]['arm']} {g[0]['design']} {res['done']}/{len(g)} runs in {res['seconds']}s"
                      + (f" FAILED {res['failed'][:2]}" if res["failed"] else ""))
            except Exception as exc:  # noqa: BLE001
                write(f"{stamp()} [{k}/{len(order)}] worker-exception {type(exc).__name__}: {exc}")
    write(f"{stamp()} done")


if __name__ == "__main__":
    main()
