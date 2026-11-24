"""Runs one block of the night runs (see nightlib.py). Resumable: finished runs are skipped.

Usage: python run_night.py <block> <workers> [--smoke] [--list] [--after <block>] [--min-free-gb G] [--stop-free-gb G] [--dry]
  block           chains | designpath | addon | a json file with a list of specs (key "_work": "shares" or "addon")
  --smoke         a few small runs into results_smoke_<block>
  --after x       waits until the log of block x ends with "x done"; gives up when x was stopped by the memory guard
  --min-free-gb   a run starts only while at least this much RAM is available (default 14)
  --stop-free-gb  below this much available RAM the block stops itself and its workers (default 5); start it again later
  --dry           eight runs that only sleep: tests the queue without any estimation

Memory guard (added 21.11.2026, 22:45). The first start (12 workers, 21:17) ended at 22:20 when the machine ran out of
memory; the main run of another project stopped at the same moment. Since then: a run starts only with enough free
RAM, every run gets a fresh worker process, the peak memory of a run (worker and its Julia process) is logged, and the
block stops itself before memory gets short. These runs are the ones that give way.
"""
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import psutil

for k in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nightlib  # noqa: E402

RESERVE_GB = 3.0     # assumed peak of one run until the first runs have been measured
YOUNG_SECONDS = 240  # a run that started less than this ago has not reached its peak yet
STOP_NOTE = "stopped by the memory guard"


def free_gb():
    return psutil.virtual_memory().available / 2 ** 30


def tree_rss():
    me, total = psutil.Process(), 0
    for p in [me] + me.children(recursive=True):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def work(spec):
    peak, stop = [0], threading.Event()

    def sample():
        while not stop.wait(5):
            peak[0] = max(peak[0], tree_rss())

    threading.Thread(target=sample, daemon=True).start()
    try:
        if spec.get("_dry"):
            time.sleep(spec["_dry"])
            rec = {"status": "dry"}
        else:
            rec = (nightlib.work_addon if spec.get("_work") == "addon" else nightlib.work_shares)(spec)
    finally:
        stop.set()
    rec = dict(rec)
    rec["peak_tree_gb"] = round(max(peak[0], tree_rss()) / 2 ** 30, 2)
    return rec


def smoke_specs(block, todo):
    if block == "chains":
        base = next(s for s in todo if s["design"] == "med" and s["spread"] == 0.55 and s["pop"] == 0 and s["variant"] == "d4")
        out = []
        for v in ("d4", "d4lo", "m1"):
            out.append({**base, "sigma_start_factor": None, **nightlib.VARIANTS[v], "variant": v, "n": 150, "run_id": f"smoke_{v}"})
        out.append({**base, "sigma_start_factor": None, "prior_variance": 4.0, "n_burn": 2000, "n_draws": 5000, "thinning": 50,
                    "variant": "mini_x", "n": 150, "run_id": "smoke_mini_x"})
        out.append({**base, "sigma_start_factor": None, **nightlib.VARIANTS["d4"], "variant": "d4", "n": 150,
                    "run_id": "smoke_d4_again"})
    elif block == "designpath":
        base = [s for s in todo if s["design"] == "med" and s["pop"] == 0 and s["n"] == 400]
        out = [dict(s, n=150, run_id="smoke_" + s["variant"]) for s in base]
    else:
        out = [dict(s, run_id="smoke_" + s["run_id"]) for s in todo if s["n"] == 150 and s["pop"] == 0 and s["rep"] == 0]
    return [dict(s, _results="results_smoke_" + block) for s in out]


def option(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    block, workers = sys.argv[1], int(sys.argv[2])
    dry = "--dry" in sys.argv
    if dry:
        block, todo_all = "dry", [dict(run_id=f"dry_{i}", _dry=4, _results="results_dry") for i in range(8)]
    elif block.endswith(".json"):
        todo_all = json.loads(Path(block).read_text(encoding="utf-8"))
        block = Path(block).stem
    else:
        todo_all = [dict(s, _work="addon" if block == "addon" else "shares") for s in nightlib.PLANS[block][0]()]
    assert len({s["run_id"] for s in todo_all}) == len(todo_all)
    if "--list" in sys.argv:
        hours = sum(nightlib.cost(s) for s in todo_all)
        print(block, len(todo_all), "runs", dict(Counter(s.get("variant") or s.get("arm") for s in todo_all)), f"cost units {hours:.3g}")
        return
    smoke = "--smoke" in sys.argv
    min_free, stop_free = option("--min-free-gb", 14.0), option("--stop-free-gb", 5.0)
    if smoke:
        todo_all = smoke_specs(block, todo_all)
    elif not dry:
        (HERE / f"specs_{block}.json").write_text(json.dumps(todo_all, indent=1), encoding="utf-8")
    log = HERE / (f"smoke_{block}_progress.log" if smoke else f"{block}_progress.log")

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    stamp = lambda: time.strftime("%d.%m %H:%M:%S")
    if "--after" in sys.argv:
        other = sys.argv[sys.argv.index("--after") + 1]
        other_log = HERE / f"{other}_progress.log"
        write(f"{stamp()} {block} waiting for {other}")
        while True:
            lines = [x for x in other_log.read_text(encoding="utf-8").splitlines() if x.strip()] if other_log.exists() else []
            if lines and lines[-1].rstrip().endswith(f"{other} done"):
                break
            if lines and STOP_NOTE in lines[-1]:
                write(f"{stamp()} {block} not started: {other} was {STOP_NOTE}")
                return
            time.sleep(30)
    if dry:
        todo = todo_all
    else:
        for s in todo_all:
            (HERE / s["_results"]).mkdir(exist_ok=True)
        todo = [s for s in todo_all if not (HERE / s["_results"] / f"{s['run_id']}.json").exists()]
    write(f"{stamp()} {block} start {len(todo)} of {len(todo_all)} runs, {workers} workers, free cores now {nightlib.free_cores()}, "
          f"{free_gb():.1f} GB RAM free, a run starts with at least {min_free:g} GB free, the block stops below {stop_free:g} GB")
    pending, running, started, peaks, k, last_note = list(todo), {}, {}, [], 0, 0.0

    def can_start():
        young = sum(1 for t in started.values() if time.time() - t < YOUNG_SECONDS)
        return free_gb() - (max(peaks) if peaks else RESERVE_GB) * young >= min_free

    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        while pending or running:
            while pending and len(running) < workers and can_start():
                s = pending.pop(0)
                fut = ex.submit(work, s)
                running[fut], started[fut] = s, time.time()
            if not running:
                if time.time() - last_note > 600:
                    write(f"{stamp()} {block} waits: {free_gb():.1f} GB RAM free, a run needs {min_free:g} GB free ({len(pending)} runs left)")
                    last_note = time.time()
                time.sleep(20)
                continue
            done, _ = wait(list(running), timeout=20, return_when=FIRST_COMPLETED)
            for fut in done:
                s = running.pop(fut)
                started.pop(fut, None)
                k += 1
                try:
                    rec = fut.result()
                    if rec.get("peak_tree_gb"):
                        peaks.append(rec["peak_tree_gb"])
                    write(f"{stamp()} [{k}/{len(todo)}] {s['run_id']} {rec.get('status')} hb={rec.get('hb_seconds')}s "
                          f"total={rec.get('total_seconds')}s sha={rec.get('frame_sha')} rhat={(rec.get('diagnostics') or {}).get('rhat_max')} "
                          f"mem={rec.get('peak_tree_gb')}GB free={free_gb():.0f}GB err={json.dumps(rec.get('error'))[:200]} {rec.get('extra_error') or ''}")
                except Exception as exc:  # noqa: BLE001
                    write(f"{stamp()} [{k}/{len(todo)}] {s['run_id']} worker-exception {type(exc).__name__}: {exc}")
            if free_gb() < stop_free:
                for child in psutil.Process().children(recursive=True):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                write(f"{stamp()} {block} {STOP_NOTE}: {free_gb():.1f} GB RAM free, {len(running)} runs interrupted, "
                      f"{len(pending) + len(running)} runs left; start the same command again later")
                os._exit(3)
            if pending and len(running) < workers and not can_start() and time.time() - last_note > 600:
                write(f"{stamp()} {block} holds back: {free_gb():.1f} GB RAM free, {len(running)} runs running, {len(pending)} waiting")
                last_note = time.time()
    write(f"{stamp()} {block} done")


if __name__ == "__main__":
    main()
