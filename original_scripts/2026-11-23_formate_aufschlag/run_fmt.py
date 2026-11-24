"""Runs the first look at the price premium with other question formats (see fmtwtp.py). Resumable.

Usage: python run_fmt.py <workers> [--smoke] [--full]
  --full    the whole block (see plan): about 380 runs
  --smoke   three runs with 150 respondents (one per format) plus the CBC study itself, into results_fmt_smoke

Plan: the CBC studies of the block 'addon' with a feature that nearly everybody prefers (arm 'agree': 10 tasks of
3 options, attributes with 3, 4, 4 levels, the feature, five prices; typical price reaction), populations 0 to 3,
repetition 0, with 150 and 400 respondents: 8 studies, each in three formats = 24 runs. The CBC values of the same
studies are already in results_addon.
Memory guard as in run_night.py: a run starts only with at least 14 GB of free RAM, and below 5 GB the block stops.
"""
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import psutil

for k in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
HERE = Path(__file__).resolve().parent
NIGHT = HERE.parents[1] / "_scratch" / "2026-11-21_nachtlauf"
sys.path.insert(0, str(HERE))
import fmtwtp  # noqa: E402

MIN_FREE_GB, STOP_FREE_GB = 14.0, 5.0


def free_gb():
    return psutil.virtual_memory().available / 2 ** 30


def plan(smoke: bool, full: bool = False) -> list[dict]:
    base = json.loads((NIGHT / "specs_addon.json").read_text(encoding="utf-8"))
    results = "results_fmt_smoke" if smoke else "results_fmt1"
    if full:
        # every CBC study of the block 'addon' with 150, 400 and 1,000 respondents in which people agree on the feature
        # or are split (10 tasks of 3 options, 8 populations, 2 repetitions), and of the block 'addon4' the studies with
        # 16 tasks of 4 options (all formats) and 8 tasks of 2 options (dual-response only: best-worst needs three options)
        pick = [(s, fmtwtp.METHODS) for s in base if s["arm"] in ("agree", "split") and s["n"] in (150, 400, 1000)]
        more = json.loads((NIGHT / "addon4.json").read_text(encoding="utf-8"))
        pick += [(s, fmtwtp.METHODS) for s in more if s["arm"] == "s16x4_typ_agree"]
        pick += [(s, ("drcbc",)) for s in more if s["arm"] == "s8x2_typ_agree"]
    else:
        chosen = [s for s in base if s["arm"] == "agree" and s["rep"] == 0 and s["pop"] in ((0,) if smoke else (0, 1, 2, 3))
                  and s["n"] in ((150,) if smoke else (150, 400))]
        pick = [(s, (("cbc",) if smoke else ()) + fmtwtp.METHODS) for s in chosen]
    out = []
    for s, methods in pick:
        for method in methods:
            out.append(dict(s, method=method, run_id=f"fmt_{method}__{s['run_id']}", cbc_run_id=s["run_id"], _results=results,
                            stage="fmt1", arm=f"{method}_{s['arm']}"))
    return sorted(out, key=lambda s: s["n"])


def main():
    workers, smoke = int(sys.argv[1]), "--smoke" in sys.argv
    todo = plan(smoke, "--full" in sys.argv)
    results = NIGHT / todo[0]["_results"]
    done = lambda s: (results / f"{s['run_id']}.json").exists() and json.loads((results / f"{s['run_id']}.json").read_text(encoding="utf-8")).get("status") == "ok"
    todo = [s for s in todo if not done(s)]
    stamp = lambda: time.strftime("%d.%m %H:%M:%S")
    print(stamp(), "runs to do:", len(todo), "| workers", workers, "| free GB", round(free_gb(), 1), flush=True)
    pending, running = list(todo), {}
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        while pending or running:
            while pending and len(running) < workers and free_gb() >= MIN_FREE_GB:
                spec = pending.pop(0)
                running[pool.submit(fmtwtp.work, spec)] = spec
                time.sleep(2)
            if free_gb() < STOP_FREE_GB:
                print(stamp(), "stopped by the memory guard", flush=True)
                for f in running:
                    f.cancel()
                os._exit(3)
            finished, _ = wait(list(running), timeout=20, return_when=FIRST_COMPLETED)
            for f in finished:
                spec = running.pop(f)
                try:
                    rec = f.result()
                    print(stamp(), spec["run_id"], rec.get("status"), "| hb s", rec.get("hb_seconds"), "| total s", rec.get("total_seconds"),
                          "| buy", rec.get("buy_share"), "|", (rec.get("error") or "")[:300] if rec.get("status") != "ok" else "", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(stamp(), spec["run_id"], "worker failed:", repr(exc)[:300], flush=True)
    print(stamp(), "done", flush=True)


if __name__ == "__main__":
    main()
