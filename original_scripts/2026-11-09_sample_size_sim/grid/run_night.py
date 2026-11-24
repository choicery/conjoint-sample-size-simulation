"""Overnight runs for figure 1 of the sample size page (16./17.11.2026).

Stages, in this order (earlier = more important). No run starts after the deadline.
  pop1   a second population (new people, new markets) for all 67 designs        -> results_pop/
  long   five times the chain length for 12 of the new designs (N=400, both reps),
         twenty times for the two thinnest old designs                           -> results_long/
  seed   same answers, HB seeds 7 and 1234, for 9 more old designs (N=400, rep 0) -> results_seed/
  pop2   a third population for all 67 designs                                    -> results_pop/
  pop3   a fourth population for all 67 designs                                   -> results_pop/

Every run also simulates 8 markets with 2 products and 4 markets with 5 products (shares_extra).
Runs pinned to the 16 efficiency cores so other work keeps the performance cores.
Usage: python run_night.py <hours until deadline | HH:MM>     (resumable)
"""
import ctypes
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
import run_more  # noqa: E402

EXTRA = {"2": 8, "5": 4}
LONG_NEW = ["a5l5t8o2", "m554433t10o2", "a4l4t8o2", "m5345345t9o3", "a5l5t10o3", "m33445t8o3", "m5433t8o3",
            "a5l4t12o3", "a4l5t10o4", "a6l3t8o4", "m4354t16o3", "a5l3t16o4"]
LONG_X20 = ["a8l5t6o3", "a6l5t6o3"]
SEED_OLD = ["a4l5t6o2", "a6l3t6o2", "a8l4t8o3", "a5l4t6o3", "a4l4t6o3", "med", "a6l3t14o3", "a4l3t12o3", "a4l3t16o4"]


def low_priority():
    """Formerly below-normal priority; Windows then throttled the runs about threefold. Now only pinned to the
    efficiency cores at normal priority (see run_more.pin_to_efficiency_cores)."""
    run_more.pin_to_efficiency_cores()


def base_designs():
    """One spec per design (old grid, fill and the 36 new ones): name, levels, tasks, options, pop_seed."""
    out = {}
    for s in list(run_more.stored_specs().values()) + run_more.build("more"):
        if s["method"] == "cbc" and s["hetero_label"] == "realistic" and s["design"] not in out:
            out[s["design"]] = s
    return out


def pop_stage(p):
    out = []
    for dname, s in base_designs().items():
        pop_seed = s["pop_seed"] + 100000 * p
        for n in run_more.NS:
            for rep in range(2):
                out.append(dict(run_id=f"{dname}_cbc_realistic_p{p}_n{n}_r{rep}", design=dname, levels=list(s["levels"]),
                                tasks=s["tasks"], options=s["options"], method="cbc", n=n, rep=rep, hetero="realistic",
                                hetero_label="realistic", none_share=None, hb_seed=42, pop_seed=pop_seed, pop=p,
                                data_seed=pop_seed * 10 + n * 3 + rep * 101, _results="results_pop",
                                extra_scenarios=EXTRA, stage=f"pop{p}"))
    return sorted(out, key=lambda s: -run_more.cost(s))


def long_stage():
    out = []
    for s in run_more.build("more"):
        if s["design"] in LONG_NEW and s["n"] == 400:
            out.append(dict(s, _results="results_long", n_burn=50000, n_draws=50000, stage="long"))
    for s in run_more.stored_specs().values():
        if (s["design"] in LONG_X20 and s["method"] == "cbc" and s["hetero_label"] == "realistic" and s["n"] == 400
                and s.get("hb_seed", 42) == 42):
            out.append(dict(s, run_id=s["run_id"] + "_x20", _results="results_long", n_burn=200000, n_draws=200000,
                            extra_scenarios=EXTRA, stage="long"))
    return sorted(out, key=lambda s: -run_more.cost(s))


def seed_stage():
    out = []
    for s in run_more.stored_specs().values():
        if (s["design"] in SEED_OLD and s["method"] == "cbc" and s["hetero_label"] == "realistic" and s["n"] == 400
                and s["rep"] == 0 and s.get("hb_seed", 42) == 42):
            for seed in (7, 1234):
                out.append(dict(s, run_id=f"{s['run_id']}_s{seed}", hb_seed=seed, _results="results_seed",
                                extra_scenarios=EXTRA, stage="seed"))
    return sorted(out, key=lambda s: -run_more.cost(s))


def run(spec):
    low_priority()
    if time.time() > spec["_deadline"]:
        return {"status": "skipped_deadline"}
    return run_more.run(spec)


def main():
    low_priority()
    arg = sys.argv[1]
    if ":" in arg:  # absolute local time HH:MM, next occurrence
        hh, mm = (int(x) for x in arg.split(":"))
        now = time.localtime()
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
        if deadline <= time.time():
            deadline += 24 * 3600
    else:
        deadline = time.time() + float(arg) * 3600
    plan = pop_stage(1) + long_stage() + seed_stage() + pop_stage(2) + pop_stage(3)
    (GRID / "grid_specs_night.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    todo = []
    for s in plan:
        (GRID / s["_results"]).mkdir(exist_ok=True)
        if not (GRID / s["_results"] / f"{s['run_id']}.json").exists():
            todo.append(dict(s, _deadline=deadline))
    log = GRID / "more_progress.log"

    def write(line):
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line, flush=True)

    write(f"{time.strftime('%d.%m %H:%M:%S')} night start {len(todo)} of {len(plan)} runs, deadline "
          f"{time.strftime('%d.%m %H:%M', time.localtime(deadline))}")
    skipped = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("GRID_WORKERS", "12"))) as ex:
        futs = {ex.submit(run, s): s for s in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                rec = fut.result()
                if rec.get("status") == "skipped_deadline":
                    skipped += 1
                    continue
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{s['stage']} {k}/{len(todo)}] {s['run_id']} {rec.get('status')} "
                      f"hb={rec.get('hb_seconds')}s total={rec.get('total_seconds')}s "
                      f"err={json.dumps(rec.get('error'))[:200]}")
            except Exception as exc:  # noqa: BLE001
                write(f"{time.strftime('%d.%m %H:%M:%S')} [{s['stage']} {k}/{len(todo)}] {s['run_id']} worker-exception {exc}")
    write(f"{time.strftime('%d.%m %H:%M:%S')} night done, {skipped} runs not started because of the deadline")


if __name__ == "__main__":
    main()
