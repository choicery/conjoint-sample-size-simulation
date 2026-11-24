"""Price premium with other question formats (23.11.2026): dual-response CBC, best-worst CBC and best-worst CBC with a
buy question after the choice.

Why: the calculator plans the price premium from 1,424 simulated studies, all with plain CBC tasks. The user asked for a
first look at the other formats ("mach mal erstmal wenige Läufe für DR-CBC, BW-CBC und BW-CBC none").

How: a run is the CBC study of the block 'addon' (nightlib.work_addon: same population, people, tasks, prices and
premium questions, same seeds) with three things exchanged:
  answers      the choice of the CBC study (same random draws) plus the extra answer of the format: a buy / no-buy
               answer, a worst choice, or both
  data         the long format of the app's converter for this format, with the price as a column of level codes
  estimation   the app's HB service for this format, the price as a linear attribute as in the CBC studies
Everything after the estimation (stored result, premium of the market simulator, true premium) is the code of the CBC
studies. Results land in _scratch/2026-11-21_nachtlauf/<spec['_results']>, next to the CBC blocks.

Simulated people: the None utility has a spread of ASC_SD between people and a mean set so that they buy in BUY_SHARE
of the tasks (the earlier format simulation had about 25% no-buy answers). The worst choice follows the reversed
utilities, never the product chosen as best; people answer it as carefully as the first choice.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NIGHT = ROOT / "_scratch" / "2026-11-21_nachtlauf"
METHODS = ("drcbc", "bwcbc", "bwcbc_none")
BUY_SHARE = 0.75
ASC_SD = 1.0
_STATE = {}


def _softmax(u):
    e = np.exp(u - u.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _install(method: str) -> None:
    import wtplib

    def simulate(pop, params, coef, codes, rng):
        n, t, o, a = codes.shape
        v = np.zeros((n, t, o))
        for ai, mat in enumerate(wtplib.partworths(params, pop["levels"])):
            v += np.take_along_axis(mat[:, None, :], codes[:, :, :, ai].reshape(n, 1, t * o), axis=2).reshape(n, t, o)
        pr = pop["price"]
        v += coef[:, None, None] * ((pr["values"] - pr["center"]) / pr["half"])[codes[:, :, :, a - 1]]
        u = v + rng.gumbel(size=(n, t, o))            # the same draws as in the CBC study with these seeds
        first = np.argmax(u, axis=2)
        resp = {"first": first}
        if method in ("drcbc", "bwcbc_none"):
            best_u = np.take_along_axis(u, first[:, :, None], axis=2)[:, :, 0]
            g_none = rng.gumbel(size=(n, t))
            z = rng.standard_normal(n) * ASC_SD
            lo, hi = -30.0, 30.0
            for _ in range(60):                       # mean of the None utility for the wanted share of buy answers
                mid = 0.5 * (lo + hi)
                if (best_u > (mid + z)[:, None] + g_none).mean() > BUY_SHARE:
                    lo = mid
                else:
                    hi = mid
            resp["buy"] = best_u > (0.5 * (lo + hi) + z)[:, None] + g_none
            _STATE["buy_share"] = float(resp["buy"].mean())
            _STATE["asc_mean"] = 0.5 * (lo + hi)
        if method in ("bwcbc", "bwcbc_none"):
            p = _softmax(-v)
            np.put_along_axis(p, first[:, :, None], 0.0, axis=2)      # never the product chosen as best
            cdf = np.cumsum(p / p.sum(axis=2, keepdims=True), axis=2)
            worst = np.minimum((rng.random((n, t))[:, :, None] > cdf).sum(axis=2), o - 1)
            assert not (worst == first).any()
            resp["worst"] = worst
        _STATE["options"] = o
        _STATE["first_sha"] = __import__("hashlib").sha256(first.astype(np.int64).tobytes()).hexdigest()[:16]
        return resp

    def frame(codes, resp, names):
        n, t, o, a = codes.shape
        first = resp["first"]
        if method == "drcbc":                         # as DRCBCDataConverter._phase_build_long_format emits it
            blk = o + 1
            alt = np.tile(np.arange(1, blk + 1), n * t)
            data = {"ID": np.repeat(np.arange(1, n + 1), t * blk), "Task": np.tile(np.repeat(np.arange(1, t + 1), blk), n), "Alt": alt}
            for ai, name in enumerate(names):
                data[name] = np.concatenate([codes[:, :, :, ai] + 1, np.zeros((n, t, 1), dtype=int)], axis=2).ravel().astype(int)
            data["NoneOpt"] = (alt == blk).astype(int)
            data["SelectedAlt"] = np.repeat((first + 1).ravel(), blk).astype(int)
            data["NoneResponse"] = np.repeat(np.where(resp["buy"], 1, 2).ravel(), blk).astype(int)
            return pd.DataFrame(data)
        if method == "bwcbc":                         # product-only single-block format of BWCBCConverter
            data = {"sys_RespNum": np.repeat(np.arange(1, n + 1), t * o), "Task": np.tile(np.repeat(np.arange(1, t + 1), o), n),
                    "Concept": np.tile(np.arange(1, o + 1), n * t)}
            for ai, name in enumerate(names):
                data[name] = (codes[:, :, :, ai] + 1).ravel().astype(int)
            pos = np.arange(o)[None, None, :]
            data["Response"] = np.where(pos == first[:, :, None], 1, np.where(pos == resp["worst"][:, :, None], -1, 0)).ravel().astype(int)
            return pd.DataFrame(data)
        # bwcbc_none, rows as BWCBCConverter emits them with the None follow-up:
        # buy -> worst block + follow-up block (all products + None, chosen = best product);
        # no  -> best block + worst block + follow-up block (chosen = None).
        rows = []
        for i in range(n):
            for k in range(t):
                b, w, buy = int(first[i, k]), int(resp["worst"][i, k]), bool(resp["buy"][i, k])
                levels = [[int(codes[i, k, j, ai]) + 1 for ai in range(a)] for j in range(o)]
                blocks = [] if buy else [("best", [1 if j == b else 0 for j in range(o)])]
                blocks.append(("worst", [-1 if j == w else 0 for j in range(o)]))
                for name, values in blocks:
                    for j in range(o):
                        rows.append([i + 1, k + 1, j + 1, *levels[j], values[j], name, int(values[j] != 0), 0])
                follow = [1 if (buy and j == b) else 0 for j in range(o)] + [0 if buy else 1]
                for j in range(o + 1):
                    rows.append([i + 1, k + 1, j + 1, *(levels[j] if j < o else [0] * a), follow[j], "none_followup", follow[j], int(j == o)])
        return pd.DataFrame(rows, columns=["sys_RespNum", "Task", "Concept", *names, "Response", "obs_type", "choice", "NoneOpt"])

    def estimate(frame_df, levels, price_values, project_dir, run_dir, hb, price_constraint):
        from simlib import _consume

        names = [f"A{i}" for i in range(1, len(levels) + 1)] + [wtplib.PRICE]
        labels = {f"A{i}": [f"L{k}" for k in range(1, l + 1)] for i, l in enumerate(levels, 1)}
        labels[wtplib.PRICE] = [wtplib.price_label(v) for v in price_values]
        types = {name: "categorical" for name in names}
        types[wtplib.PRICE] = "linear"
        n_levels = list(levels) + [len(price_values)]
        linear_values = {wtplib.PRICE: [float(v) for v in price_values]}
        linear_scales = {wtplib.PRICE: 1.0}
        constraints = {wtplib.PRICE: price_constraint} if price_constraint else None
        common = dict(n_chains=hb["n_chains"], n_burn=hb["n_burn"], n_draws=hb["n_draws"], prior_variance=hb["prior_variance"],
                      prior_mean=hb["prior_mean"], prior_variance_mean=hb["prior_variance_mean"], seed=hb["seed"],
                      target_accept=hb["target_accept"], thinning=hb["thinning"], df_offset=hb["df_offset"], parallel_chains=False,
                      individual_rhat=False, respondent_holdout_enabled=False, respondent_holdout_fraction=0.2)
        if method in ("bwcbc", "bwcbc_none"):
            from app.services.hb_analysis_julia_bwcbc import JuliaBWCBCHBService

            svc = JuliaBWCBCHBService(project_dir, run_dir=run_dir)
            gen = svc.run_hb_analysis_streaming(
                choicemodel_data=frame_df, attr_names=names, n_levels=n_levels, attr_labels=labels, attr_types=types,
                linear_values=linear_values, linear_scales=linear_scales, n_tasks=int(frame_df["Task"].nunique()),
                n_concepts=_STATE["options"], constraints=constraints, include_none_option=(method == "bwcbc_none"),
                estimation_tasks="both", task_id="sim",
                respondent_mapping={i: f"sim{i}" for i in range(1, int(frame_df["sys_RespNum"].nunique()) + 1)}, **common)
        else:
            import app.services.drcbc_hb_julia as drmod

            class Converter:
                """Stands in for DRCBCDataConverter: hands over the simulated long format as the converter would."""

                def __init__(self, project_dir, drcbc_id):
                    self.excluded_participant_ids = set()
                    self.run_excluded_participants = ()
                    self.respondent_mapping = {i: f"sim{i}" for i in range(1, int(frame_df["ID"].nunique()) + 1)}

                def convert(self, output_file=None, excluded_ids=None, *, none_required=True, allow_partial_profiles=False, **_):
                    df = frame_df.copy()
                    if output_file:
                        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(output_file, index=False, encoding="utf-8")
                    n_products = int(df.loc[df["NoneOpt"] == 0].groupby(["ID", "Task"]).size().iloc[0])
                    meta = {"n_respondents": int(df["ID"].nunique()), "n_tasks": int(df["Task"].nunique()),
                            "n_alternatives": n_products + 1, "n_products": n_products,
                            "attributes": {a: {"levels": labels[a], "n_levels": len(labels[a])} for a in names},
                            "none_yes_count": 0, "none_no_count": 0, "warnings": []}
                    return df, meta

                def get_attribute_info(self):
                    attributes = {a: {"type": types[a], "levels": labels[a], "level_names": labels[a], "n_levels": len(labels[a])} for a in names}
                    return {"attr_names": list(names), "n_levels": list(n_levels), "attr_labels": dict(labels), "attr_types": dict(types),
                            "attributes": attributes}

            drmod.DRCBCDataConverter = Converter
            svc = drmod.DRCBCHBService(project_dir, "sim", run_dir=run_dir)
            gen = svc.run_analysis(attr_types=types, linear_values=linear_values, linear_scales=linear_scales, constraints=constraints,
                                   include_none_option=True, estimation_method="full_hb_mcmc", **common)
        return _consume(gen)

    wtplib.simulate_choices = simulate
    wtplib.cbc_frame = frame           # nightlib.work_addon wraps this one and stores the fingerprint of the data
    wtplib.estimate = estimate


def work(spec: dict) -> dict:
    for key in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    if str(NIGHT) not in sys.path:
        sys.path.insert(0, str(NIGHT))
    import nightlib

    nightlib.pin()
    if str(nightlib.CHOICERY) not in sys.path:
        sys.path.insert(0, str(nightlib.CHOICERY))
    method = spec["method"]
    if method != "cbc":
        assert method in METHODS, method
        _install(method)
    rec = nightlib.work_addon(spec)
    if method != "cbc":
        rec.update({k: _STATE.get(k) for k in ("buy_share", "asc_mean", "first_sha")})
        # The evaluation of the CBC harness reads the price as one column per price level; the services of the other
        # formats store it as one column. That evaluation is not needed here: the estimation result is stored, and the
        # premium is read from it with the app's market simulator afterwards (soslib). An empty entry per premium
        # question lets soslib take the run.
        stored = (NIGHT / spec["_results"] / "hb_results" / f"{spec['run_id']}.json.gz").exists()
        if rec.get("status") == "missing_columns" and stored and all(c.startswith("Price_") for c in rec.get("missing_beta_cols") or ["x"]):
            rec["status"] = "ok"
            rec["evaluation_of_cbc_harness"] = "skipped: price stored as one column"
            rec["wtp"] = [{} for _ in range(3)]
            import shutil

            shutil.rmtree(NIGHT / ("work_" + spec["_results"]) / spec["run_id"], ignore_errors=True)
        nightlib._write(NIGHT / spec["_results"] / f"{spec['run_id']}.json", rec)
    return rec
