"""Build simlib_fmt.py from the stored simlib.py (2026-11-09_sample_size_sim), which stays untouched.

Added on top of the stored pipeline:
  1. method 'bwcbc_none': best-worst CBC with the buy / no-buy follow-up (same patches as grid/bw_none/make_lib.py of 10.11.).
  2. For the formats with a None (drcbc, bwcbc_none, cbc_none) every run also stores the six standard markets WITHOUT the
     None alternative (rec['shares_products_only']): simulator with include_none=False against the true product shares.
     These markets are the same as in the plain CBC runs of the same population, so the formats can be compared directly.
  3. The extra markets (2 and 5 products, share of preference, never with None) are computed for every format.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "2026-11-09_sample_size_sim" / "simlib.py"
src = SRC.read_text(encoding="utf-8")

REPL = [
    ('with_none = method in ("cbc_none", "drcbc")', 'with_none = method in ("cbc_none", "drcbc", "bwcbc_none")'),
    ('    elif method == "bwcbc":\n        from app.services.hb_analysis_julia_bwcbc import JuliaBWCBCHBService',
     '    elif method in ("bwcbc", "bwcbc_none"):\n        from app.services.hb_analysis_julia_bwcbc import JuliaBWCBCHBService'),
    ('            include_none_option=False, estimation_tasks="both", task_id="sim",',
     '            include_none_option=(method == "bwcbc_none"), estimation_tasks="both", task_id="sim",'),
    # Response generation: best = argmax of the noisy utilities, buy if it beats the None utility (dual response),
    # worst from the full set with reversed utilities, redrawn until it differs from best.
    ('    else:\n        raise ValueError(method)\n    return out\n',
     '''    elif method == "bwcbc_none":
        g_none = rng.gumbel(size=(n, t))
        u = v + gumbel
        best = np.argmax(u, axis=2)
        best_u = np.take_along_axis(u, best[:, :, None], axis=2)[:, :, 0]
        out["buy"] = best_u > (asc[:, None] + g_none)
        p_worst = softmax(-v)
        worst = np.zeros((n, t), dtype=int)
        redraws = 0
        for i in range(n):
            for k in range(t):
                w = rng.choice(o, p=p_worst[i, k])
                while w == best[i, k]:
                    redraws += 1
                    w = rng.choice(o, p=p_worst[i, k])
                worst[i, k] = w
        out["best"], out["worst"], out["worst_redraws"] = best, worst, redraws
    else:
        raise ValueError(method)
    return out
'''),
    ('        else:\n            frame = bwcbc_frame(codes, resp)\n',
     '''        elif method == "bwcbc_none":
            frame = bwcbc_none_frame(codes, resp)
            rec["observed_no_buy_share"] = float((~resp["buy"]).mean())
            rec["worst_redraws"] = int(resp["worst_redraws"])
        else:
            frame = bwcbc_frame(codes, resp)
'''),
    # 2. product-only standard markets for the None formats
    ('        rec["shares"] = rows\n',
     '''        rec["shares"] = rows
        if with_none:
            rows_po = []
            for si, scen in enumerate(scen_list):
                pop_sh = true_shares(pop, scen, False)
                ins_sh = true_shares(pop, scen, False, params=params)
                sims = simulator_shares(results, project_dir, scen, False)
                for rule, est_sh in sims["shares"].items():
                    if len(est_sh) != len(pop_sh):
                        rec.setdefault("share_length_mismatch_po", []).append((si, rule, len(est_sh), len(pop_sh)))
                        continue
                    for j in range(len(pop_sh)):
                        rows_po.append({"scenario": si, "n_products": len(scen), "alt": j, "is_none": False, "rule": rule,
                                        "est": float(est_sh[j]), "pop": float(pop_sh[j]), "insample": float(ins_sh[j])})
            rec["shares_products_only"] = rows_po
'''),
    # 3. extra markets for every format (they never contain None)
    ('        if spec.get("extra_scenarios") and not with_none:\n', '        if spec.get("extra_scenarios"):\n'),
]
for old, new in REPL:
    assert src.count(old) == 1, (src.count(old), old[:80])
    src = src.replace(old, new)

src += '''

def bwcbc_none_frame(codes, resp) -> pd.DataFrame:
    """Rows exactly as BWCBCConverter emits them with the None follow-up enabled:
    buy  -> worst block + follow-up block (all products + None, chosen = best product);
    no   -> best block + worst block + follow-up block (chosen = None)."""
    n, t, o, a = codes.shape
    rows = []
    for i in range(n):
        for k in range(t):
            b, w, buy = int(resp["best"][i, k]), int(resp["worst"][i, k]), bool(resp["buy"][i, k])
            levels = [[int(codes[i, k, j, ai]) + 1 for ai in range(a)] for j in range(o)]
            blocks = [] if buy else [("best", [1 if j == b else 0 for j in range(o)])]
            blocks.append(("worst", [-1 if j == w else 0 for j in range(o)]))
            for name, resp_vals in blocks:
                for j in range(o):
                    rows.append([i + 1, k + 1, j + 1, *levels[j], resp_vals[j], name, int(resp_vals[j] != 0), 0])
            follow = [1 if (buy and j == b) else 0 for j in range(o)] + [0 if buy else 1]
            for j in range(o + 1):
                lv = levels[j] if j < o else [0] * a
                rows.append([i + 1, k + 1, j + 1, *lv, follow[j], "none_followup", follow[j], int(j == o)])
    cols = ["sys_RespNum", "Task", "Concept"] + [f"A{ai + 1}" for ai in range(a)] + ["Response", "obs_type", "choice", "NoneOpt"]
    return pd.DataFrame(rows, columns=cols)
'''
(HERE / "simlib_fmt.py").write_text(src, encoding="utf-8")
print("written", len(src))
