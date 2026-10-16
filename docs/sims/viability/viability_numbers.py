"""The viability numbers of a program tree on SIMULATED axons (no real data):
python docs/sims/viability/viability_numbers.py <tree> <npz folder> <out.json>
(NPZ files from batch_columns.write_simulated_input; docs/viability_and_reference_period.md section 11 compares the
program before and after this change).
For each NPZ: build the rings as the review does, z_quality, viability_v2 (lab coordinates; with the file's precision
when the tree has file_precision_nm) and viability_v2c; dump per pair verdicts, tolerances, h, P."""
import json
import math
import os
import sys

tree, folder, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tree)
os.chdir(tree)
import numpy as np  # noqa: E402

from tools import mps_axial_precision as ap  # noqa: E402
from tools.mps_axial_clusters import viability_v2c  # noqa: E402
from tools.mps_columns_window import build_review_rings  # noqa: E402
from tools.mps_zquality_window import inputs_from_npz  # noqa: E402


def f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, 6)


res_all = {}
for name in sorted(os.listdir(folder)):
    if not name.endswith(".npz"):
        continue
    inp = inputs_from_npz(os.path.join(folder, name))
    res, _p, _l, _w = build_review_rings(inp)
    xyz = (inp.x_nm, inp.y_nm, inp.z_nm)
    zq = ap.z_quality(res)
    kw = {}
    if hasattr(ap, "file_precision_nm"):
        kw = dict(precision_nm=ap.file_precision_nm(inp.lp_lateral_nm, inp.lpz_nm))
    v2 = ap.viability_v2(res, zq, xyz_lab_nm=xyz, **kw)
    v2c = viability_v2c(res, v2, xyz_lab_nm=xyz)
    d = {"p_ref_nm": f(v2.p_ref_nm), "p_nm": f(getattr(v2, "p_nm", v2.p_ref_nm)), "h_nm": f(v2.h_nm),
         "group_z_scale": f(getattr(v2, "group_z_scale", 0.5)), "x": f(v2.x), "f_min": f(v2.f_min),
         "zq_axon": zq.axon_verdict, "v2_viable": [list(p) for p in v2.viable_pairs],
         "v2_marginal": [list(p) for p in v2.marginal_pairs], "pairs": []}
    for b, p in zip(zq.boundaries, v2.pairs):
        d["pairs"].append({"pair": f"{p.ring_a}-{p.ring_b}", "d_sep": f(b.d_sep), "exp_spur": f(b.exp_spur_frac),
                           "zq_verdict": b.verdict, "spur_viable": f(getattr(b, "spur_viable", 0.02)),
                           "spur_marginal": f(getattr(b, "spur_marginal", 0.05)), "peak_a": p.peak_a,
                           "peak_b": p.peak_b, "valley": p.valley, "count_ok": p.count_ok, "v2_verdict": p.verdict})
    d["v2c"] = {f"{q.variant}:{q.ring_a}-{q.ring_b}": [q.verdict, f(q.k_min), q.valley_at_cut] for q in v2c.pairs}
    d["v2c_tnfix"] = {f"{lk.ring_a}-{lk.ring_b}": f(lk.exp_spur_frac_clusters) for lk in v2c.leaks}
    res_all[name] = d
    print(name, "done", flush=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(res_all, fh, indent=1)
