#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import numpy as np

Q30 = 30
Q15 = 15
Q31 = 2**31

BUNDLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_JSON = BUNDLE_ROOT / "workdir" / "lut128" / "selected_pareto_lut128.json"
DEFAULT_OUT = BUNDLE_ROOT / "workdir" / "coeff_banks" / "coeffs_q31_all_LH_lut128.json"

CANDIDATE_BOUNDARY_KEYS = [
    "boundaries_lut128",
    "boundaries_final",
    "boundaries_nearest",
    "boundaries_quantized",
    "boundaries_lut",
]

def gen_coeffs_integer_u(boundaries, deg, nsamp=2000):
    boundaries = np.asarray(boundaries, dtype=np.float64)
    S = len(boundaries) - 1
    if len(deg) != S:
        raise ValueError(f"deg length {len(deg)} != S {S}")

    # Match runtime loader
    b_q30 = np.rint(boundaries * (1 << Q30)).astype(np.int64)
    lo = b_q30[:-1]
    hi = b_q30[1:]
    mid_q30 = (lo + hi) // 2
    half_q30 = (hi - lo) // 2

    inv_half_q48 = np.zeros(S, dtype=np.int64)
    for s in range(S):
        h = int(half_q30[s])
        inv_half_q48[s] = (1 << 48) // h if h > 0 else 0

    coeffs_q31 = []
    for s in range(S):
        d = int(deg[s])
        if d < 0 or d > 5:
            raise ValueError("deg must be 0..5")

        mid_real = mid_q30[s] / float(1 << Q30)

        # deg 0 or degenerate segment -> midpoint constant
        if d == 0 or half_q30[s] <= 0:
            c0 = math.exp(mid_real)
            coeffs_q31.append([int(round(c0 * Q31))])
            continue

        # Chebyshev nodes in [-1,1]
        n = max(int(nsamp), 200 * (d + 1))
        k = np.arange(n)
        u_cheb = np.cos((2 * k + 1) / (2 * n) * np.pi)

        # x_q30 sampled using integer mid/half
        x_q30 = mid_q30[s] + (half_q30[s] * u_cheb).astype(np.int64)

        # runtime-style u_q15
        num = ((x_q30 - mid_q30[s]) << Q15).astype(np.int64)
        abs_num = np.abs(num)
        u_mag = (abs_num * inv_half_q48[s]) >> 48
        u_q15 = np.where(num < 0, -u_mag, u_mag).astype(np.int64)
        u_q15 = np.clip(u_q15, -32768, 32767)

        # fit in float-u corresponding to runtime u_q15
        u = u_q15.astype(np.float64) / float(1 << Q15)

        # target y = exp(x_real)
        x_real = x_q30.astype(np.float64) / float(1 << Q30)
        y = np.exp(x_real)

        # power basis ascending [c0..cd]
        c = np.polynomial.polynomial.polyfit(u, y, deg=d)

        # quantize to Q31
        cq = [int(np.round(ci * Q31)) for ci in c.tolist()]
        cq = [max(-(2**31), min(2**31 - 1, v)) for v in cq]
        coeffs_q31.append(cq)

    return coeffs_q31


def _pick_boundaries(rec: dict, boundary_key: str):
    """
    Returns: (boundaries_list, key_used)
    Search order:
      - chosen[boundary_key] if boundary_key != 'auto'
      - chosen[candidate] for candidates (auto mode)
      - rec[candidate] as fallback (some generators store it outside chosen)
      - chosen['boundaries'] final fallback
    """
    chosen = rec.get("chosen", {})
    if not isinstance(chosen, dict):
        chosen = {}

    if boundary_key != "auto":
        if boundary_key in chosen:
            return chosen[boundary_key], f"chosen.{boundary_key}"
        if boundary_key in rec:
            return rec[boundary_key], f"rec.{boundary_key}"
        if "boundaries" in chosen:
            return chosen["boundaries"], "chosen.boundaries(fallback)"
        return None, "missing"

    # auto mode
    for k in CANDIDATE_BOUNDARY_KEYS:
        if k in chosen:
            return chosen[k], f"chosen.{k}"
    for k in CANDIDATE_BOUNDARY_KEYS:
        if k in rec:
            return rec[k], f"rec.{k}"
    if "boundaries" in chosen:
        return chosen["boundaries"], "chosen.boundaries(fallback)"
    return None, "missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_json", default=str(DEFAULT_SELECTED_JSON))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--nsamp", type=int, default=2000)
    ap.add_argument("--only_status_ok", action="store_true",
                    help="If set, only process entries with status == 'ok'.")
    ap.add_argument("--boundary_key", default="auto",
                    help="Which boundary field to use. Use 'auto' (default) to prefer LUT keys like boundaries_lut128/boundaries_final, else give an exact key.")
    args = ap.parse_args()

    with open(args.selected_json, "r") as f:
        sel = json.load(f)

    selected = sel.get("selected", {})
    if not isinstance(selected, dict):
        raise RuntimeError("selected JSON missing top-level key: 'selected'")

    out = {
        "source_selected_pareto": args.selected_json,
        "meta": sel.get("meta", {}),
        "selection_policy": sel.get("selection_policy", {}),
        "notes": {
            "fit_target": "exp(x)",
            "fit_space": "u in [-1,1] derived from runtime u_q15 mapping",
            "basis": "power basis ascending c0..cd",
            "coeff_scale": "Q31",
            "boundary_scale": "Q30",
            "boundary_key_mode": args.boundary_key,
            "auto_candidates": CANDIDATE_BOUNDARY_KEYS,
        },
        "coeffs_tree": {},
        "stats": {
            "processed": 0,
            "skipped_no_chosen": 0,
            "skipped_status": 0,
            "skipped_missing_fields": 0,
            "errors": 0,
        }
    }

    for L, heads in selected.items():
        if not isinstance(heads, dict):
            continue
        for H, rec in heads.items():
            if not isinstance(rec, dict):
                continue

            status = rec.get("status", None)
            if args.only_status_ok and status != "ok":
                out["stats"]["skipped_status"] += 1
                continue

            chosen = rec.get("chosen", None)
            if not isinstance(chosen, dict):
                out["stats"]["skipped_no_chosen"] += 1
                continue

            segments = chosen.get("segments", None)
            deg = chosen.get("deg", None)

            boundaries, boundaries_key_used = _pick_boundaries(rec, args.boundary_key)

            if segments is None or boundaries is None or deg is None:
                out["stats"]["skipped_missing_fields"] += 1
                continue

            try:
                segments = int(segments)
                boundaries_f = [float(x) for x in boundaries]
                deg_i = [int(x) for x in deg]

                if len(boundaries_f) != segments + 1:
                    raise ValueError(f"{L}/{H}: boundaries len {len(boundaries_f)} != segments+1 {segments+1}")
                if len(deg_i) != segments:
                    raise ValueError(f"{L}/{H}: deg len {len(deg_i)} != segments {segments}")

                coeffs_q31 = gen_coeffs_integer_u(boundaries_f, deg_i, nsamp=args.nsamp)

                # keep orig too if present
                boundaries_orig = chosen.get("boundaries", None)
                if boundaries_orig is not None:
                    boundaries_orig = [float(x) for x in boundaries_orig]

                out["coeffs_tree"].setdefault(L, {})
                out["coeffs_tree"][L][H] = {
                    "segments": segments,
                    "clamp_min": float(boundaries_f[0]),
                    "clamp_max": float(boundaries_f[-1]),
                    "boundaries": boundaries_f,
                    "boundaries_key_used": boundaries_key_used,
                    "boundaries_orig": boundaries_orig,
                    "deg": deg_i,
                    "coeffs_q31": coeffs_q31,

                    "cost": chosen.get("cost", None),
                    "kl": chosen.get("kl", None),
                    "expected_deg": chosen.get("expected_deg", None),
                    "rank": chosen.get("rank", None),
                    "per_run_json": chosen.get("per_run_json", None),
                    "log": chosen.get("log", None),
                    "source": chosen.get("source", None),
                    "status": status,
                }

                out["stats"]["processed"] += 1

            except Exception as e:
                out["stats"]["errors"] += 1
                out["coeffs_tree"].setdefault(L, {})
                out["coeffs_tree"][L][H] = {
                    "status": "error",
                    "error": str(e),
                    "segments": segments,
                    "boundaries_key_used": boundaries_key_used,
                }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {args.out}")
    print("Stats:", out["stats"])


if __name__ == "__main__":
    main()
