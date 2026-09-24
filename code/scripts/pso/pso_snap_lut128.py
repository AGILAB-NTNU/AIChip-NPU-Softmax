#!/usr/bin/env python3
"""
Title: Quantize to the nearest bin (no guesswork) — LUT128 boundary snapping with collision skip

Goal:
  Quantize chosen boundaries to LUT128 grid (nearest EDGE), but:
    - If 2+ boundaries fall into the SAME LUT BIN (interior boundaries), SKIP snapping entirely
      and keep original boundaries as-is. (This matches your runtime idea: BST fallback.)

Input:
  selected_pareto.json (your chosen per L/H configs)

Output:
  - out_json: same JSON with per-head annotations:
      boundaries_lut128_nearest (if computed)
      boundaries_lut128 (final; either snapped or original if skipped)
      lut128_outcome: ok|skip|fail
      lut128_skip_reason / lut128_fail_reason
      lut128_bin_collision stats (when skipped)
  - out_csv: detailed per-head diagnostics with reasons

Run:
  python3 scripts/pso/pso_snap_lut128.py \
    --in_json  workdir/pareto/selected_pareto.json \
    --out_json workdir/lut128/selected_pareto_lut128.json \
    --out_csv  workdir/lut128/selected_pareto_lut128_debug.csv \
    --bins 128 --xmin -20 --xmax 0 --fix_endpoints
"""

import argparse
import json
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def is_finite_list(xs: List[float]) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in xs)


def strictly_increasing(xs: List[float]) -> bool:
    return all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))


def lut_edges(xmin: float, xmax: float, bins: int) -> np.ndarray:
    return np.linspace(xmin, xmax, bins + 1, dtype=np.float64)


def bin_indices_for_boundaries(boundaries: np.ndarray, xmin: float, xmax: float, bins: int) -> np.ndarray:
    """
    LUT bin index in [0, bins-1] for each boundary value.
    Bins are [edge[i], edge[i+1]) except last edge at xmax.
    For boundary==xmax, force bin=bins-1.
    """
    bw = (xmax - xmin) / float(bins)
    # floor mapping
    idx = np.floor((boundaries - xmin) / bw).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    # exact xmax -> last bin
    idx = np.where(boundaries >= xmax, bins - 1, idx)
    idx = np.where(boundaries <= xmin, 0, idx)
    return idx


def nearest_edge_indices(boundaries: np.ndarray, xmin: float, xmax: float, bins: int) -> np.ndarray:
    # idx = round((x-xmin)/(xmax-xmin) * bins)
    t = (boundaries - xmin) / (xmax - xmin)
    idx = np.rint(t * bins).astype(np.int64)
    return np.clip(idx, 0, bins)


def min_width(vals: List[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    return float(min(diffs))


def collision_map(indices: np.ndarray) -> Dict[int, int]:
    uniq, counts = np.unique(indices, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts) if c >= 2}


def enforce_strictly_increasing_indices(
    idx_near: np.ndarray,
    bins: int,
    fix_endpoints: bool,
    endpoint_tol: float,
    b0: float,
    bN: float,
    xmin: float,
    xmax: float,
) -> Tuple[np.ndarray, bool, Optional[str]]:
    """
    Make indices strictly increasing (idx[i] < idx[i+1]) by spreading duplicates.
    Returns (idx_final, changed, fail_reason).
    NOTE: With your new rule, we typically won't reach here for bin-collision heads.
    """
    n = idx_near.size
    idx = idx_near.copy()

    if fix_endpoints:
        if abs(b0 - xmin) <= endpoint_tol:
            idx[0] = 0
        if abs(bN - xmax) <= endpoint_tol:
            idx[-1] = bins

    if n > (bins + 1):
        return idx, False, f"too_many_boundaries_for_lut: n={n} > edges={bins+1}"

    for _ in range(5):
        for i in range(1, n):
            need = idx[i - 1] + 1
            if idx[i] < need:
                idx[i] = need

        for i in range(n - 2, -1, -1):
            need = idx[i + 1] - 1
            if idx[i] > need:
                idx[i] = need

        idx = np.clip(idx, 0, bins)

        if fix_endpoints:
            if abs(b0 - xmin) <= endpoint_tol:
                idx[0] = 0
            if abs(bN - xmax) <= endpoint_tol:
                idx[-1] = bins

        if np.all(idx[1:] > idx[:-1]):
            break

    if not np.all(idx[1:] > idx[:-1]):
        return idx, True, "cannot_make_strictly_increasing_after_spread"

    changed = not np.array_equal(idx, idx_near)
    return idx, changed, None


def get_chosen(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    chosen = entry.get("chosen")
    return chosen if isinstance(chosen, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--xmin", type=float, default=-20.0)
    ap.add_argument("--xmax", type=float, default=0.0)
    ap.add_argument("--fix_endpoints", action="store_true")
    ap.add_argument("--endpoint_tol", type=float, default=1e-6)
    args = ap.parse_args()

    with open(args.in_json, "r") as f:
        root = json.load(f)

    sel = root.get("selected", {})
    if not isinstance(sel, dict):
        raise ValueError("Input JSON missing top-level key: selected")

    edges = lut_edges(args.xmin, args.xmax, args.bins)
    bin_width = (args.xmax - args.xmin) / float(args.bins)

    # Summary counters
    heads_total = 0
    boundaries_total = 0

    ok = 0
    skipped = 0
    failed = 0
    missing = 0

    moved_total = 0
    max_move_bins_global = 0

    heads_with_bin_collisions = 0
    total_colliding_bins = 0
    max_bin_multiplicity = 1

    csv_rows: List[Dict[str, Any]] = []

    for Lk, heads in sel.items():
        if not (isinstance(Lk, str) and Lk.startswith("L") and isinstance(heads, dict)):
            continue

        for Hk, rec in heads.items():
            if not (isinstance(Hk, str) and Hk.startswith("H") and isinstance(rec, dict)):
                continue

            heads_total += 1
            row: Dict[str, Any] = {
                "L": Lk, "H": Hk,
                "selected_status": rec.get("status", ""),
                "outcome": "",  # ok|skip|fail|missing
                "bins": args.bins,
                "xmin": args.xmin,
                "xmax": args.xmax,
                "bin_width": bin_width,
                "segments": "",
                "n_boundaries": "",
                "min_width_orig": "",
                "min_width_nearest": "",
                "min_width_final": "",
                "bin_collision": "",
                "colliding_bins_count": "",
                "colliding_bins": "",
                "bin_multiplicity_map": "",
                "max_bin_multiplicity": "",
                "moved_count": "",
                "max_move_bins": "",
                "reason": "",
                "idx_nearest": "",
                "idx_final": "",
                "boundaries_orig": "",
                "boundaries_nearest": "",
                "boundaries_final": "",
            }

            chosen = get_chosen(rec)
            if chosen is None:
                missing += 1
                row["outcome"] = "missing"
                row["reason"] = "missing_chosen"
                csv_rows.append(row)
                continue

            boundaries = chosen.get("boundaries")
            if not isinstance(boundaries, list):
                failed += 1
                row["outcome"] = "fail"
                row["reason"] = "missing_or_invalid_boundaries"
                csv_rows.append(row)
                chosen["lut128_outcome"] = "fail"
                chosen["lut128_fail_reason"] = row["reason"]
                continue

            if not is_finite_list(boundaries):
                failed += 1
                row["outcome"] = "fail"
                row["reason"] = "non_finite_boundaries"
                csv_rows.append(row)
                chosen["lut128_outcome"] = "fail"
                chosen["lut128_fail_reason"] = row["reason"]
                continue

            b = [float(x) for x in boundaries]
            if len(b) < 2:
                failed += 1
                row["outcome"] = "fail"
                row["reason"] = "boundaries_too_short"
                csv_rows.append(row)
                chosen["lut128_outcome"] = "fail"
                chosen["lut128_fail_reason"] = row["reason"]
                continue

            if not strictly_increasing(b):
                failed += 1
                row["outcome"] = "fail"
                row["reason"] = "boundaries_not_strictly_increasing_in_input"
                csv_rows.append(row)
                chosen["lut128_outcome"] = "fail"
                chosen["lut128_fail_reason"] = row["reason"]
                continue

            S = chosen.get("segments", len(b) - 1)
            row["segments"] = int(S)
            row["n_boundaries"] = int(len(b))
            row["min_width_orig"] = min_width(b)
            row["boundaries_orig"] = json.dumps(b)

            b_np = np.asarray(b, dtype=np.float64)
            boundaries_total += b_np.size

            # ---------- NEW RULE: detect multi-boundaries-in-the-same-bin (interior boundaries) ----------
            bin_idx = bin_indices_for_boundaries(b_np, args.xmin, args.xmax, args.bins)
            # interior only (exclude endpoints, which are usually -20 and 0)
            if len(bin_idx) > 2:
                interior = bin_idx[1:-1]
            else:
                interior = np.array([], dtype=np.int64)

            bin_mult = collision_map(interior)  # bin->count for count>=2
            has_bin_collision = len(bin_mult) > 0

            row["bin_collision"] = bool(has_bin_collision)
            row["colliding_bins_count"] = int(len(bin_mult))
            row["colliding_bins"] = json.dumps(sorted(bin_mult.keys()))
            row["bin_multiplicity_map"] = json.dumps(bin_mult)
            row["max_bin_multiplicity"] = int(max(bin_mult.values()) if bin_mult else 1)

            if has_bin_collision:
                # SKIP snapping entirely for this head (your request)
                skipped += 1
                heads_with_bin_collisions += 1
                total_colliding_bins += len(bin_mult)
                max_bin_multiplicity = max(max_bin_multiplicity, row["max_bin_multiplicity"])

                row["outcome"] = "skip"
                row["reason"] = "skip_multi_boundaries_in_same_bin"
                row["boundaries_nearest"] = ""  # not computed (by choice)
                row["boundaries_final"] = json.dumps(b)  # unchanged
                row["min_width_nearest"] = ""
                row["min_width_final"] = row["min_width_orig"]
                row["idx_nearest"] = ""
                row["idx_final"] = ""
                row["moved_count"] = 0
                row["max_move_bins"] = 0
                csv_rows.append(row)

                # annotate JSON (keep original, plus collision stats)
                chosen["lut128_outcome"] = "skip"
                chosen["lut128_skip_reason"] = row["reason"]
                chosen["boundaries_lut128"] = b  # unchanged
                chosen["lut128_stats"] = {
                    "bin_width": bin_width,
                    "bin_collision": True,
                    "bin_multiplicity_map": bin_mult,
                    "colliding_bins": sorted(bin_mult.keys()),
                    "max_bin_multiplicity": row["max_bin_multiplicity"],
                    "min_width_orig": row["min_width_orig"],
                }
                continue

            # ---------- Otherwise: proceed with nearest-edge snapping ----------
            idx_near = nearest_edge_indices(b_np, args.xmin, args.xmax, args.bins)
            b_near = edges[idx_near].astype(np.float64)

            row["idx_nearest"] = json.dumps(idx_near.tolist())
            row["boundaries_nearest"] = json.dumps(b_near.tolist())
            row["min_width_nearest"] = min_width(b_near.tolist())

            # enforce strictly increasing in edge indices (spread if necessary)
            idx_final, changed, fail_reason = enforce_strictly_increasing_indices(
                idx_near=idx_near,
                bins=args.bins,
                fix_endpoints=args.fix_endpoints,
                endpoint_tol=args.endpoint_tol,
                b0=b[0], bN=b[-1],
                xmin=args.xmin, xmax=args.xmax,
            )

            if fail_reason is not None:
                # This is now a real "fail" (since it wasn't a bin-collision skip)
                failed += 1
                row["outcome"] = "fail"
                row["reason"] = fail_reason
                b_final = edges[idx_final].astype(np.float64)
                row["idx_final"] = json.dumps(idx_final.tolist())
                row["boundaries_final"] = json.dumps(b_final.tolist())
                row["min_width_final"] = min_width(b_final.tolist())
                csv_rows.append(row)

                chosen["lut128_outcome"] = "fail"
                chosen["lut128_fail_reason"] = fail_reason
                chosen["boundaries_lut128_nearest_idx"] = idx_near.tolist()
                chosen["boundaries_lut128_nearest"] = b_near.tolist()
                chosen["boundaries_lut128_idx"] = idx_final.tolist()
                chosen["boundaries_lut128"] = b_final.tolist()
                chosen["lut128_stats"] = {
                    "bin_width": bin_width,
                    "bin_collision": False,
                    "moved_count": int(np.count_nonzero(idx_final - idx_near)),
                    "max_move_bins": int(np.max(np.abs(idx_final - idx_near))) if idx_final.size else 0,
                    "min_width_orig": row["min_width_orig"],
                    "min_width_nearest": row["min_width_nearest"],
                    "min_width_final": row["min_width_final"],
                }
                continue

            # OK
            ok += 1
            b_final = edges[idx_final].astype(np.float64)

            move = idx_final - idx_near
            moved_count = int(np.count_nonzero(move))
            max_move_bins = int(np.max(np.abs(move))) if move.size else 0
            moved_total += moved_count
            max_move_bins_global = max(max_move_bins_global, max_move_bins)

            row["outcome"] = "ok"
            row["reason"] = ""
            row["idx_final"] = json.dumps(idx_final.tolist())
            row["boundaries_final"] = json.dumps(b_final.tolist())
            row["min_width_final"] = min_width(b_final.tolist())
            row["moved_count"] = moved_count
            row["max_move_bins"] = max_move_bins
            csv_rows.append(row)

            chosen["lut128_outcome"] = "ok"
            chosen["boundaries_lut128_nearest_idx"] = idx_near.tolist()
            chosen["boundaries_lut128_nearest"] = b_near.tolist()
            chosen["boundaries_lut128_idx"] = idx_final.tolist()
            chosen["boundaries_lut128"] = b_final.tolist()
            chosen["lut128_stats"] = {
                "bin_width": bin_width,
                "bin_collision": False,
                "moved_count": moved_count,
                "max_move_bins": max_move_bins,
                "min_width_orig": row["min_width_orig"],
                "min_width_nearest": row["min_width_nearest"],
                "min_width_final": row["min_width_final"],
            }

    # Write JSON
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(root, f, indent=2)

    # Write CSV
    fieldnames = [
        "L","H","selected_status","outcome",
        "bins","xmin","xmax","bin_width",
        "segments","n_boundaries",
        "min_width_orig","min_width_nearest","min_width_final",
        "bin_collision","colliding_bins_count","colliding_bins","bin_multiplicity_map","max_bin_multiplicity",
        "moved_count","max_move_bins",
        "reason",
        "idx_nearest","idx_final",
        "boundaries_orig","boundaries_nearest","boundaries_final",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    print(
        f"[lut{args.bins}] heads={heads_total} boundaries={boundaries_total} "
        f"ok={ok} skip={skipped} fail={failed} missing={missing} "
        f"| moved={moved_total} max_move_bins={max_move_bins_global} "
        f"| heads_with_bin_collisions={heads_with_bin_collisions} "
        f"total_colliding_bins={total_colliding_bins} max_bin_multiplicity={max_bin_multiplicity}"
    )
    print(f"[write] {args.out_json}")
    print(f"[write] {args.out_csv}")


if __name__ == "__main__":
    main()
