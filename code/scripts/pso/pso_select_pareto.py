#!/usr/bin/env python3
import argparse, json, math, csv, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def pick_top1(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    entries is the list of top-k (usually 5) dicts for a given (S,L,H).
    Pick the one with smallest 'rank'. If missing, fall back to smallest 'item.rank'.
    """
    if not entries:
        return None

    def rank_of(e):
        r = e.get("rank")
        if isinstance(r, (int, float)):
            return r
        ir = safe_get(e, ["item", "rank"], None)
        if isinstance(ir, (int, float)):
            return ir
        return 10**9

    return min(entries, key=rank_of)

def compute_cost(expected_deg: float, segments: int) -> float:
    # cost = expected_degree + log2(no. of segment)
    return float(expected_deg) + math.log2(int(segments))

def pareto_front(points: List[Dict[str, Any]], kl_key: str) -> List[Dict[str, Any]]:
    """
    Each point has fields: 'kl', 'cost', etc.
    Pareto front for minimizing (cost, kl).
    A dominates B if cost<= and kl<= with at least one strict.
    """
    front = []
    for a in points:
        dominated = False
        for b in points:
            if a is b:
                continue
            if (b["cost"] <= a["cost"] and b["kl"] <= a["kl"]) and (b["cost"] < a["cost"] or b["kl"] < a["kl"]):
                dominated = True
                break
        if not dominated:
            front.append(a)

    # sort front for nicer viewing: increasing cost, then KL
    front.sort(key=lambda x: (x["cost"], x["kl"]))
    return front

def choose_from_front(front: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Default policy: min cost, tie-breaker min KL.
    (You can later change this to 'knee' selection if you want.)
    """
    if not front:
        return None
    return min(front, key=lambda x: (x["cost"], x["kl"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Path to PSO_WIKI_v1_*.json (the big summary JSON)")
    ap.add_argument("--kl_key", default="kl_p95_u16",
                    choices=["kl_mean_u16", "kl_p95_u16", "kl_max_u16"],
                    help="Which KL metric to threshold and Pareto against")
    ap.add_argument("--kl_thr", type=float, required=True, help="KL divergence threshold (keep candidates with KL <= thr)")
    ap.add_argument("--segments", default="S04,S08,S12,S16",
                    help="Comma-separated segments groups to consider (e.g. S04,S08,S12,S16)")
    ap.add_argument("--out_json", default="workdir/pareto/selected_pareto.json", help="Output JSON with selected per (L,H)")
    ap.add_argument("--out_csv", default="workdir/pareto/selected_pareto.csv", help="Output CSV (easy to grep/sort)")
    ap.add_argument("--dump_pareto_json", default=None,
                    help="Optional: write full Pareto fronts per (L,H) to this JSON")
    args = ap.parse_args()

    seg_groups = [s.strip() for s in args.segments.split(",") if s.strip()]

    with open(args.in_json, "r") as f:
        root = json.load(f)

    tree = root.get("results_tree", {})
    if not isinstance(tree, dict):
        print("ERROR: results_tree missing or not an object", file=sys.stderr)
        sys.exit(2)

    selected: Dict[str, Any] = {
        "meta": root.get("meta", {}),
        "selection_policy": {
            "kl_key": args.kl_key,
            "kl_thr": args.kl_thr,
            "cost": "expected_deg + log2(segments)",
            "segments_considered": seg_groups,
            "choose_from_pareto": "min cost, tie-breaker min KL",
        },
        "selected": {}  # selected[L][H] = chosen
    }

    pareto_dump = {}  # pareto_dump[L][H] = [front points]

    # Iterate over segments group -> layers -> heads
    # We’ll build per (L,H) candidate list from the 4 segments
    candidates_by_lh: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for S in seg_groups:
        if S not in tree:
            continue
        layers = tree[S]
        if not isinstance(layers, dict):
            continue
        for L, heads in layers.items():
            if not isinstance(heads, dict):
                continue
            for H, entries in heads.items():
                if not isinstance(entries, list):
                    continue

                top1 = pick_top1(entries)
                if top1 is None:
                    continue

                item = top1.get("item", {})
                if not isinstance(item, dict):
                    item = {}

                # Extract metrics
                kl = item.get(args.kl_key, None)
                exp_deg = item.get("expected_deg", None)

                if kl is None or exp_deg is None:
                    continue

                try:
                    kl = float(kl)
                    exp_deg = float(exp_deg)
                except Exception:
                    continue

                seg_int = int(top1.get("segments", S.replace("S", "")))
                cost = compute_cost(exp_deg, seg_int)

                pt = {
                    "segments": seg_int,
                    "S": S,
                    "L": L,
                    "H": H,
                    "rank": top1.get("rank", item.get("rank", None)),
                    "kl": kl,
                    "expected_deg": exp_deg,
                    "cost": cost,
                    "deg": item.get("deg", None),
                    "boundaries": item.get("boundaries", None),
                    "min_width": item.get("min_width", None),
                    "max_width": item.get("max_width", None),
                    "objective_f": item.get("objective_f", None),
                    "pbest_cost": item.get("pbest_cost", None),
                    "flip_rate": item.get("flip_rate", None),
                    "dens_pen_raw": item.get("dens_pen_raw", None),
                    "per_run_json": top1.get("per_run_json", None),
                    "log": top1.get("log", None),
                    "source": top1.get("source", None),
                }

                candidates_by_lh.setdefault((L, H), []).append(pt)

    # Now decide per (L,H)
    for (L, H), pts in candidates_by_lh.items():
        # Require we have at least 1 candidate (ideally 4)
        # Apply KL threshold
        feasible = [p for p in pts if p["kl"] <= args.kl_thr]

        # If nothing feasible, keep Pareto anyway (no threshold) and mark as failed
        used_pts = feasible if feasible else pts

        front = pareto_front(used_pts, args.kl_key)
        choice = choose_from_front(front)

        selected["selected"].setdefault(L, {})
        if choice is None:
            selected["selected"][L][H] = {
                "status": "missing",
                "reason": "no candidates found"
            }
            continue

        selected["selected"][L][H] = {
            "status": "ok" if feasible else "no_feasible_under_threshold",
            "chosen": choice,
            "candidates_count": len(pts),
            "feasible_count": len(feasible),
        }

        pareto_dump.setdefault(L, {})
        pareto_dump[L][H] = front

    # Write outputs
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(selected, f, indent=2)

    # CSV flatten
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "L","H","status","segments","cost","kl","expected_deg","rank",
            "deg","boundaries","min_width","max_width","objective_f","pbest_cost",
            "flip_rate","dens_pen_raw","per_run_json","log","source"
        ])
        for L in sorted(selected["selected"].keys()):
            for H in sorted(selected["selected"][L].keys()):
                rec = selected["selected"][L][H]
                status = rec.get("status")
                chosen = rec.get("chosen", {})
                w.writerow([
                    L, H, status,
                    chosen.get("segments"),
                    chosen.get("cost"),
                    chosen.get("kl"),
                    chosen.get("expected_deg"),
                    chosen.get("rank"),
                    json.dumps(chosen.get("deg")),
                    json.dumps(chosen.get("boundaries")),
                    chosen.get("min_width"),
                    chosen.get("max_width"),
                    chosen.get("objective_f"),
                    chosen.get("pbest_cost"),
                    chosen.get("flip_rate"),
                    chosen.get("dens_pen_raw"),
                    chosen.get("per_run_json"),
                    chosen.get("log"),
                    chosen.get("source"),
                ])

    if args.dump_pareto_json:
        Path(args.dump_pareto_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_pareto_json, "w") as f:
            json.dump({
                "kl_key": args.kl_key,
                "kl_thr": args.kl_thr,
                "segments_considered": seg_groups,
                "pareto_fronts": pareto_dump
            }, f, indent=2)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_csv}")
    if args.dump_pareto_json:
        print(f"Wrote: {args.dump_pareto_json}")

if __name__ == "__main__":
    main()
