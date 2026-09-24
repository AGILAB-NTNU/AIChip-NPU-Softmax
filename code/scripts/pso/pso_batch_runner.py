#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


# -------------------------
# Parsing helpers
# -------------------------
def parse_multi_range(s: str) -> List[int]:
    """
    Accepts:
      "0-4"        -> [0..4]
      "7"          -> [7]
      "0-2,5,9-10" -> [0,1,2,5,9,10]
    """
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a); b = int(b)
            if b < a:
                raise ValueError(f"Bad range token: {part}")
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_int_list(s: str) -> List[int]:
    return sorted(set(int(x.strip()) for x in (s or "").split(",") if x.strip()))


def bin_path_for(input_dir: Path, layer: int, head: int) -> Path:
    return input_dir / f"layer{layer:02d}_head{head:02d}_add2_u16.bin"


def fmt_job(layer: int, head: int, seg: int) -> str:
    return f"L{layer:02d}_H{head:02d}_S{seg:02d}"


def key_job(seg: int, layer: int, head: int) -> Tuple[int, int, int]:
    return (int(seg), int(layer), int(head))


# -------------------------
# Job definition
# -------------------------
@dataclass(frozen=True)
class Job:
    pso_script: Path
    bin_path: Path
    encoding_json: Path
    layer: int
    head: int
    K: int
    Q: int
    segments: int
    deg_max: int
    out_json: Path
    log_path: Path
    common_args: List[str]
    env_vars: Dict[str, str]


# -------------------------
# Per-run JSON validation + extraction
# -------------------------
def is_valid_per_run_json(path: Path) -> Tuple[bool, str]:
    """
    Returns (ok, reason). ok only if:
      - json parses
      - has "topk" list with at least 1 item
      - topk[0] has boundaries + deg (basic sanity)
    """
    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except Exception as e:
        return (False, f"json parse error: {e}")

    if not isinstance(obj, dict):
        return (False, "json root is not a dict")

    topk = obj.get("topk", None)
    if not isinstance(topk, list) or len(topk) == 0:
        return (False, "missing/empty topk")

    it0 = topk[0]
    if not isinstance(it0, dict):
        return (False, "topk[0] not a dict")
    if "boundaries" not in it0 or "deg" not in it0:
        return (False, "topk[0] missing boundaries/deg")
    if not isinstance(it0["boundaries"], list) or not isinstance(it0["deg"], list):
        return (False, "topk[0] boundaries/deg not lists")

    return (True, "")


def extract_results_from_per_run_json(
    per_run_path: Path,
    layer: int,
    head: int,
    segments: int,
    log_path: Optional[Path],
    source: str,  # "existing" or "new"
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Returns (flat_entries, error_str).
    Each entry keeps the FULL 'item' dict under 'item' so you never lose fields.
    """
    try:
        with open(per_run_path, "r") as f:
            obj = json.load(f)
    except Exception as e:
        return ([], f"failed reading {per_run_path}: {e}")

    topk = obj.get("topk", [])
    if not isinstance(topk, list):
        return ([], f"topk is not a list in {per_run_path}")

    out: List[Dict[str, Any]] = []
    for item in topk:
        rank = -1
        if isinstance(item, dict):
            try:
                rank = int(item.get("rank", -1))
            except Exception:
                rank = -1

        out.append({
            "segments": int(segments),
            "layer": int(layer),
            "head": int(head),
            "rank": int(rank),
            "per_run_json": str(per_run_path),
            "log": str(log_path) if log_path is not None else None,
            "source": source,
            "item": item,  # FULL payload
        })

    return (out, None)


# -------------------------
# Atomic JSON write (safe checkpoints)
# -------------------------
def write_json_atomic(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# -------------------------
# Resume report (what exists + what will run next)
# -------------------------
def print_resume_report(
    run_tag: str,
    per_run_root: Path,
    seg_list: List[int],
    expected_keys: List[Tuple[int, int, int]],  # (S,L,H)
    done_valid: set,
    invalid_existing: Dict[Tuple[int, int, int], str],
    pending_keys: List[Tuple[int, int, int]],
    rerun: bool,
):
    exp_total = len(expected_keys)
    done_total = len(done_valid)
    invalid_total = len(invalid_existing)
    pending_total = len(pending_keys)

    print("\n================= RESUME / STATUS =================")
    print(f"[runner] run_tag:      {run_tag}")
    print(f"[runner] per-run root: {per_run_root}")
    print(f"[runner] expected jobs (bin exists): {exp_total}")
    print(f"[runner] valid existing json:        {done_total}")
    if invalid_total:
        print(f"[runner] INVALID existing json:      {invalid_total}  (will rerun these)")
    print(f"[runner] --rerun:                   {rerun}")
    if rerun:
        print(f"[runner] pending now (rerun all):   {pending_total}")
    else:
        print(f"[runner] pending now (resume):      {pending_total}")

    # per-segment breakdown
    for S in seg_list:
        expS = [k for k in expected_keys if k[0] == S]
        doneS = [k for k in done_valid if k[0] == S]
        invS  = [k for k in invalid_existing.keys() if k[0] == S]
        pendS = [k for k in pending_keys if k[0] == S]
        msg = f"  S{S:02d}: done {len(doneS)}/{len(expS)}  pending {len(pendS)}"
        if invS:
            msg += f"  (invalid {len(invS)})"
        print(msg)

    if pending_keys:
        S, L, H = pending_keys[0]
        print(f"[runner] next pending (sorted): S{S:02d} L{L:02d} H{H:02d}")
    else:
        print(f"[runner] next pending (sorted): - (nothing to run)")

    # quick examples
    done_ex = sorted(list(done_valid))[:5]
    pend_ex = pending_keys[:5]
    if done_ex:
        print("[runner] examples already done:")
        for S, L, H in done_ex:
            print(f"   - S{S:02d} L{L:02d} H{H:02d}")
    if pend_ex:
        print("[runner] examples pending:")
        for S, L, H in pend_ex:
            print(f"   - S{S:02d} L{L:02d} H{H:02d}")

    print("===================================================\n")


# -------------------------
# Run pso_optimize.py (single job)
# -------------------------
def run_one(job: Job) -> Tuple[bool, str, str, str]:
    """
    Runs pso_optimize.py and returns (ok, out_json, log_path, error_msg)
    """
    cmd = [
        "python3", str(job.pso_script),
        "--add2-u16-bin", str(job.bin_path),
        "--encoding-json", str(job.encoding_json),
        "--layer-idx", str(job.layer),
        "--K", str(job.K),
        "--Q", str(job.Q),
        "--segments", str(job.segments),
        "--deg-max", str(job.deg_max),

        # export topk for runner
        "--stage2-topk", "5",
        "--stage2-json-out", str(job.out_json),
    ] + list(job.common_args)

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.out_json.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(job.env_vars)

    with open(job.log_path, "w") as lf:
        lf.write("CMD:\n" + " ".join(cmd) + "\n\n")
        lf.flush()
        p = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    if p.returncode != 0:
        return (False, str(job.out_json), str(job.log_path), f"pso_optimize failed rc={p.returncode}")

    if not job.out_json.exists():
        return (False, str(job.out_json), str(job.log_path), "pso_optimize did not produce stage2 json")

    ok, reason = is_valid_per_run_json(job.out_json)
    if not ok:
        return (False, str(job.out_json), str(job.log_path), f"pso_optimize produced invalid json: {reason}")

    return (True, str(job.out_json), str(job.log_path), "")


# -------------------------
# Progress monitor (running jobs + ETA)
# -------------------------
class ProgressState:
    def __init__(self, total_pending: int, jobs_parallel: int):
        self.total_pending = total_pending
        self.jobs_parallel = max(1, int(jobs_parallel))
        self.done = 0
        self.running: Dict[str, float] = {}   # job_name -> start_time
        self.durations: List[float] = []
        self.lock = threading.Lock()

    def estimate_eta_seconds(self) -> Optional[float]:
        with self.lock:
            if self.done <= 0 or len(self.durations) == 0:
                return None
            avg = sum(self.durations) / max(1, len(self.durations))
            remaining = self.total_pending - self.done
            # crude parallel ETA (better than single-thread ETA)
            return max(0.0, (remaining * avg) / float(self.jobs_parallel))

    def running_preview(self, max_items: int = 3) -> str:
        with self.lock:
            items = list(self.running.keys())
        if not items:
            return "-"
        items = items[:max_items]
        if len(items) < len(self.running):
            return ", ".join(items) + ", ..."
        return ", ".join(items)


def monitor_thread_fn(state: ProgressState, stop_evt: threading.Event, bar=None, refresh_sec: float = 1.0):
    last_print = 0.0
    while not stop_evt.is_set():
        eta = state.estimate_eta_seconds()
        running = state.running_preview(max_items=3)

        if _HAVE_TQDM and bar is not None:
            postfix = {"running": running}
            if eta is not None:
                finish = datetime.now() + timedelta(seconds=float(eta))
                postfix["eta"] = str(timedelta(seconds=int(eta)))
                postfix["finish"] = finish.strftime("%H:%M:%S")
            bar.set_postfix(postfix, refresh=False)
            bar.refresh()
        else:
            now = time.time()
            if now - last_print > 5.0:
                last_print = now
                with state.lock:
                    done = state.done
                    total = state.total_pending
                if eta is not None:
                    finish = datetime.now() + timedelta(seconds=float(eta))
                    print(f"[runner][progress] {done}/{total} done | running: {running} | ETA {timedelta(seconds=int(eta))} | finish ~{finish.strftime('%H:%M:%S')}")
                else:
                    print(f"[runner][progress] {done}/{total} done | running: {running}")

        stop_evt.wait(refresh_sec)


# -------------------------
# Final object builder
# -------------------------
def build_final_object(
    run_tag: str,
    stamp: str,
    args,
    layers: List[int],
    heads: List[int],
    seg_list: List[int],
    input_dir: Path,
    pso_script: Path,
    encoding_json: Path,
    common_args: List[str],
    env_vars: Dict[str, str],
    counts: Dict[str, int],
    missing_bins: List[str],
    invalid_existing_detected: List[Dict[str, Any]],
    results_flat: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    logs_root: Path,
    per_run_root: Path,
) -> Dict[str, Any]:
    # hierarchical convenience view:
    # results_tree["S16"]["L02"]["H00"] = list of entries (rank-sorted)
    results_tree: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

    for r in results_flat:
        S = int(r["segments"])
        L = int(r["layer"])
        H = int(r["head"])
        seg_key = f"S{S:02d}"
        lay_key = f"L{L:02d}"
        head_key = f"H{H:02d}"
        results_tree.setdefault(seg_key, {}).setdefault(lay_key, {}).setdefault(head_key, []).append(r)

    for seg_key in results_tree:
        for lay_key in results_tree[seg_key]:
            for head_key in results_tree[seg_key][lay_key]:
                results_tree[seg_key][lay_key][head_key].sort(key=lambda x: int(x.get("rank", 0)))

    return {
        "meta": {
            "run_tag": run_tag,
            "run_started_stamp": stamp,
            "layers_arg": args.layers,
            "heads_arg": args.heads,
            "segments_arg": args.segments,
            "layers": layers,
            "heads": heads,
            "segments": seg_list,
            "K": int(args.K),
            "Q": int(args.Q),
            "deg_max": int(args.deg_max),
            "jobs_parallel": int(args.jobs),
            "rerun": bool(args.rerun),
            "pso_script": str(pso_script),
            "encoding_json": str(encoding_json),
            "input_dir": str(input_dir),
            "common_args": common_args,
            "env_vars": env_vars,
            "counts": counts,
            "layout": {
                "per_run": "per_run/S{segments:02d}/{run_tag}/Lxx_Hyy_Szz.json",
                "logs": "logs/{run_tag}__{stamp}/S{segments:02d}/Lxx_Hyy_Szz.log",
                "logs_root": str(logs_root),
                "per_run_root": str(per_run_root),
            },
        },
        "missing_bins": missing_bins,
        "invalid_existing_detected": invalid_existing_detected,  # what was found before rerun
        "results_flat": results_flat,     # every topk entry, full payload under "item"
        "results_tree": results_tree,     # hierarchical view
        "failures": failures,             # actual failures during this run / load
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pso-script", required=True)
    ap.add_argument("--encoding-json", required=True)

    ap.add_argument("--layers", required=True, help='e.g. "0-21" or "0-2,5,7-9"')
    ap.add_argument("--heads", default="0-31", help='e.g. "0-31" or "0-2,7"')

    ap.add_argument("--segments", default="16,12,8,4", help="Comma list of segment counts")
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--Q", type=int, default=128)
    ap.add_argument("--deg-max", type=int, default=5)

    ap.add_argument("--jobs", type=int, default=6, help="Max concurrent PSO processes")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rerun", action="store_true", help="Rerun even if per-run json exists")

    ap.add_argument("--tag", type=str, default=None,
                    help="Fixed tag. If set, no timestamp is appended to tag.")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="Do not append timestamp to run_tag (so outputs are reused).")

    ap.add_argument("--refresh-sec", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="Write .partial.json every N completed pending jobs")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pso_script = Path(args.pso_script)
    encoding_json = Path(args.encoding_json)
    out_dir = Path(args.out_dir)

    layers = parse_multi_range(args.layers)
    heads  = parse_multi_range(args.heads)
    seg_list = parse_int_list(args.segments)

    if not layers:
        raise ValueError("No layers parsed.")
    if not heads:
        raise ValueError("No heads parsed.")
    if not seg_list:
        raise ValueError("No segments parsed.")

    # PSO knobs (your provided command)
    common_args = [
        "--lam-rob", "10.0", "--rob-p", "99.9", "--lam-max", "5.0",
        "--lam-flip", "10.0",
        "--lam-cost", "0.001",
        "--lam-dens", "0.01",
        "--tiny-width-penalty", "10.0",
        "--no-smart-init",
        # optionally add "--no-split"
    ]

    # IMPORTANT for multi-process: stop numpy/blas from spawning threads per process
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_tag = f"L{layers[0]:02d}-{layers[-1]:02d}_H{heads[0]:02d}-{heads[-1]:02d}"

    if args.tag:
        run_tag = args.tag
        tag_has_timestamp = False
    elif args.no_timestamp:
        run_tag = base_tag
        tag_has_timestamp = False
    else:
        run_tag = f"{base_tag}_{stamp}"
        tag_has_timestamp = True

    # IMPORTANT NOTE (this is what was confusing earlier):
    # If run_tag changes every run (timestamped), you create a NEW per_run folder,
    # so it cannot "resume" from old results unless you reuse the same run_tag.
    if tag_has_timestamp and (not args.rerun):
        print("[runner][NOTE] run_tag includes a timestamp, so this run will write to a NEW per_run folder.")
        print("              To RESUME into the SAME per_run folder, use --no-timestamp or --tag <fixed>.")
        print("")

    # Layout:
    # per_run/S16/<run_tag>/Lxx_Hyy_S16.json   (stable for resume)
    # logs/<run_tag>__<stamp>/S16/Lxx_Hyy_S16.log   (timestamped; safe to re-run)
    per_run_root = out_dir / "per_run"
    logs_root = out_dir / "logs" / (run_tag + "__" + stamp)

    final_json_path_stable   = out_dir / f"PSO_{run_tag}.json"
    final_json_path_snapshot = out_dir / f"PSO_{run_tag}_snapshot_{stamp}.json"
    final_json_path_partial  = out_dir / f"PSO_{run_tag}.partial.json"

    # -------------------------
    # Scan what exists (resume logic)
    # -------------------------
    expected_keys: List[Tuple[int, int, int]] = []  # (S,L,H) only if bin exists
    done_valid: set = set()  # set of (S,L,H) with valid per_run json
    invalid_existing: Dict[Tuple[int, int, int], str] = {}  # (S,L,H)->reason
    missing_bins: List[str] = []

    for S in seg_list:
        for L in layers:
            for H in heads:
                bin_path = bin_path_for(input_dir, L, H)
                if not bin_path.exists():
                    # only count missing once per (L,H), but keeping simple list is fine
                    missing_bins.append(str(bin_path))
                    continue

                expected_keys.append((S, L, H))

                per_run_dir = per_run_root / f"S{S:02d}" / run_tag
                out_json = per_run_dir / f"L{L:02d}_H{H:02d}_S{S:02d}.json"

                if out_json.exists():
                    ok, reason = is_valid_per_run_json(out_json)
                    if ok:
                        done_valid.add((S, L, H))
                    else:
                        invalid_existing[(S, L, H)] = reason

    expected_keys = sorted(expected_keys)  # sorted by (S,L,H)

    # Determine pending keys
    if args.rerun:
        pending_keys = expected_keys[:]  # rerun all expected
        skip_count = 0
        rerun_invalid_count = len(invalid_existing)
    else:
        pending_keys = [k for k in expected_keys if k not in done_valid]  # includes invalid + missing json
        skip_count = len(done_valid)
        rerun_invalid_count = len(invalid_existing)

    pending_keys = sorted(pending_keys)

    # Print a clear resume report
    print_resume_report(
        run_tag=run_tag,
        per_run_root=per_run_root,
        seg_list=seg_list,
        expected_keys=expected_keys,
        done_valid=done_valid,
        invalid_existing=invalid_existing,
        pending_keys=pending_keys,
        rerun=bool(args.rerun),
    )

    # Build pending Job objects
    pending_jobs: List[Job] = []
    for (S, L, H) in pending_keys:
        bin_path = bin_path_for(input_dir, L, H)

        per_run_dir = per_run_root / f"S{S:02d}" / run_tag
        out_json = per_run_dir / f"L{L:02d}_H{H:02d}_S{S:02d}.json"

        log_dir = logs_root / f"S{S:02d}"
        log_path = log_dir / f"L{L:02d}_H{H:02d}_S{S:02d}.log"

        pending_jobs.append(Job(
            pso_script=pso_script,
            bin_path=bin_path,
            encoding_json=encoding_json,
            layer=L,
            head=H,
            K=int(args.K),
            Q=int(args.Q),
            segments=S,
            deg_max=int(args.deg_max),
            out_json=out_json,
            log_path=log_path,
            common_args=common_args,
            env_vars=env_vars,
        ))

    # Deterministic job order (helpful for “where will it start”)
    pending_jobs.sort(key=lambda j: (int(j.segments), int(j.layer), int(j.head)))

    # Print “continuation” in plain words
    if not args.rerun:
        if skip_count > 0:
            print(f"[runner] Skipping {skip_count} jobs because valid per_run JSON already exists (resume).")
        if rerun_invalid_count > 0:
            print(f"[runner] Rerunning {rerun_invalid_count} jobs because existing JSON is invalid/corrupt.")
    else:
        print(f"[runner] --rerun enabled: running ALL expected jobs ({len(pending_jobs)} pending).")

    if pending_jobs:
        j0 = pending_jobs[0]
        print(f"[runner] Will start from: {fmt_job(j0.layer, j0.head, j0.segments)}")
    else:
        print("[runner] No pending jobs. (Everything is already completed & valid.)")

    print(f"[runner] pending jobs:       {len(pending_jobs)} (jobs_parallel={args.jobs})")
    print(f"[runner] per-run root:       {per_run_root}   (segment-separated)")
    print(f"[runner] logs root:          {logs_root}      (timestamped)")
    print(f"[runner] final stable:       {final_json_path_stable}")
    print(f"[runner] final snapshot:     {final_json_path_snapshot}")
    print(f"[runner] final partial:      {final_json_path_partial}")
    print("")

    # -------------------------
    # Load existing results into a map (so final JSON always contains ALL known values)
    # (Dedup by (S,L,H,rank); newer overwrites older)
    # -------------------------
    results_map: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    invalid_existing_detected: List[Dict[str, Any]] = [
        {"segments": k[0], "layer": k[1], "head": k[2], "reason": v}
        for k, v in sorted(invalid_existing.items())
    ]

    def load_one_existing(S: int, L: int, H: int) -> None:
        per_run_dir = per_run_root / f"S{S:02d}" / run_tag
        out_json = per_run_dir / f"L{L:02d}_H{H:02d}_S{S:02d}.json"
        ok, _ = is_valid_per_run_json(out_json)
        if not ok:
            return
        entries, err = extract_results_from_per_run_json(
            out_json, layer=L, head=H, segments=S, log_path=None, source="existing"
        )
        if err is not None:
            failures.append({"out_json": str(out_json), "log": None, "error": err})
            return
        for e in entries:
            k = (int(e["segments"]), int(e["layer"]), int(e["head"]), int(e["rank"]))
            results_map[k] = e

    # Load existing valid results (unless rerun; but still fine—new results overwrite)
    if done_valid:
        load_list = sorted(list(done_valid))
        if _HAVE_TQDM:
            it = tqdm(load_list, desc="Loading existing", unit="file", dynamic_ncols=True)
        else:
            it = load_list
        for (S, L, H) in it:
            load_one_existing(S, L, H)

    # -------------------------
    # Progress state for pending jobs
    # -------------------------
    state = ProgressState(total_pending=len(pending_jobs), jobs_parallel=int(args.jobs))
    stop_evt = threading.Event()

    bar = None
    if _HAVE_TQDM:
        bar = tqdm(total=len(pending_jobs), desc="PSO runs", unit="job", dynamic_ncols=True)

    monitor = threading.Thread(
        target=monitor_thread_fn,
        args=(state, stop_evt, bar, float(args.refresh_sec)),
        daemon=True
    )
    monitor.start()

    def run_one_monitored(job: Job) -> Tuple[bool, Job, str, str, str]:
        name = fmt_job(job.layer, job.head, job.segments)
        t0 = time.time()
        with state.lock:
            state.running[name] = t0

        ok, out_json, log_path, err = run_one(job)

        t1 = time.time()
        with state.lock:
            state.running.pop(name, None)
            state.durations.append(t1 - t0)
            state.done += 1

        return (ok, job, out_json, log_path, err)

    # -------------------------
    # Run pending jobs + checkpoints
    # -------------------------
    completed_since_ckpt = 0
    CKPT_EVERY = max(1, int(args.ckpt_every))

    try:
        with cf.ThreadPoolExecutor(max_workers=int(args.jobs)) as ex:
            futs = [ex.submit(run_one_monitored, j) for j in pending_jobs]

            for fut in cf.as_completed(futs):
                ok, job, out_json, log_path, err = fut.result()
                name = fmt_job(job.layer, job.head, job.segments)

                if bar is not None:
                    bar.update(1)
                    bar.set_postfix({"last": name}, refresh=False)

                if not ok:
                    failures.append({"out_json": out_json, "log": log_path, "error": err})
                    print(f"[runner][FAIL] {out_json}  ({err})")
                else:
                    entries, e = extract_results_from_per_run_json(
                        Path(out_json),
                        layer=job.layer,
                        head=job.head,
                        segments=job.segments,
                        log_path=Path(log_path),
                        source="new",
                    )
                    if e is not None:
                        failures.append({"out_json": out_json, "log": log_path, "error": e})
                        print(f"[runner][FAIL] {out_json}  ({e})")
                    else:
                        for ent in entries:
                            k = (int(ent["segments"]), int(ent["layer"]), int(ent["head"]), int(ent["rank"]))
                            results_map[k] = ent
                        print(f"[runner][OK] {out_json}")

                completed_since_ckpt += 1
                if completed_since_ckpt >= CKPT_EVERY:
                    completed_since_ckpt = 0
                    final_partial = finalize_now(
                        run_tag=run_tag,
                        stamp=stamp,
                        args=args,
                        layers=layers,
                        heads=heads,
                        seg_list=seg_list,
                        input_dir=input_dir,
                        pso_script=pso_script,
                        encoding_json=encoding_json,
                        common_args=common_args,
                        env_vars=env_vars,
                        per_run_root=per_run_root,
                        logs_root=logs_root,
                        expected_keys=expected_keys,
                        pending_jobs_count=len(pending_jobs),
                        done_valid_count=len(done_valid),
                        skip_count=skip_count,
                        rerun_invalid_count=rerun_invalid_count,
                        invalid_existing_detected=invalid_existing_detected,
                        missing_bins=missing_bins,
                        results_map=results_map,
                        failures=failures,
                    )
                    write_json_atomic(final_json_path_partial, final_partial)

    finally:
        stop_evt.set()
        monitor.join(timeout=2.0)
        if bar is not None:
            bar.close()

    # -------------------------
    # Finalize (stable + snapshot)
    # -------------------------
    final_obj = finalize_now(
        run_tag=run_tag,
        stamp=stamp,
        args=args,
        layers=layers,
        heads=heads,
        seg_list=seg_list,
        input_dir=input_dir,
        pso_script=pso_script,
        encoding_json=encoding_json,
        common_args=common_args,
        env_vars=env_vars,
        per_run_root=per_run_root,
        logs_root=logs_root,
        expected_keys=expected_keys,
        pending_jobs_count=len(pending_jobs),
        done_valid_count=len(done_valid),
        skip_count=skip_count,
        rerun_invalid_count=rerun_invalid_count,
        invalid_existing_detected=invalid_existing_detected,
        missing_bins=missing_bins,
        results_map=results_map,
        failures=failures,
    )

    write_json_atomic(final_json_path_stable, final_obj)
    write_json_atomic(final_json_path_snapshot, final_obj)

    print(f"\n[runner] wrote final stable:    {final_json_path_stable}")
    print(f"[runner] wrote final snapshot: {final_json_path_snapshot}")
    print(f"[runner] results_flat={len(final_obj.get('results_flat', []))} failures={len(final_obj.get('failures', []))}")


def finalize_now(
    run_tag: str,
    stamp: str,
    args,
    layers: List[int],
    heads: List[int],
    seg_list: List[int],
    input_dir: Path,
    pso_script: Path,
    encoding_json: Path,
    common_args: List[str],
    env_vars: Dict[str, str],
    per_run_root: Path,
    logs_root: Path,
    expected_keys: List[Tuple[int, int, int]],
    pending_jobs_count: int,
    done_valid_count: int,
    skip_count: int,
    rerun_invalid_count: int,
    invalid_existing_detected: List[Dict[str, Any]],
    missing_bins: List[str],
    results_map: Dict[Tuple[int, int, int, int], Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Convert map -> sorted flat list
    results_flat = list(results_map.values())
    results_flat.sort(key=lambda r: (
        int(r.get("segments", 0)),
        int(r.get("layer", 0)),
        int(r.get("head", 0)),
        int(r.get("rank", 0)),
    ))
    failures_sorted = sorted(failures, key=lambda f: (f.get("out_json", ""), f.get("error", "")))

    counts = {
        "expected_total_jobs": int(len(expected_keys)),
        "valid_existing_detected": int(done_valid_count),
        "invalid_existing_detected": int(len(invalid_existing_detected)),
        "skipped_because_valid_exists": int(skip_count),
        "rerun_because_invalid_exists": int(rerun_invalid_count),
        "pending_jobs_this_run": int(pending_jobs_count),
        "results_flat_entries": int(len(results_flat)),
        "failures": int(len(failures_sorted)),
        "missing_bins": int(len(missing_bins)),
    }

    return build_final_object(
        run_tag=run_tag,
        stamp=stamp,
        args=args,
        layers=layers,
        heads=heads,
        seg_list=seg_list,
        input_dir=input_dir,
        pso_script=pso_script,
        encoding_json=encoding_json,
        common_args=common_args,
        env_vars=env_vars,
        counts=counts,
        missing_bins=missing_bins,
        invalid_existing_detected=invalid_existing_detected,
        results_flat=results_flat,
        failures=failures_sorted,
        logs_root=logs_root,
        per_run_root=per_run_root,
    )


if __name__ == "__main__":
    main()
