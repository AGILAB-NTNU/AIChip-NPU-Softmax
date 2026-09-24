#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np

EPS = 1e-12
SOFTMAX_SCALE  = 1.0 / 65535.0
SOFTMAX_OFFSET = 0

ps = None
fe_mod = None
FitnessEvaluator = None


def _load_runtime_dependencies():
    global ps, fe_mod, FitnessEvaluator

    try:
        import pyswarms as ps_mod
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency 'pyswarms'. Install runtime dependencies with "
            "'python3 -m pip install -r code/requirements.txt'."
        ) from exc

    try:
        import pso_fitness_eval as fe_mod_local
        from pso_fitness_eval import FitnessEvaluator as fitness_evaluator_local
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing PSO runtime dependencies. Install runtime dependencies with "
            "'python3 -m pip install -r code/requirements.txt'."
        ) from exc

    ps = ps_mod
    fe_mod = fe_mod_local
    FitnessEvaluator = fitness_evaluator_local
    fe_mod.decode_particle = decode_particle_impl

# -------------------------
# Encodings helpers (same keys as your compare script)
# -------------------------
def _load_encodings(path: str) -> dict:
    with open(path, "r") as f:
        enc = json.load(f)
    return enc["activation_encodings"] if "activation_encodings" in enc else enc

def _get_enc(enc_act: dict, key: str) -> dict:
    if key not in enc_act:
        raise KeyError(f"Encoding key not found: {key}")
    e0 = enc_act[key][0]
    return {
        "scale": float(e0["scale"]),
        "offset": int(e0["offset"]),
        "min": float(e0.get("min", 0.0)),
        "max": float(e0.get("max", 0.0)),
        "bitwidth": int(e0.get("bitwidth", 16)),
        "dtype": e0.get("dtype", "int"),
    }

def get_layer_add2_encoding(enc: dict, layer_idx: int) -> dict:
    key = f"/model_layers_{layer_idx}_self_attn_Add_2/Add_output_0"
    return _get_enc(enc, key)

# -------------------------
# Load u16 Add2 tensor from .bin (Q,K)
# -------------------------
def load_add2_u16_bin(path: str, K: int, Q: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if Q is None:
        if raw.size % K != 0:
            raise ValueError(f"Cannot infer Q: file elems={raw.size} not divisible by K={K}")
        Q = raw.size // K
    expected = Q * K
    if raw.size != expected:
        raise ValueError(f"Shape mismatch: elems={raw.size}, expected={expected} for (Q,K)=({Q},{K})")
    return raw.reshape(Q, K)

def load_w_density_bin(path: str, K: int, Q: int) -> np.ndarray:
    """
    Optional weights; stored as float32 binary Q*K.
    Returns (Q,K) float64.
    """
    raw = np.fromfile(path, dtype=np.float32)
    expected = Q * K
    if raw.size != expected:
        raise ValueError(f"w_density mismatch: elems={raw.size}, expected={expected}")
    return raw.reshape(Q, K).astype(np.float64)

# -------------------------
# Required decode_particle() implementation
# Particle format:
#   particle[:S]   = w_logits (segment widths via softmax)
#   particle[S:2S] = deg_real (rounded/clipped to int [0..deg_max])
# -------------------------
def decode_particle_impl(particle: np.ndarray, clamp_min: float, clamp_max: float, segments: int, deg_max: int):
    p = np.asarray(particle, dtype=np.float64).reshape(-1)
    S = int(segments)
    if p.size != 2 * S:
        raise ValueError(f"particle must have length 2S={2*S}, got {p.size}")

    w_logits = p[:S]
    deg_real = p[S:]

    # widths from softmax(logits) * range
    rng = float(clamp_max - clamp_min)
    w = w_logits - np.max(w_logits)
    e = np.exp(w)
    frac = e / (np.sum(e) + EPS)
    widths = frac * rng

    # boundaries by cumulative sum
    b = np.empty((S + 1,), dtype=np.float64)
    b[0] = float(clamp_min)
    b[1:] = b[0] + np.cumsum(widths)
    b[-1] = float(clamp_max)

    # degrees
    deg = np.rint(deg_real).astype(np.int32)
    deg = np.clip(deg, 0, int(deg_max)).astype(np.int32)

    # Make sure boundaries strictly increasing (tiny nudges if numerical ties)
    for i in range(1, S + 1):
        if b[i] <= b[i - 1]:
            b[i] = b[i - 1] + 1e-9
    b[0] = float(clamp_min)
    b[-1] = float(clamp_max)

    widths_out = np.diff(b)
    return b, deg, widths_out

# -------------------------
# Debug helpers
# -------------------------
def stats_array(name, x):
    x = np.asarray(x)
    print(f"[{name}] shape={x.shape} dtype={x.dtype} "
          f"min={x.min():.6g} max={x.max():.6g} mean={x.mean():.6g} std={x.std():.6g}")

def clip_init_to_bounds(X, lb, ub, tag="init"):
    X = np.asarray(X, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64).reshape(1, -1)
    ub = np.asarray(ub, dtype=np.float64).reshape(1, -1)
    below = X < lb
    above = X > ub
    n_below = int(np.sum(below))
    n_above = int(np.sum(above))
    n_total = X.size
    if n_below or n_above:
        print(f"[{tag}] Out-of-bounds: below={n_below}/{n_total}, above={n_above}/{n_total}. Clipping.")
        X = np.maximum(X, lb)
        X = np.minimum(X, ub)
    else:
        print(f"[{tag}] All init positions within bounds ✅")
    return X

def print_decode_summary(fe, particle, title=""):
    f, det = fe(particle)
    print("\n---", title, "---")
    print("objective(f):", f)
    if "fitness" in det:
        print("det_fitness:", det["fitness"])

    # Your evaluator uses these keys
    for k in [
        "kl_mean_u16","kl_p95_u16","kl_max_u16",
        "expected_deg","dens_pen_raw","flip_rate",
        "pen_rob","pen_max","pen_dens","pen_cost","pen_flip","pen_tiny","pen_bad",
        "min_width","max_width","num_inv_half_zero"
    ]:
        if k in det:
            print(f"{k}:", det[k])

    if "deg" in det:
        print("deg:", det["deg"])
    if "boundaries" in det:
        b = det["boundaries"]
        print("boundaries preview:")
        for i in range(min(17, len(b))):
            print(f"  b[{i}]={b[i]:.6f}")
        if len(b) > 17:
            print("  ...")

# -------------------------
# Smart init (inverse-CDF from TRAIN z/w inside evaluator)
# -------------------------
def _boundaries_to_w_logits(b, clamp_min, clamp_max, eps=1e-12):
    b = np.asarray(b, dtype=np.float64)
    widths = np.diff(b)
    widths = np.clip(widths, eps, None)
    widths = widths / np.maximum(np.sum(widths), eps)
    logits = np.log(widths)
    logits -= np.mean(logits)
    return logits

def inverse_cdf_initialization_local(fe, jitter=0.02, seed=0, min_width=1e-6):
    rng = np.random.default_rng(seed)
    S = int(fe.S)

    # pull TRAIN z/w from evaluator
    z_tr = fe.z[fe.train_rows, :].reshape(-1).astype(np.float64)
    w_tr = fe.w_density[fe.train_rows, :].reshape(-1).astype(np.float64)

    # sort by z
    order = np.argsort(z_tr)
    zS = z_tr[order]
    wS = np.maximum(w_tr[order], 0.0)
    cum = np.concatenate(([0.0], np.cumsum(wS)))
    total = float(cum[-1])

    if total <= 0:
        b = np.linspace(fe.clamp_min, fe.clamp_max, S + 1)
        return _boundaries_to_w_logits(b, fe.clamp_min, fe.clamp_max) + rng.normal(0.0, jitter, size=S)

    targets = total * np.linspace(0.0, 1.0, S + 1)

    b = np.empty((S + 1,), dtype=np.float64)
    b[0] = fe.clamp_min
    b[-1] = fe.clamp_max

    for i in range(1, S):
        j = int(np.searchsorted(cum, targets[i], side="left"))
        j = np.clip(j, 1, len(cum) - 1)
        b[i] = float(zS[j - 1])

    b = np.clip(b, fe.clamp_min, fe.clamp_max)
    b[1:-1] = np.sort(b[1:-1])
    b[0] = fe.clamp_min
    b[-1] = fe.clamp_max

    # repair tiny widths
    for i in range(1, S + 1):
        if b[i] <= b[i - 1] + min_width:
            b[i] = b[i - 1] + min_width

    if b[-1] > fe.clamp_max:
        b[-1] = fe.clamp_max

    w_logits = _boundaries_to_w_logits(b, fe.clamp_min, fe.clamp_max)
    return w_logits + rng.normal(0.0, jitter, size=S)

def make_stage1_init(fe, n_particles, smart_particles=8, smart_jitter=0.02, seed=0):
    rng = np.random.default_rng(seed)
    S = int(fe.S)
    X = np.zeros((n_particles, S), dtype=np.float64)

    n_smart = min(smart_particles, n_particles)
    for i in range(n_smart):
        X[i] = inverse_cdf_initialization_local(fe, jitter=smart_jitter, seed=seed + 1000 + i)

    for i in range(n_smart, n_particles):
        X[i] = rng.normal(0.0, 1.0, size=S)

    return X

def make_stage2_init(fe, n_particles, best_w_logits, deg_init=2.0, w_jitter=0.05, deg_jitter=0.25, seed=0):
    rng = np.random.default_rng(seed)
    S = int(fe.S)
    X = np.zeros((n_particles, 2 * S), dtype=np.float64)

    X[0, :S] = best_w_logits
    X[0, S:] = deg_init

    for i in range(1, n_particles):
        X[i, :S] = best_w_logits + rng.normal(0.0, w_jitter, size=S)
        X[i, S:] = deg_init + rng.normal(0.0, deg_jitter, size=S)

    return X
def print_stage2_final_topk(fe, opt2, k=5):
    """
    Print the best-ever (PBEST) top-k particles across the whole Stage2 run.
    """
    # best positions each particle ever visited
    pos = np.asarray(opt2.swarm.pbest_pos, dtype=np.float64)

    # best costs each particle ever achieved
    costs = np.asarray(opt2.swarm.pbest_cost, dtype=np.float64).reshape(-1)

    if pos.shape[0] != costs.shape[0]:
        raise RuntimeError(f"pbest mismatch: pos {pos.shape} vs costs {costs.shape}")

    k = min(int(k), costs.size)
    idx = np.argsort(costs)[:k]

    print(f"\n[Stage2] PBEST TOP-{k} particles (best-ever across all iterations):")
    for rank, i in enumerate(idx, start=1):
        # costs[i] is the true pbest objective
        print_decode_summary(fe, pos[i], title=f"Stage2 PBEST TOP#{rank}  pbest_cost={costs[i]:.6g}")


def collect_stage2_topk(fe, opt2, k=5):
    """
    Collect top-k particles from PBEST (best-ever across the entire Stage2 run).
    Returns list of dicts with rank, objective_cost, and key metrics.
    """
    pos = np.asarray(opt2.swarm.pbest_pos, dtype=np.float64)
    costs = np.asarray(opt2.swarm.pbest_cost, dtype=np.float64).reshape(-1)

    if pos.shape[0] != costs.shape[0]:
        raise RuntimeError(f"pbest mismatch: pos {pos.shape} vs costs {costs.shape}")

    k = max(1, int(k))
    idx = np.argsort(costs)[:min(k, costs.size)]

    out = []
    for rank, i in enumerate(idx, start=1):
        f, det = fe(pos[i])  # recompute to get boundaries/deg/metrics
        out.append({
            "rank": rank,
            "pbest_cost": float(costs[i]),     # <-- IMPORTANT: the real best-ever PSO objective
            "objective_f": float(f),           # should be close to pbest_cost (unless your det["fitness"] confusion)
            "kl_mean_u16": float(det.get("kl_mean_u16", np.nan)),
            "kl_p95_u16": float(det.get("kl_p95_u16", np.nan)),
            "kl_max_u16": float(det.get("kl_max_u16", np.nan)),
            "expected_deg": float(det.get("expected_deg", np.nan)),
            "flip_rate": float(det.get("flip_rate", np.nan)),
            "dens_pen_raw": float(det.get("dens_pen_raw", np.nan)),
            "deg": det.get("deg", None),
            "boundaries": det.get("boundaries", None),
            "min_width": float(det.get("min_width", np.nan)),
            "max_width": float(det.get("max_width", np.nan)),
        })
    return out



def write_stage2_topk_json(path, meta: dict, topk_list: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"meta": meta, "topk": topk_list}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--add2-u16-bin", required=True)
    ap.add_argument("--encoding-json", required=True)
    ap.add_argument("--layer-idx", type=int, required=True)

    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--Q", type=int, default=None)

    ap.add_argument("--segments", type=int, default=16)
    ap.add_argument("--deg-max", type=int, default=5)
    ap.add_argument("--clamp-min", type=float, default=-20.0)
    ap.add_argument("--clamp-max", type=float, default=0.0)

    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--no-split", action="store_true",
                help="Disable train/val split: fit and evaluate on all rows")

    # evaluator knobs (MATCH pso_fitness_eval.py signature)
    ap.add_argument("--lam-rob", type=float, default=0.5)
    ap.add_argument("--rob-p", type=float, default=95.0)
    ap.add_argument("--lam-max", type=float, default=0.0)
    ap.add_argument("--lam-dens", type=float, default=0.1)
    ap.add_argument("--lam-cost", type=float, default=0.05)
    ap.add_argument("--cost-power", type=float, default=1.0)
    ap.add_argument("--lam-flip", type=float, default=1.0)
    ap.add_argument("--min-width", type=float, default=0.02)
    ap.add_argument("--tiny-width-penalty", type=float, default=5.0)

    ap.add_argument("--w-density-bin", type=str, default=None)

    # PSO controls\
    ap.add_argument("--no-smart-init", action="store_true",
                help="Disable inverse-CDF smart init; use random init only")
    ap.add_argument("--particles1", type=int, default=40)
    ap.add_argument("--iters1", type=int, default=200)
    ap.add_argument("--particles2", type=int, default=60)
    ap.add_argument("--iters2", type=int, default=300)

    ap.add_argument("--n-smart", type=int, default=10)
    ap.add_argument("--smart-jitter", type=float, default=0.02)
    ap.add_argument("--deg-fixed", type=float, default=2.0)

    ap.add_argument("--w-lo", type=float, default=-15.0)
    ap.add_argument("--w-hi", type=float, default=15.0)
    ap.add_argument("--w-jitter", type=float, default=0.05)
    ap.add_argument("--deg-jitter", type=float, default=0.35)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stage2-topk", type=int, default=5,
                help="How many best Stage2 particles to export")
    ap.add_argument("--stage2-json-out", type=str, default=None,
                help="If set, write Stage2 topK solutions to this JSON path")
    args = ap.parse_args()
    _load_runtime_dependencies()

    if not os.path.isfile(args.add2_u16_bin):
        raise FileNotFoundError(args.add2_u16_bin)

    enc = _load_encodings(args.encoding_json)
    e_add2 = get_layer_add2_encoding(enc, args.layer_idx)

    add2_u16 = load_add2_u16_bin(args.add2_u16_bin, K=args.K, Q=args.Q)  # (rows,K)
    rows, K = add2_u16.shape

    w_density = None
    if args.w_density_bin is not None:
        w_density = load_w_density_bin(args.w_density_bin, K=K, Q=rows)  # (rows,K)

    fe = FitnessEvaluator(
        add2_u16=add2_u16,
        add2_scale=float(e_add2["scale"]),
        add2_offset=int(e_add2["offset"]),
        softmax_scale=float(SOFTMAX_SCALE),
        softmax_offset=int(SOFTMAX_OFFSET),
        clamp_min=float(args.clamp_min),
        clamp_max=float(args.clamp_max),
        w_density=w_density,
        segments=int(args.segments),
        deg_max=int(args.deg_max),
        train_frac=float(args.train_frac),
        split_seed=int(args.split_seed),
        use_split=(not args.no_split),
        lam_rob=float(args.lam_rob),
        rob_p=float(args.rob_p),
        lam_max=float(args.lam_max),
        lam_dens=float(args.lam_dens),
        lam_cost=float(args.lam_cost),
        cost_power=float(args.cost_power),
        lam_flip=float(args.lam_flip),
        min_width=float(args.min_width),
        tiny_width_penalty=float(args.tiny_width_penalty),
        seed=int(args.seed),
    )

    S = int(fe.S)

    print("\n================= SETUP =================")
    print("BIN:", args.add2_u16_bin)
    print("enc:", args.encoding_json, "layer:", args.layer_idx)
    print(f"add2_u16 shape={add2_u16.shape} dtype={add2_u16.dtype}")
    print(f"Add2 scale={e_add2['scale']} offset={e_add2['offset']}")
    print(f"Softmax scale={SOFTMAX_SCALE} offset={SOFTMAX_OFFSET}")
    print(f"Clamp=[{fe.clamp_min},{fe.clamp_max}] segments={S} deg_max={fe.deg_max}")
    print(f"Split: train_rows={fe.train_rows.size} val_rows={fe.val_rows.size}")
    print("Stage1 dim =", S, "Stage2 dim =", fe.dim)
    print("=========================================\n")

    # -------------------------
    # STAGE 1: boundaries only
    # -------------------------
    fixed_deg = float(args.deg_fixed)
    deg_fixed_vec = np.full((S,), fixed_deg, dtype=np.float64)

    def fitness_stage1(X):
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        degs = np.tile(deg_fixed_vec[None, :], (N, 1))
        full = np.concatenate([X, degs], axis=1)  # (N,2S)
        return fe.batch(full)

    lb1 = np.full((S,), args.w_lo, dtype=np.float64)
    ub1 = np.full((S,), args.w_hi, dtype=np.float64)
    bounds1 = (lb1, ub1)
    options1 = dict(c1=1.4, c2=1.4, w=0.7)

    smart_n = 0 if args.no_smart_init else int(args.n_smart)

    init1 = make_stage1_init(
        fe,
        n_particles=int(args.particles1),
        smart_particles=smart_n,
        smart_jitter=float(args.smart_jitter),
        seed=args.seed + 10,
    )
    init1 = clip_init_to_bounds(init1, lb1, ub1, tag="stage1.init1")

    # sanity decode
    p0_full = np.concatenate([init1[0], deg_fixed_vec])
    print_decode_summary(fe, p0_full, title="Stage1: first init particle decoded")

    opt1 = ps.single.GlobalBestPSO(
        n_particles=int(args.particles1),
        dimensions=S,
        options=options1,
        bounds=bounds1,
        init_pos=init1,
    )

    print("\n================= STAGE 1 (boundaries only) =================")
    best_cost1, best_pos1 = opt1.optimize(fitness_stage1, iters=int(args.iters1), verbose=True)
    print("[Stage1] best_cost =", best_cost1)
    best1_full = np.concatenate([best_pos1, deg_fixed_vec])
    print_decode_summary(fe, best1_full, title="Stage1 BEST decoded")

    # -------------------------
    # STAGE 2: boundaries + degrees
    # -------------------------
    def fitness_stage2(X):
        return fe.batch(X)

    lb2 = np.concatenate([np.full((S,), args.w_lo), np.full((S,), 0.0)]).astype(np.float64)
    ub2 = np.concatenate([np.full((S,), args.w_hi), np.full((S,), float(args.deg_max))]).astype(np.float64)
    bounds2 = (lb2, ub2)
    options2 = dict(c1=1.2, c2=1.6, w=0.7)

    init2 = make_stage2_init(
        fe,
        n_particles=int(args.particles2),
        best_w_logits=best_pos1,
        deg_init=fixed_deg,
        w_jitter=float(args.w_jitter),
        deg_jitter=float(args.deg_jitter),
        seed=args.seed + 20,
    )
    init2 = clip_init_to_bounds(init2, lb2, ub2, tag="stage2.init2")

    print_decode_summary(fe, init2[0], title="Stage2: init2[0] decoded")

    opt2 = ps.single.GlobalBestPSO(
        n_particles=int(args.particles2),
        dimensions=fe.dim,
        options=options2,
        bounds=bounds2,
        init_pos=init2,
    )


    print("\n================= STAGE 2 (boundaries + degrees) =================")
    best_cost2, best_pos2 = opt2.optimize(fitness_stage2, iters=int(args.iters2), verbose=True)
    print("[Stage2] best_cost =", best_cost2)
    print_stage2_final_topk(fe, opt2, k=5)
    #print_decode_summary(fe, best_pos2, title="FINAL BEST (Stage2) decoded")

    # Export topK to json (for pso_batch_runner.py)
    if args.stage2_json_out:
        meta = {
            "layer_idx": int(args.layer_idx),
            "segments": int(args.segments),
            "deg_max": int(args.deg_max),
            "K": int(args.K),
            "Q": int(args.Q) if args.Q is not None else None,
            "add2_u16_bin": args.add2_u16_bin,
            "encoding_json": args.encoding_json,
            "lam_rob": float(args.lam_rob),
            "rob_p": float(args.rob_p),
            "lam_max": float(args.lam_max),
            "lam_dens": float(args.lam_dens),
            "lam_cost": float(args.lam_cost),
            "cost_power": float(args.cost_power),
            "lam_flip": float(args.lam_flip),
            "min_width": float(args.min_width),
            "tiny_width_penalty": float(args.tiny_width_penalty),
            "no_smart_init": bool(args.no_smart_init),
            "no_split": bool(args.no_split),
            "seed": int(args.seed),
        }
        topk = collect_stage2_topk(fe, opt2, k=int(args.stage2_topk))
        write_stage2_topk_json(args.stage2_json_out, meta=meta, topk_list=topk)
        print(f"[Stage2] wrote top{args.stage2_topk} -> {args.stage2_json_out}")


if __name__ == "__main__":
    main()
