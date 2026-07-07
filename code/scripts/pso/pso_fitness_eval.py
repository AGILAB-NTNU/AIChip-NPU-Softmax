#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fitness_pso_eval.py  (u16-EXACT evaluation)

This evaluator matches your comparison script *exactly* for accuracy metrics:

- GOLD:
    gold_u16 = softmax_gold_u16_from_add2_u16(add2_u16, add2_scale, add2_offset, softmax_scale, softmax_offset)

- MY APPROX:
    approx_u16 = softmax_my_u16_from_add2_u16(add2_u16, P, add2_scale, softmax_scale, softmax_offset)
    where:
      * boundaries are quantized to Q30
      * coeffs are quantized to Q31
      * exp evaluated with integer-like Horner (Q31*Q15->Q31) using u computed from Q30 mid/half
      * softmax is computed in u16 counts (same integer div with rounding)

- KL METRIC:
    dequant u16 -> renorm rows -> KL(p||q) rowwise -> mean/p95/max

Then FITNESS adds regularizers aligned to your goal:
- expected_compute = sum_s mass_s * deg_s   (mass from TRAIN usage distribution)
- density_split penalty = KL(uniform || masses)  (pushes boundaries toward equal-mass / density-based split)
- flip penalty = fraction(gold_u16==0 & approx_u16>0) (a common cause of huge KL)
- tiny-width penalty (prevents boundary collapse)
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np

# -------------------------- constants --------------------------
EPS = 1e-12

Q30 = 30
Q15 = 15

CLAMP_MIN_DEFAULT = -20.0

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

# -------------------------- REQUIRED external decode_particle --------------------------
# You MUST provide your project’s decode_particle().
# It should return:
#   b_float: (S+1,) float boundaries in [clamp_min, clamp_max], strictly inc
#   deg:     (S,)   int degrees in [0..deg_max]
#   widths:  (S,)   float widths (b[i+1]-b[i])
def decode_particle(particle: np.ndarray, clamp_min: float, clamp_max: float, segments: int, deg_max: int):
    raise RuntimeError("decode_particle() must be provided by your codebase.")


# -------------------------- fixed-point helpers --------------------------
def _sat_int32(x: np.ndarray) -> np.ndarray:
    return np.clip(x, INT32_MIN, INT32_MAX).astype(np.int64)

def _mul_q31_q15_rnd_sat32(y_q31: np.ndarray, u_q15: np.ndarray) -> np.ndarray:
    """
    Multiply Q31 * Q15 -> Q31 with rounding and int32 saturation.
    prod is Q46. Shift right by 15 to get Q31.
    Symmetric rounding: +2^14 for >=0, -2^14 for <0.
    """
    prod = y_q31.astype(np.int64) * u_q15.astype(np.int64)  # Q46
    rnd = (1 << 14)
    prod = prod + np.where(prod >= 0, rnd, -rnd)
    out = (prod >> 15)
    return _sat_int32(out)


# -------------------------- u16 quant/dequant (same as compare script) --------------------------
def dequant_u16_np(q_u16: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q_u16.astype(np.int32) + int(offset)) * float(scale)


# -------------------------- GOLD softmax (same as compare script) --------------------------
def softmax_gold_u16_from_add2_u16(
    add2_u16: np.ndarray,
    axis: int = -1,
    add2_scale: float = None,
    add2_offset: int = None,
    softmax_scale: float = None,
    softmax_offset: int = 0,
) -> np.ndarray:
    assert add2_scale is not None and softmax_scale is not None
    x = (add2_u16.astype(np.int32) + int(add2_offset)) * float(add2_scale)
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    p = e / np.sum(e, axis=axis, keepdims=True)
    q = np.rint(p / float(softmax_scale) - float(softmax_offset)).astype(np.int64)
    return np.clip(q, 0, 65535).astype(np.uint16)


# -------------------------- MY APPROX params + exp (same math as compare script) --------------------------
@dataclass
class MyApproxParams:
    S: int
    b_q30: np.ndarray        # (S+1,)
    mid_q30: np.ndarray      # (S,)
    half_q30: np.ndarray     # (S,)
    inv_half_q48: np.ndarray # (S,)
    c0: np.ndarray; c1: np.ndarray; c2: np.ndarray; c3: np.ndarray; c4: np.ndarray; c5: np.ndarray

def _build_my_params_from_q30_q31(b_q30: np.ndarray, coeffs_q31_list: list[list[int]]) -> MyApproxParams:
    b_q30 = np.asarray(b_q30, dtype=np.int64).reshape(-1)
    S = b_q30.size - 1

    lo = b_q30[:-1]
    hi = b_q30[1:]
    mid_q30  = (lo + hi) // 2
    half_q30 = (hi - lo) // 2

    inv_half_q48 = np.zeros((S,), dtype=np.int64)
    for s in range(S):
        h = int(half_q30[s])
        inv_half_q48[s] = ((1 << 48) // h) if h > 0 else 0

    # expand each segment coeff list (c0..cd) into fixed c0..c5 arrays
    c0 = np.zeros((S,), dtype=np.int64)
    c1 = np.zeros((S,), dtype=np.int64)
    c2 = np.zeros((S,), dtype=np.int64)
    c3 = np.zeros((S,), dtype=np.int64)
    c4 = np.zeros((S,), dtype=np.int64)
    c5 = np.zeros((S,), dtype=np.int64)

    if len(coeffs_q31_list) != S:
        raise ValueError(f"coeffs list has {len(coeffs_q31_list)} segments, but S={S}")

    for s in range(S):
        cs = list(coeffs_q31_list[s])
        if len(cs) < 1 or len(cs) > 6:
            raise ValueError(f"segment {s}: coeffs length must be 1..6 (deg 0..5)")
        c0[s] = int(cs[0])
        if len(cs) > 1: c1[s] = int(cs[1])
        if len(cs) > 2: c2[s] = int(cs[2])
        if len(cs) > 3: c3[s] = int(cs[3])
        if len(cs) > 4: c4[s] = int(cs[4])
        if len(cs) > 5: c5[s] = int(cs[5])

    return MyApproxParams(S=S, b_q30=b_q30, mid_q30=mid_q30, half_q30=half_q30,
                          inv_half_q48=inv_half_q48, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5)

def _exp_my_poly_q31_from_dq(dq: np.ndarray, scale_q30: int, P: MyApproxParams) -> np.ndarray:
    """
    EXACT same flow as your compare script.
    dq: int32 <= 0
    """
    x_q30 = dq.astype(np.int64) * np.int64(scale_q30)
    x_q30 = np.clip(x_q30, P.b_q30[0], P.b_q30[-1]).astype(np.int64)

    seg = np.searchsorted(P.b_q30, x_q30, side="right") - 1
    seg = np.clip(seg, 0, P.S - 1).astype(np.int32)

    mid = P.mid_q30[seg]
    num = ((x_q30 - mid) << Q15).astype(np.int64)

    abs_num = np.abs(num)
    invh = P.inv_half_q48[seg].astype(np.int64)
    u_mag = (abs_num * invh) >> 48
    u_q15 = np.where(num < 0, -u_mag, u_mag).astype(np.int64)
    u_q15 = np.clip(u_q15, -32768, 32767).astype(np.int64)

    c0 = P.c0[seg]; c1 = P.c1[seg]; c2 = P.c2[seg]; c3 = P.c3[seg]; c4 = P.c4[seg]; c5 = P.c5[seg]

    y = c5
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c4)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c3)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c2)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c1)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c0)

    y = np.clip(y, 0, INT32_MAX).astype(np.uint32)
    return y

def _softmax_from_expq31(exp_q31: np.ndarray, axis: int, softmax_scale: float, softmax_offset: int = 0) -> np.ndarray:
    sum_q31 = np.sum(exp_q31.astype(np.uint64), axis=axis, keepdims=True)
    sum_q31 = np.maximum(sum_q31, 1)

    scale_inv = int(round(1.0 / float(softmax_scale)))  # usually 65535
    num = exp_q31.astype(np.uint64) * np.uint64(scale_inv)
    prob_u16 = (num + (sum_q31 // 2)) // sum_q31

    if softmax_offset != 0:
        prob_u16 = prob_u16 - np.uint64(softmax_offset)

    return np.clip(prob_u16, 0, 65535).astype(np.uint16)

def softmax_my_u16_from_add2_u16(
    add2_u16: np.ndarray,
    P: MyApproxParams,
    axis: int = -1,
    clamp_min: float = CLAMP_MIN_DEFAULT,
    add2_scale: float = None,
    softmax_scale: float = None,
    softmax_offset: int = 0,
) -> np.ndarray:
    assert add2_u16.dtype == np.uint16
    assert add2_scale is not None and softmax_scale is not None

    q = add2_u16.astype(np.int32)
    qmax = np.max(q, axis=axis, keepdims=True)
    dq = q - qmax

    dq_min = int(np.ceil(float(clamp_min) / float(add2_scale)))
    dq = np.maximum(dq, dq_min)

    scale_q30 = int(round(float(add2_scale) * (1 << Q30)))
    exp_q31 = _exp_my_poly_q31_from_dq(dq, scale_q30=scale_q30, P=P)

    return _softmax_from_expq31(exp_q31, axis=axis, softmax_scale=softmax_scale, softmax_offset=softmax_offset)


# -------------------------- fitting coeffs consistent with runtime u mapping --------------------------
def _fit_coeffs_train_q31(
    z_train: np.ndarray,         # (n_train, K) float, already clamped to [clamp_min, 0]
    w_train: np.ndarray,         # (n_train, K) float weights (>=0)
    b_q30: np.ndarray,           # (S+1,) int64 boundaries in Q30
    deg: np.ndarray,             # (S,) int
) -> list[list[int]]:
    """
    Fit y=exp(z) in each segment using u computed from Q30 mid/half (so it matches runtime mapping),
    then quantize coefficients to Q31 (int32-saturated), returning per-segment lists [c0..cd].
    """
    S = b_q30.size - 1

    lo_q30 = b_q30[:-1]
    hi_q30 = b_q30[1:]
    mid_q30 = (lo_q30 + hi_q30) // 2
    half_q30 = (hi_q30 - lo_q30) // 2

    coeffs_q31_list: list[list[int]] = []

    zt = np.asarray(z_train, dtype=np.float64)
    wt = np.asarray(w_train, dtype=np.float64)
    yt = np.exp(zt)

    for s in range(S):
        d = int(deg[s])

        mid_f = float(mid_q30[s]) / float(1 << Q30)
        half_f = float(half_q30[s]) / float(1 << Q30)

        # degenerate segment
        if d <= 0 or half_f <= 0.0:
            c0 = int(np.clip(np.rint(np.exp(mid_f) * (1 << 31)), INT32_MIN, INT32_MAX))
            coeffs_q31_list.append([c0])
            continue

        lo_f = float(lo_q30[s]) / float(1 << Q30)
        hi_f = float(hi_q30[s]) / float(1 << Q30)

        if s == S - 1:
            m = (zt >= lo_f) & (zt <= hi_f)
        else:
            m = (zt >= lo_f) & (zt < hi_f)

        if not np.any(m):
            c0 = int(np.clip(np.rint(np.exp(mid_f) * (1 << 31)), INT32_MIN, INT32_MAX))
            coeffs_q31_list.append([c0])
            continue

        zz = zt[m]
        yy = yt[m]
        ww = np.maximum(wt[m], 0.0)

        # u from Q30 mid/half
        u = (zz - mid_f) / (half_f + EPS)
        u = np.clip(u, -1.0, 1.0)

        # weighted LS in power basis, ascending coeffs c0..cd
        d_eff = min(d, max(0, u.size - 1))
        # build Vandermonde
        V = np.vstack([u**k for k in range(d_eff + 1)]).T
        sw = np.sqrt(ww + EPS)
        Vw = V * sw[:, None]
        yw = yy * sw
        c_float, *_ = np.linalg.lstsq(Vw, yw, rcond=None)

        cq = np.rint(c_float * (1 << 31)).astype(np.int64)
        cq = np.clip(cq, INT32_MIN, INT32_MAX).astype(np.int64)
        coeffs_q31_list.append([int(v) for v in cq.tolist()])

    return coeffs_q31_list


# -------------------------- density masses (TRAIN) --------------------------
def _segment_masses_train_from_z(z_train_flat: np.ndarray, w_train_flat: np.ndarray, b_float: np.ndarray) -> np.ndarray:
    """
    Compute normalized masses per segment using TRAIN z samples (float) + weights,
    with the SAME boundary inclusion convention used in your fitter:
      seg s: [b[s], b[s+1]) except last: [b[S-1], b[S]]
    """
    zS = np.sort(z_train_flat)
    # But we need weight-aligned sorting for fast prefix sums.
    order = np.argsort(z_train_flat)
    zS = z_train_flat[order].astype(np.float64)
    wS = w_train_flat[order].astype(np.float64)
    cum = np.concatenate(([0.0], np.cumsum(wS)))

    S = b_float.size - 1
    masses = np.zeros((S,), dtype=np.float64)
    for s in range(S):
        lo = float(b_float[s])
        hi = float(b_float[s + 1])
        i0 = np.searchsorted(zS, lo, side="left")
        if s == S - 1:
            i1 = np.searchsorted(zS, hi, side="right")
        else:
            i1 = np.searchsorted(zS, hi, side="left")
        masses[s] = cum[i1] - cum[i0]

    masses = masses / (np.sum(masses) + EPS)
    return masses


# -------------------------- EXACT compare-KL (same as your compare()) --------------------------
def _kl_rows_from_u16(
    gold_u16: np.ndarray,
    approx_u16: np.ndarray,
    softmax_scale: float,
    softmax_offset: int,
    eps: float = 1e-12,
) -> np.ndarray:
    gold = dequant_u16_np(gold_u16, softmax_scale, softmax_offset).astype(np.float64)
    appr = dequant_u16_np(approx_u16, softmax_scale, softmax_offset).astype(np.float64)

    K = gold.shape[-1]
    gold = gold.reshape(-1, K)
    appr = appr.reshape(-1, K)

    gold = gold / np.maximum(gold.sum(axis=1, keepdims=True), eps)
    appr = appr / np.maximum(appr.sum(axis=1, keepdims=True), eps)

    p = np.clip(gold, eps, 1.0)
    q = np.clip(appr, eps, 1.0)
    kl_pq = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    return kl_pq.astype(np.float64)


# -------------------------- FITNESS EVALUATOR (exact pipeline) --------------------------
class FitnessEvaluator:
    """
    Evaluate PSO particle using EXACT same pipeline as compare script.

    Inputs are raw captured Add_2 quantized tensors:
      add2_u16: shape (Q, K) or (N_rows, K), dtype uint16

    Encodings must be the same used in comparison:
      add2_scale, add2_offset, softmax_scale, softmax_offset

    Fitness = KL_mean + lam_rob*KL_p95 + lam_max*KL_max
              + lam_cost * E[deg]  (usage-weighted)
              + lam_dens * KL(uniform || masses)
              + lam_flip * flip_rate
              + tiny_width_penalty + 100*bad_frac
    """

    def __init__(
        self,
        add2_u16: np.ndarray,          # (Q,K) uint16 (or rows,K)
        add2_scale: float,
        add2_offset: int,
        softmax_scale: float,
        softmax_offset: int = 0,
        clamp_min: float = CLAMP_MIN_DEFAULT,
        clamp_max: float = 0.0,
        w_density: np.ndarray | None = None,  # optional weights same shape as add2_u16

        segments: int = 16,
        deg_max: int = 5,
        train_frac: float = 0.8,
        split_seed: int = 123,
        use_split: bool = True,

        # robustness on KL (still computed EXACT like compare)
        lam_rob: float = 0.5,
        rob_p: float = 95.0,
        lam_max: float = 0.0,

        # density-based split penalty (push masses toward uniform)
        lam_dens: float = 0.1,

        # expected compute penalty
        lam_cost: float = 0.05,
        cost_power: float = 1.0,   # 1 => deg, 2 => deg^2

        # flip penalty (gold==0 & approx>0)
        lam_flip: float = 1.0,

        # tiny width penalty
        min_width: float = 0.02,
        tiny_width_penalty: float = 5.0,

        seed: int = 0,
    ):
        self.S = int(segments)
        self.deg_max = int(deg_max)
        self.train_frac = float(train_frac)
        self.split_seed = int(split_seed)
        self.use_split = bool(use_split)

        self.add2_scale = float(add2_scale)
        self.add2_offset = int(add2_offset)
        self.softmax_scale = float(softmax_scale)
        self.softmax_offset = int(softmax_offset)

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        self.lam_rob = float(lam_rob)
        self.rob_p = float(rob_p)
        self.lam_max = float(lam_max)

        self.lam_dens = float(lam_dens)
        self.lam_cost = float(lam_cost)
        self.cost_power = float(cost_power)
        self.lam_flip = float(lam_flip)

        self.min_width = float(min_width)
        self.tiny_width_penalty = float(tiny_width_penalty)

        self.rng = np.random.default_rng(int(seed))
        

        add2_u16 = np.asarray(add2_u16)
        if add2_u16.dtype != np.uint16:
            raise ValueError("add2_u16 must be uint16")
        if add2_u16.ndim != 2:
            raise ValueError("add2_u16 must have shape (rows, K)")
        self.rows, self.K = add2_u16.shape
        self.add2_u16 = add2_u16

        # weights
        if w_density is None:
            w_density = np.ones_like(add2_u16, dtype=np.float64)
        else:
            w_density = np.asarray(w_density, dtype=np.float64)
            if w_density.shape != add2_u16.shape:
                raise ValueError("w_density must match add2_u16 shape")
            w_density = np.maximum(w_density, 0.0)

        # normalize weights mean=1 (stability)
        w_density = w_density / (np.mean(w_density) + EPS)
        self.w_density = w_density

        # ---- Precompute dq + z (runtime-relevant), for fitting + masses ----
        q = add2_u16.astype(np.int32)
        qmax = np.max(q, axis=1, keepdims=True)
        dq = q - qmax  # <=0

        dq_min = int(np.ceil(self.clamp_min / self.add2_scale))
        dq_clamped = np.maximum(dq, dq_min)

        # z = dq*scale (this equals (x-xmax) and matches runtime input domain)
        z = dq_clamped.astype(np.float64) * self.add2_scale
        z = np.clip(z, self.clamp_min, self.clamp_max)
        self.z = z  # (rows,K)

        # ---- Train/val split on ROWS (exactly like you did) ----
        if not self.use_split:
            # no split: train=all, val=all (so fitter + eval use same rows)
            self.train_rows = np.arange(self.rows, dtype=np.int64)
            self.val_rows   = np.arange(self.rows, dtype=np.int64)
        else:
            rng_split = np.random.default_rng(self.split_seed)
            perm = rng_split.permutation(self.rows)
            n_train = int(np.floor(self.train_frac * self.rows))
            n_train = max(1, min(self.rows - 1, n_train))
            self.train_rows = perm[:n_train]
            self.val_rows = perm[n_train:]

        # uniform mass target for density-based split
        self.mass_target = np.full((self.S,), 1.0 / float(self.S), dtype=np.float64)

    @property
    def dim(self) -> int:
        return 2 * self.S

    def batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"batch expects shape (N,{self.dim}), got {X.shape}")
        out = np.empty((X.shape[0],), dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self(X[i])[0]
        return out

    def _quantize_boundaries_q30(self, b_float: np.ndarray) -> np.ndarray:
        b_q30 = np.rint(np.asarray(b_float, dtype=np.float64) * (1 << Q30)).astype(np.int64)
        # force endpoints
        b_q30[0]  = int(np.rint(self.clamp_min * (1 << Q30)))
        b_q30[-1] = int(np.rint(self.clamp_max * (1 << Q30)))
        # enforce strictly increasing in Q30 integers
        for i in range(1, b_q30.size):
            if b_q30[i] <= b_q30[i - 1]:
                b_q30[i] = b_q30[i - 1] + 1
        return b_q30

    def __call__(self, particle: np.ndarray):
        # (1) decode particle -> boundaries/deg
        b_float, deg, widths = decode_particle(
            particle,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            segments=self.S,
            deg_max=self.deg_max,
        )
        deg = np.asarray(deg, dtype=np.int32)

        # (2) TRAIN masses for density + expected compute
        z_tr = self.z[self.train_rows, :]
        w_tr = self.w_density[self.train_rows, :]
        masses = _segment_masses_train_from_z(
            z_tr.reshape(-1),
            w_tr.reshape(-1),
            np.asarray(b_float, dtype=np.float64),
        )

        # density penalty = KL(uniform || masses)
        m_clip = np.maximum(masses, 1e-300)
        u = self.mass_target
        dens_raw = float(np.sum(u * (np.log(u) - np.log(m_clip))))
        pen_dens = float(self.lam_dens * dens_raw)

        # expected compute = sum(m * deg^p)
        d = deg.astype(np.float64)
        if self.cost_power == 2.0:
            d = d * d
        expected_deg = float(np.sum(masses * d))
        pen_cost = float(self.lam_cost * expected_deg)

        # (3) quantize boundaries to Q30
        b_q30 = self._quantize_boundaries_q30(b_float)

        # (4) fit coeffs on TRAIN -> Q31 ints (lists)
        coeffs_q31_list = _fit_coeffs_train_q31(z_tr, w_tr, b_q30=b_q30, deg=deg)

        # (5) build runtime params + evaluate approx on VAL (EXACT compare pipeline)
        P = _build_my_params_from_q30_q31(b_q30=b_q30, coeffs_q31_list=coeffs_q31_list)

        add2_va = self.add2_u16[self.val_rows, :]  # (val_rows,K) uint16
        add2_va_3d = add2_va[None, :, :]           # [1,Q,K] style? -> we keep axis=-1, so 2D ok too.
        # but our functions expect np arrays; they work for 2D as well. Keep 2D.

        gold_u16 = softmax_gold_u16_from_add2_u16(
            add2_va,
            axis=-1,
            add2_scale=self.add2_scale,
            add2_offset=self.add2_offset,
            softmax_scale=self.softmax_scale,
            softmax_offset=self.softmax_offset,
        )

        approx_u16 = softmax_my_u16_from_add2_u16(
            add2_va,
            P=P,
            axis=-1,
            clamp_min=self.clamp_min,
            add2_scale=self.add2_scale,
            softmax_scale=self.softmax_scale,
            softmax_offset=self.softmax_offset,
        )

        # (6) KL EXACT like compare()
        kl_rows = _kl_rows_from_u16(
            gold_u16, approx_u16,
            softmax_scale=self.softmax_scale,
            softmax_offset=self.softmax_offset,
            eps=1e-12,
        )
        kl_mean = float(np.mean(kl_rows))
        kl_p95  = float(np.percentile(kl_rows, self.rob_p))
        kl_max  = float(np.max(kl_rows))

        pen_rob = float(self.lam_rob * kl_p95)
        pen_max = float(self.lam_max * kl_max)

        # (7) flip penalty (gold==0 & approx>0) on VAL
        flip_rate = float(np.mean((gold_u16 == 0) & (approx_u16 > 0)))
        pen_flip = float(self.lam_flip * flip_rate)

        # (8) tiny width penalty
        widths = np.asarray(widths, dtype=np.float64)
        short = np.maximum(0.0, (self.min_width - widths) / (self.min_width + EPS))
        pen_tiny = float(self.tiny_width_penalty * np.sum(short * short))

        # (9) bad_frac (degenerate inv_half==0 segments) – count as penalty
        # If any half_q30==0 => inv_half==0 => u becomes 0-ish; treat as "bad".
        bad_frac = float(np.mean(P.inv_half_q48 == 0))
        pen_bad = float(100.0 * bad_frac)

        fitness = float(
            kl_mean + pen_rob + pen_max +
            pen_dens + pen_cost +
            pen_flip +
            pen_tiny + pen_bad
        )

        details = {
            "fitness": fitness,

            # EXACT accuracy stats (compare-metric)
            "kl_mean_u16": kl_mean,
            "kl_p95_u16": kl_p95,
            "kl_max_u16": kl_max,

            # objective pieces
            "expected_deg": expected_deg,
            "dens_pen_raw": dens_raw,
            "flip_rate": flip_rate,

            "pen_rob": pen_rob,
            "pen_max": pen_max,
            "pen_dens": pen_dens,
            "pen_cost": pen_cost,
            "pen_flip": pen_flip,
            "pen_tiny": pen_tiny,
            "pen_bad": pen_bad,

            # decoded
            "deg": deg.tolist(),
            "boundaries": [float(x) for x in b_float],
            "min_width": float(np.min(widths)),
            "max_width": float(np.max(widths)),
            "segment_masses_train": masses.tolist(),

            # sanity / debug
            "num_inv_half_zero": int(np.sum(P.inv_half_q48 == 0)),
        }

        return fitness, details


if __name__ == "__main__":
    print("This module defines FitnessEvaluator(add2_u16, encodings, ...).")
    print("You must provide decode_particle() from your codebase.")
