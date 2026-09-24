#!/usr/bin/env python3
from __future__ import annotations

"""
Capture pre-softmax tensors for ALL layers (call order) and ALL heads, then compute
gold_u16 vs approx_u16 metrics per (layer, head). Writes compare_metrics.csv.

Methods:
  - qcom  : Qualcomm-style baseline, 16 equal segments, deg-4, fixed tables (existing)
  - qcom3 : Qualcomm-style baseline, 16 equal segments, deg-3, fixed tables (NEW)
  - my    : Variable segments (1..16), variable degree (0..5), BST lookup via searchsorted

Run:
  python3 scripts/emulation/emulation_softmax_metrics.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --context-len 2048 --chunk-len 128 --device cpu \
    --outdir ./workdir/emulation_metrics --max-layers 32 --topk 5 \
    --encoding-json configs/encodings/tinyllama.encodings \
    --methods qcom,qcom3,my \
    --my-approx-json workdir/coeff_banks/coeffs_q31_all_LH_lut128.json
"""
import os, json, argparse, csv
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

torch = None
F = None
AutoTokenizer = None
AutoModelForCausalLM = None


def _load_runtime_dependencies():
    global torch, F, AutoTokenizer, AutoModelForCausalLM

    try:
        import torch as torch_mod
        import torch.nn.functional as F_mod
        from transformers import AutoModelForCausalLM as auto_model_for_causal_lm_mod
        from transformers import AutoTokenizer as auto_tokenizer_mod
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing emulation dependencies. Install runtime dependencies with "
            "'python3 -m pip install -r code/requirements.txt'."
        ) from exc

    torch = torch_mod
    F = F_mod
    AutoTokenizer = auto_tokenizer_mod
    AutoModelForCausalLM = auto_model_for_causal_lm_mod

# NOTE: these are default softmax enc constants; we will use per-layer encodings in compare().
softmax_scale = 1.0 / 65535.0
softmax_offset = 0

EPS = 1e-12

# ---------------- Encodings ----------------
def _load_encodings(path: str) -> dict:
    with open(path, "r") as f:
        enc = json.load(f)
    if "activation_encodings" in enc:
        return enc["activation_encodings"]
    return enc

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

def get_layer_softmax_encoding(enc: dict, layer_idx: int) -> dict:
    key = f"/model_layers_{layer_idx}_self_attn_Softmax/Softmax_output_0"
    return _get_enc(enc, key)

# ---------------- Fixed-point constants ----------------
Q30 = 30
Q15 = 15

CLAMP_MIN = -20.0
CLAMP_MAX = 0.0

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

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

# ---------------- Qualcomm baseline tables (16 equal seg, deg4) ----------------
# Q30 clamp range [-20,0]
X_MIN_Q30 = -21474836480        # -20 * 2^30
X_RNG_Q30 =  21474836480        #  20 * 2^30
HALF_W_Q30 = 671088640          # 0.625 * 2^30
INV_XRNG_Q48 = 13107            # floor(2^48 / X_RNG_Q30)
INV_HALF_Q48 = 419430           # floor(2^48 / HALF_W_Q30)

X_MID_Q30 = np.array([
    -20803747840, -19461570560, -18119393280, -16777216000,
    -15435038720, -14092861440, -12750684160, -11408506880,
    -10066329600,  -8724152320,  -7381975040,  -6039797760,
     -4697620480,  -3355443200,  -2013265920,   -671088640
], dtype=np.int64)

# deg4 tables
C0 = np.array([8, 29, 101, 352, 1227, 4284, 14951, 52186, 182146, 635752, 2218994, 7745050, 27032879, 94354019, 329327887, 1149467273], dtype=np.int64)
C1 = np.array([5, 18, 63, 220, 767, 2676, 9342, 32606, 113806, 397222, 1386440, 4839153, 16890302, 58952946, 205766001, 718193911], dtype=np.int64)
C2 = np.array([2, 6, 20, 69, 240, 836, 2920, 10191, 35568, 124146, 433311, 1512406, 5278814, 18424871, 64309120, 224460885], dtype=np.int64)
C3 = np.array([0, 1, 4, 15, 51, 178, 622, 2170, 7574, 26437, 92276, 322074, 1124147, 3923659, 13694914, 47799948], dtype=np.int64)
C4 = np.array([0, 0, 1, 2, 8, 28, 97, 338, 1179, 4115, 14361, 50126, 174957, 610659, 2131410, 7439353], dtype=np.int64)

# NEW: deg3 tables (true fitted tables you provided)
C0_D3 = np.array([8, 29, 101, 351, 1226, 4280, 14943, 52147, 182007, 635274, 2217261, 7738907, 27011122, 94277759, 329061395, 1148536803], dtype=np.int64)
C1_D3 = np.array([5, 18, 63, 216, 767, 2672, 9341, 32603, 113795, 397185, 1386246, 4838570, 16888589, 58947286, 205746563, 718126388], dtype=np.int64)
C2_D3 = np.array([2, 6, 20, 71, 248, 864, 3016, 10528, 36752, 128262, 447678, 1562677, 5453959, 19035870, 66441396, 231902939], dtype=np.int64)
C3_D3 = np.array([0, 1, 4, 15, 51, 179, 623, 2176, 7594, 26507, 92517, 322812, 1126959, 3933792, 13730603, 47924832], dtype=np.int64)

def _exp_qcom16_poly_q31_from_dq(dq: np.ndarray, scale_q30: int) -> np.ndarray:
    """16 equal segments, degree-4 polynomial exp approximation in Q31."""
    x_q30 = dq.astype(np.int64) * np.int64(scale_q30)
    x_q30 = np.clip(x_q30, X_MIN_Q30, 0).astype(np.int64)

    x_shift = (x_q30 - np.int64(X_MIN_Q30)).astype(np.int64)
    seg = (x_shift * np.int64(16 * INV_XRNG_Q48)) >> 48
    seg = np.clip(seg, 0, 15).astype(np.int32)

    mid = X_MID_Q30[seg]
    num = ((x_q30 - mid) << Q15).astype(np.int64)
    abs_num = np.abs(num)
    u_mag = (abs_num * np.int64(INV_HALF_Q48)) >> 48
    u_q15 = np.where(num < 0, -u_mag, u_mag).astype(np.int64)
    u_q15 = np.clip(u_q15, -32768, 32767).astype(np.int64)

    c0 = C0[seg]; c1 = C1[seg]; c2 = C2[seg]; c3 = C3[seg]; c4 = C4[seg]

    y = c4
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c3)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c2)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c1)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c0)

    y = np.clip(y, 0, INT32_MAX).astype(np.uint32)
    return y

def _exp_qcom16_poly_deg3_q31_from_dq(dq: np.ndarray, scale_q30: int) -> np.ndarray:
    """16 equal segments, degree-3 polynomial exp approximation in Q31 (NEW tables)."""
    x_q30 = dq.astype(np.int64) * np.int64(scale_q30)
    x_q30 = np.clip(x_q30, X_MIN_Q30, 0).astype(np.int64)

    x_shift = (x_q30 - np.int64(X_MIN_Q30)).astype(np.int64)
    seg = (x_shift * np.int64(16 * INV_XRNG_Q48)) >> 48
    seg = np.clip(seg, 0, 15).astype(np.int32)

    mid = X_MID_Q30[seg]
    num = ((x_q30 - mid) << Q15).astype(np.int64)
    abs_num = np.abs(num)
    u_mag = (abs_num * np.int64(INV_HALF_Q48)) >> 48
    u_q15 = np.where(num < 0, -u_mag, u_mag).astype(np.int64)
    u_q15 = np.clip(u_q15, -32768, 32767).astype(np.int64)

    c0 = C0_D3[seg]; c1 = C1_D3[seg]; c2 = C2_D3[seg]; c3 = C3_D3[seg]

    y = c3
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c2)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c1)
    y = _sat_int32(_mul_q31_q15_rnd_sat32(y, u_q15) + c0)

    y = np.clip(y, 0, INT32_MAX).astype(np.uint32)
    return y

# ---------------- Your proposed method: variable segments/degree + BST ----------------
@dataclass
class MyApproxParams:
    S: int
    b_q30: np.ndarray        # (S+1,) int64 boundaries in Q30, increasing
    mid_q30: np.ndarray      # (S,)   int64
    half_q30: np.ndarray     # (S,)   int64
    inv_half_q48: np.ndarray # (S,)   int64 floor(2^48/half)
    # coefficients in Q31, power basis in u, always length 6 arrays (deg0..5)
    c0: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    c4: np.ndarray
    c5: np.ndarray

def load_my_approx_obj(obj: dict) -> MyApproxParams:
    boundaries = np.asarray(obj["boundaries"], dtype=np.float64).reshape(-1)
    if boundaries.size < 2:
        raise ValueError("boundaries must have length >= 2")
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError("boundaries must be strictly increasing")

    S = boundaries.size - 1
    if S < 1 or S > 16:
        raise ValueError(f"S={S} out of expected range 1..16")

    b_q30 = np.rint(boundaries * (1 << Q30)).astype(np.int64)

    lo = b_q30[:-1]
    hi = b_q30[1:]
    mid_q30 = (lo + hi) // 2
    half_q30 = (hi - lo) // 2

    inv_half_q48 = np.zeros((S,), dtype=np.int64)
    for s in range(S):
        h = int(half_q30[s])
        if h > 0:
            inv_half_q48[s] = (1 << 48) // h
        else:
            inv_half_q48[s] = 0

    coeffs_q31 = obj.get("coeffs_q31", None)
    if coeffs_q31 is None:
        raise ValueError("config must include coeffs_q31 (Q31 ints)")

    if len(coeffs_q31) != S:
        raise ValueError(f"coeffs_q31 has {len(coeffs_q31)} segments but boundaries imply S={S}")

    c0 = np.zeros((S,), dtype=np.int64)
    c1 = np.zeros((S,), dtype=np.int64)
    c2 = np.zeros((S,), dtype=np.int64)
    c3 = np.zeros((S,), dtype=np.int64)
    c4 = np.zeros((S,), dtype=np.int64)
    c5 = np.zeros((S,), dtype=np.int64)

    for s in range(S):
        cs = list(coeffs_q31[s])
        if len(cs) < 1 or len(cs) > 6:
            raise ValueError(f"segment {s}: coeffs length must be 1..6 (deg 0..5)")
        c0[s] = int(cs[0])
        if len(cs) > 1: c1[s] = int(cs[1])
        if len(cs) > 2: c2[s] = int(cs[2])
        if len(cs) > 3: c3[s] = int(cs[3])
        if len(cs) > 4: c4[s] = int(cs[4])
        if len(cs) > 5: c5[s] = int(cs[5])

    return MyApproxParams(
        S=S, b_q30=b_q30, mid_q30=mid_q30, half_q30=half_q30, inv_half_q48=inv_half_q48,
        c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5
    )

def load_my_approx_bank_from_coeffs_tree_json(path: str) -> Dict[Tuple[int, int], MyApproxParams]:
    """
    Loads a coeff-bank JSON in this format:

      {
        ...,
        "coeffs_tree": {
          "L00": {
            "H00": { "segments":4, "boundaries":[...], "coeffs_q31":[...], ... },
            "H01": { ... }
          },
          "L01": { ... }
        }
      }

    Returns:
      bank[(layer_idx, head_idx)] = MyApproxParams(...)
    """
    with open(path, "r") as f:
        root = json.load(f)

    if "coeffs_tree" not in root:
        raise ValueError("Expected top-level key 'coeffs_tree' in my-approx json")

    tree = root["coeffs_tree"]
    bank: Dict[Tuple[int, int], MyApproxParams] = {}

    for Lk, heads_dict in tree.items():
        if not isinstance(Lk, str) or not Lk.startswith("L"):
            continue
        layer_idx = int(Lk[1:])  # "L00" -> 0
        if not isinstance(heads_dict, dict):
            continue

        for Hk, cfg in heads_dict.items():
            if not isinstance(Hk, str) or not Hk.startswith("H"):
                continue
            head_idx = int(Hk[1:])  # "H00" -> 0
            if not isinstance(cfg, dict):
                continue

            P = load_my_approx_obj(cfg)
            bank[(layer_idx, head_idx)] = P

    if len(bank) == 0:
        raise RuntimeError("Loaded 0 (layer,head) entries from coeffs_tree. Check JSON structure.")
    return bank

def _exp_my_poly_q31_from_dq(dq: np.ndarray, scale_q30: int, P: MyApproxParams) -> np.ndarray:
    """
    Same integer flow as Qualcomm, but:
      - variable segment boundaries
      - segment chosen by BST (np.searchsorted on boundaries)
      - variable degree supported by storing c0..c5 with zeros for missing degrees
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

# ---------------- Quant / Dequant helpers ----------------
def quant_to_u16_torch(x_real: torch.Tensor, scale: float, offset: int) -> torch.Tensor:
    q = torch.round(x_real / float(scale) - float(offset)).to(torch.int64)
    q = torch.clamp(q, 0, 65535).to(torch.uint16)
    return q

def dequant_u16_np(q_u16: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (q_u16.astype(np.int32) + int(offset)) * float(scale)

# ---------------- GOLD softmax ----------------
def softmax_gold_u16_from_add2_u16(
    add2_u16: np.ndarray,
    axis: int = -1,
    add2_scale: float = None,
    add2_offset: int = None,
    softmax_scale: float = None,
    softmax_offset: int = 0) -> np.ndarray:
    assert add2_scale is not None and softmax_scale is not None
    x = (add2_u16.astype(np.int32) + int(add2_offset)) * float(add2_scale)
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    p = e / np.sum(e, axis=axis, keepdims=True)
    q = np.rint(p / float(softmax_scale) - float(softmax_offset)).astype(np.int64)
    return np.clip(q, 0, 65535).astype(np.uint16)

# ---------------- APPROX softmax: shared shell ----------------
def _softmax_from_expq31(add2_u16: np.ndarray, axis: int, clamp_min: float,
                        add2_scale: float, softmax_scale: float,
                        exp_q31: np.ndarray, softmax_offset: int = 0) -> np.ndarray:
    sum_q31 = np.sum(exp_q31.astype(np.uint64), axis=axis, keepdims=True)
    sum_q31 = np.maximum(sum_q31, 1)

    scale_inv = int(round(1.0 / float(softmax_scale)))  # usually 65535
    num = exp_q31.astype(np.uint64) * np.uint64(scale_inv)
    prob_u16 = (num + (sum_q31 // 2)) // sum_q31

    if softmax_offset != 0:
        prob_u16 = prob_u16 - np.uint64(softmax_offset)

    return np.clip(prob_u16, 0, 65535).astype(np.uint16)

def softmax_qcom_u16_from_add2_u16(
    add2_u16: np.ndarray,
    axis: int = -1,
    clamp_min: float = -20.0,
    add2_scale: float = None,
    softmax_scale: float = None,
    softmax_offset: int = 0) -> np.ndarray:
    """16 equal segments, degree-4 exp tables."""
    assert add2_u16.dtype == np.uint16
    assert add2_scale is not None and softmax_scale is not None

    q = add2_u16.astype(np.int32)
    qmax = np.max(q, axis=axis, keepdims=True)
    dq = q - qmax

    dq_min = int(np.ceil(clamp_min / float(add2_scale)))
    dq = np.maximum(dq, dq_min)

    scale_q30 = int(round(float(add2_scale) * (1 << Q30)))
    exp_q31 = _exp_qcom16_poly_q31_from_dq(dq, scale_q30=scale_q30)

    return _softmax_from_expq31(add2_u16, axis, clamp_min, add2_scale, softmax_scale, exp_q31, softmax_offset)

def softmax_qcom3_u16_from_add2_u16(
    add2_u16: np.ndarray,
    axis: int = -1,
    clamp_min: float = -20.0,
    add2_scale: float = None,
    softmax_scale: float = None,
    softmax_offset: int = 0) -> np.ndarray:
    """16 equal segments, degree-3 exp tables (NEW baseline)."""
    assert add2_u16.dtype == np.uint16
    assert add2_scale is not None and softmax_scale is not None

    q = add2_u16.astype(np.int32)
    qmax = np.max(q, axis=axis, keepdims=True)
    dq = q - qmax

    dq_min = int(np.ceil(clamp_min / float(add2_scale)))
    dq = np.maximum(dq, dq_min)

    scale_q30 = int(round(float(add2_scale) * (1 << Q30)))
    exp_q31 = _exp_qcom16_poly_deg3_q31_from_dq(dq, scale_q30=scale_q30)

    return _softmax_from_expq31(add2_u16, axis, clamp_min, add2_scale, softmax_scale, exp_q31, softmax_offset)

def softmax_my_u16_from_add2_u16(
    add2_u16: np.ndarray,
    P: MyApproxParams,
    axis: int = -1,
    clamp_min: float = -20.0,
    add2_scale: float = None,
    softmax_scale: float = None,
    softmax_offset: int = 0) -> np.ndarray:
    assert add2_u16.dtype == np.uint16
    assert add2_scale is not None and softmax_scale is not None

    q = add2_u16.astype(np.int32)
    qmax = np.max(q, axis=axis, keepdims=True)
    dq = q - qmax

    dq_min = int(np.ceil(clamp_min / float(add2_scale)))
    dq = np.maximum(dq, dq_min)

    scale_q30 = int(round(float(add2_scale) * (1 << Q30)))
    exp_q31 = _exp_my_poly_q31_from_dq(dq, scale_q30=scale_q30, P=P)

    return _softmax_from_expq31(add2_u16, axis, clamp_min, add2_scale, softmax_scale, exp_q31, softmax_offset)

# ---------------- Metrics ----------------
def compare(
    gold_u16: np.ndarray,
    approx_u16: np.ndarray,
    k: int = 5,
    softmax_scale: float = None,
    softmax_offset: int = 0,
    eps: float = 1e-12,
) -> dict:
    """
    Compare gold vs approx softmax outputs in u16 encoding.

    Inputs:
      gold_u16, approx_u16: [1,Q,K] uint16 (encoded with softmax_scale/offset)
      k: requested top-k for overlap metric (will be clamped to K)
      softmax_scale/offset: encoding parameters used to dequantize to probabilities

    Notes:
      - We renormalize rows to sum to 1 because u16 quantization can introduce small drift.
      - If k > K, we clamp to K and report both k_requested and k_used.
      - KL is computed in nats (natural log). We clip probs by eps for stability.
    """
    if softmax_scale is None:
        raise ValueError("softmax_scale must not be None")

    gold = dequant_u16_np(gold_u16, softmax_scale, softmax_offset).astype(np.float64)
    appr = dequant_u16_np(approx_u16, softmax_scale, softmax_offset).astype(np.float64)

    K = gold.shape[-1]
    gold = gold.reshape(-1, K)
    appr = appr.reshape(-1, K)

    gold = gold / np.maximum(gold.sum(axis=1, keepdims=True), eps)
    appr = appr / np.maximum(appr.sum(axis=1, keepdims=True), eps)

    top1 = float(np.mean(np.argmax(gold, axis=1) == np.argmax(appr, axis=1)))

    k_req = int(k)
    k_used = int(np.clip(k_req, 1, K))

    gold_topk = np.argpartition(-gold, kth=k_used - 1, axis=1)[:, :k_used]
    appr_topk = np.argpartition(-appr, kth=k_used - 1, axis=1)[:, :k_used]

    overlap = np.empty((gold.shape[0],), dtype=np.float64)
    for i in range(gold.shape[0]):
        overlap[i] = len(set(gold_topk[i]).intersection(set(appr_topk[i]))) / float(k_used)

    max_abs = float(np.max(np.abs(gold - appr)))

    diff_counts = approx_u16.astype(np.int32) - gold_u16.astype(np.int32)
    max_abs_u16 = int(np.max(np.abs(diff_counts)))
    num_nonzero_u16 = int(np.count_nonzero(diff_counts))
    total_u16 = int(diff_counts.size)

    p = np.clip(gold, eps, 1.0)
    q = np.clip(appr, eps, 1.0)
    kl_pq = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    kl_qp = np.sum(q * (np.log(q) - np.log(p)), axis=1)

    return {
        "rows": int(gold.shape[0]),
        "k_requested": k_req,
        "k_used": k_used,
        "top1_match_rate": top1,
        "topk_set_overlap_mean": float(np.mean(overlap)),
        "max_abs_prob_error": max_abs,
        "max_abs_u16_counts": max_abs_u16,
        "num_nonzero_u16": num_nonzero_u16,
        "total_u16": total_u16,
        "kl_gold_to_approx_mean": float(np.mean(kl_pq)),
        "kl_gold_to_approx_max": float(np.max(kl_pq)),
        "kl_approx_to_gold_mean": float(np.mean(kl_qp)),
        "kl_approx_to_gold_max": float(np.max(kl_qp)),
    }

# ---------------- Capture ALL layers (by call order) ----------------
class SoftmaxCatcherAllLayers:
    """
    Captures every softmax input tensor with shape [B,H,Q,K] matching Q/K.
    Stores them in order; assumes order corresponds to layer order.
    """
    def __init__(self, target_q: int, target_k: int, max_layers: int, encodings: dict, debug_print: bool = False):
        self.target_q = target_q
        self.target_k = target_k
        self.max_layers = max_layers
        self.encodings = encodings
        self.debug_print = debug_print
        self.captured_list = []
        self._orig = None

    def _patched(self, x, dim=None, _stacklevel=3, dtype=None):
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            B, H, Q, K = x.shape
            if Q == self.target_q and K == self.target_k:
                if len(self.captured_list) < self.max_layers:
                    layer_idx = len(self.captured_list)
                    e_add2 = get_layer_add2_encoding(self.encodings, layer_idx)

                    x_cpu = x.detach().cpu().to(torch.float32)
                    q_u16 = quant_to_u16_torch(x_cpu, e_add2["scale"], e_add2["offset"]).numpy()
                    self.captured_list.append(q_u16)
                    print(f"[capture] layer_idx {layer_idx}: quantized using Add2(scale={e_add2['scale']}, offset={e_add2['offset']}) -> {q_u16.shape}")
        return self._orig(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    def __enter__(self):
        self._orig = F.softmax
        F.softmax = self._patched
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        F.softmax = self._orig

# ---------------- Build inputs to force Q=chunk_len, K=context_len ----------------
def build_ctx_token_ids_from_prompt(tokenizer, prompt_text: str, total_len: int, device: str,
                                    truncate_side: str = "tail"):
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(ids) == 0:
        ids = [tokenizer.eos_token_id]

    if len(ids) >= total_len:
        ids = ids[:total_len] if truncate_side == "head" else ids[-total_len:]
    else:
        reps = (total_len + len(ids) - 1) // len(ids)
        ids = (ids * reps)[:total_len]

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attn

def force_eager_attention(model):
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="./workdir/emulation_metrics")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--chunk-len", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=32)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--debug-softmax-shapes", action="store_true")
    ap.add_argument("--encoding-json", required=True)

    ap.add_argument("--methods", type=str, default="qcom",
                    help="Comma-separated: qcom,qcom3,my (e.g. --methods qcom,qcom3,my)")
    ap.add_argument("--my-approx-json", type=str, default=None,
                    help="Required if 'my' is in --methods. JSON with coeffs_tree (Lxx/Hyy entries).")

    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--truncate-side", type=str, default="tail", choices=["head", "tail"])
    args = ap.parse_args()

    _load_runtime_dependencies()

    os.makedirs(args.outdir, exist_ok=True)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("qcom", "qcom3", "my"):
            raise ValueError(f"Unknown method '{m}'. Use qcom, qcom3, or my.")

    my_bank = None
    if "my" in methods:
        if args.my_approx_json is None:
            raise ValueError("--my-approx-json is required when using method 'my'")
        my_bank = load_my_approx_bank_from_coeffs_tree_json(args.my_approx_json)
        print(f"[info] loaded my-approx bank entries: {len(my_bank)} (layer,head)")

    print("[info] loading:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    enc = _load_encodings(args.encoding_json)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    force_eager_attention(model)

    ctx = args.context_len
    qlen = args.chunk_len
    assert ctx > qlen, "context-len must be > chunk-len"
    prefix_len = ctx - qlen

    # prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    elif args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = "Hello! Please summarize the following text and explain key points."

    input_ids, full_mask = build_ctx_token_ids_from_prompt(
        tok, prompt_text=prompt_text, total_len=ctx, device=args.device, truncate_side=args.truncate_side
    )
    prefix_ids = input_ids[:, :prefix_len]
    chunk_ids = input_ids[:, prefix_len:ctx]
    prefix_mask = full_mask[:, :prefix_len]
    chunk_full_mask = full_mask

    # Save spec
    e0_add2 = get_layer_add2_encoding(enc, 0)
    e0_smx = get_layer_softmax_encoding(enc, 0)
    spec = {
        "encoding_json": args.encoding_json,
        "example_layer0_add2": e0_add2,
        "example_layer0_softmax": e0_smx,
        "methods": methods,
        "my_approx_json": args.my_approx_json,
        "axis": -1,
        "context_len": ctx,
        "prefix_len": prefix_len,
        "chunk_len": qlen,
        "max_layers": args.max_layers,
        "topk": args.topk,
    }
    with open(os.path.join(args.outdir, "softmax_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # 1) prefix pass to build cache
    with torch.no_grad():
        out1 = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True)
        past = out1.past_key_values

    # 2) chunk pass capture
    with SoftmaxCatcherAllLayers(target_q=qlen, target_k=ctx, max_layers=args.max_layers, encodings=enc,
                                 debug_print=args.debug_softmax_shapes) as catcher:
        with torch.no_grad():
            _ = model(input_ids=chunk_ids, attention_mask=chunk_full_mask, past_key_values=past, use_cache=True)

    if len(catcher.captured_list) == 0:
        raise RuntimeError("Captured 0 matching softmax calls. Try --debug-softmax-shapes or verify Q/K.")

    num_layers = min(len(catcher.captured_list), args.max_layers)
    print(f"[info] captured layers: {len(catcher.captured_list)} (reporting {num_layers})")

    # Compute metrics per (layer, head, method)
    csv_rows = []
    for layer_idx in range(num_layers):
        add2_u16_layer = catcher.captured_list[layer_idx]  # [B,H,Q,K] u16
        B, Hh, Q, K = add2_u16_layer.shape
        assert B == 1, "This script assumes batch=1 for metrics."

        e_add2 = get_layer_add2_encoding(enc, layer_idx)
        e_smx  = get_layer_softmax_encoding(enc, layer_idx)

        for head in range(Hh):
            add2_u16 = add2_u16_layer[0, head, :, :]  # [Q,K]
            add2_u16 = add2_u16[None, :, :]           # [1,Q,K]

            gold_u16 = softmax_gold_u16_from_add2_u16(
                add2_u16,
                add2_scale=e_add2["scale"], add2_offset=e_add2["offset"],
                softmax_scale=e_smx["scale"], softmax_offset=e_smx["offset"],
                axis=-1
            )

            for method in methods:
                if method == "qcom":
                    approx_u16 = softmax_qcom_u16_from_add2_u16(
                        add2_u16,
                        add2_scale=e_add2["scale"],
                        softmax_scale=e_smx["scale"],
                        axis=-1,
                        clamp_min=CLAMP_MIN,
                        softmax_offset=e_smx["offset"],
                    )
                elif method == "qcom3":
                    approx_u16 = softmax_qcom3_u16_from_add2_u16(
                        add2_u16,
                        add2_scale=e_add2["scale"],
                        softmax_scale=e_smx["scale"],
                        axis=-1,
                        clamp_min=CLAMP_MIN,
                        softmax_offset=e_smx["offset"],
                    )
                else:
                    if my_bank is None:
                        raise RuntimeError("Internal error: my_bank is None but method 'my' selected.")
                    key = (layer_idx, head)
                    if key not in my_bank:
                        raise KeyError(f"Missing my-approx config for layer={layer_idx}, head={head} in coeffs_tree json.")
                    P = my_bank[key]

                    approx_u16 = softmax_my_u16_from_add2_u16(
                        add2_u16,
                        P=P,
                        add2_scale=e_add2["scale"],
                        softmax_scale=e_smx["scale"],
                        axis=-1,
                        clamp_min=CLAMP_MIN,
                        softmax_offset=e_smx["offset"],
                    )

                m = compare(
                    gold_u16, approx_u16, k=args.topk,
                    softmax_scale=e_smx["scale"],
                    softmax_offset=e_smx["offset"],
                )
                m.update({
                    "method": method,
                    "layer": layer_idx,
                    "head": head,
                    "Q": int(Q),
                    "K": int(K),
                })
                csv_rows.append(m)

    out_csv = os.path.join(args.outdir, "compare_metrics.csv")
    fieldnames = [
        "method","layer","head","Q","K","rows","k",
        "top1_match_rate","topk_set_overlap_mean",
        "max_abs_prob_error","max_abs_u16_counts","num_nonzero_u16","total_u16",
        "kl_gold_to_approx_mean","kl_gold_to_approx_max",
        "kl_approx_to_gold_mean","kl_approx_to_gold_max","k_requested","k_used"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"[write] {out_csv}")

    # Quick summary per method
    for method in methods:
        sub = [r for r in csv_rows if r["method"] == method]
        top1s = [r["top1_match_rate"] for r in sub]
        overlaps = [r["topk_set_overlap_mean"] for r in sub]
        kls = [r["kl_gold_to_approx_mean"] for r in sub]
        print(f"[summary:{method}] top1 mean={float(np.mean(top1s)):.4f} min={float(np.min(top1s)):.4f} max={float(np.max(top1s)):.4f}")
        print(f"[summary:{method}] topk mean={float(np.mean(overlaps)):.4f} min={float(np.min(overlaps)):.4f} max={float(np.max(overlaps)):.4f}")
        print(f"[summary:{method}] KL(p||q) mean={float(np.mean(kls)):.6e} min={float(np.min(kls)):.6e} max={float(np.max(kls)):.6e}")

    print("Done.")

if __name__ == "__main__":
    main()
