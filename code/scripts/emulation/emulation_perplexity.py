#!/usr/bin/env python3
from __future__ import annotations

"""
GPU Perplexity evaluation for attention-softmax approximation:
  fp vs qcom (uniform 16-seg deg-4) vs qcom3 (uniform 16-seg deg-3) vs my (proposed).

✅ Fully GPU path (NO numpy, NO .cpu() per softmax call)
✅ Patches ONLY attention softmax calls with shape [B,H,Q,K] where Q==chunk_len and K==context_len
✅ Prefix pass (context_len - chunk_len) runs normal softmax (unpatched)
✅ Chunk pass runs patched softmax per method

DEBUG FEATURES (env vars, no code edits needed):
  DEBUG_PATCH=1        -> print patch events and patch summary
  DEBUG_WINDOW=71      -> only print debug for this window_idx (0-based)
  ONLY_LAYER=21        -> patch ONLY this layer (other layers use FP softmax)
  ONLY_HEAD=5          -> patch ONLY this head (requires ONLY_LAYER; others use FP softmax)
  DEBUG_SANITY=1       -> NaN/Inf + row-sum checks
  DEBUG_EXP=1          -> exp saturation stats (zeros, INT32_MAX)
  DEBUG_LIMIT=4        -> max debug prints per window per layer

Run (GPU):
  python3 scripts/emulation/emulation_perplexity.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --encoding-json configs/encodings/tinyllama.encodings \
    --my-approx-json workdir/coeff_banks/coeffs_q31_all_LH_lut128.json \
    --context-len 2048 --chunk-len 128 --device cuda \
    --methods fp,qcom,qcom3,my --dtype fp16 \
    --max-tokens 200000  --max-windows 200 \
    --outdir ./workdir/perplexity \
    --save-csv
"""

import os, json, argparse, csv, math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

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

# ---------------- constants ----------------
Q30 = 30
Q15 = 15
CLAMP_MIN = -20.0

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

# Global current-layer tracker set by hooks
CURRENT_LAYER = {"idx": -1}

# qcom exp poly constants (Q30 clamp range [-20,0])
X_MIN_Q30     = -21474836480  # -20 * 2^30
X_RNG_Q30     =  21474836480  #  20 * 2^30
HALF_W_Q30    =   671088640   #  0.625 * 2^30
INV_XRNG_Q48  =       13107   # floor(2^48 / X_RNG_Q30)
INV_HALF_Q48  =      419430   # floor(2^48 / HALF_W_Q30)

# ---------------- Qualcomm tables (uniform 16 segments) ----------------
# Segment midpoints (Q30) shared across deg-4 and deg-3
X_MID_Q30 = [
    -20803747840, -19461570560, -18119393280, -16777216000,
    -15435038720, -14092861440, -12750684160, -11408506880,
    -10066329600,  -8724152320,  -7381975040,  -6039797760,
     -4697620480,  -3355443200,  -2013265920,   -671088640
]

# Deg-4 coefficients (Q31) (existing baseline)
C0_D4 = [8, 29, 101, 352, 1227, 4284, 14951, 52186, 182146, 635752, 2218994, 7745050, 27032879, 94354019, 329327887, 1149467273]
C1_D4 = [5, 18, 63, 220, 767, 2676, 9342, 32606, 113806, 397222, 1386440, 4839153, 16890302, 58952946, 205766001, 718193911]
C2_D4 = [2, 6, 20, 69, 240, 836, 2920, 10191, 35568, 124146, 433311, 1512406, 5278814, 18424871, 64309120, 224460885]
C3_D4 = [0, 1, 4, 15, 51, 178, 622, 2170, 7574, 26437, 92276, 322074, 1124147, 3923659, 13694914, 47799948]
C4_D4 = [0, 0, 1, 2, 8, 28, 97, 338, 1179, 4115, 14361, 50126, 174957, 610659, 2131410, 7439353]

# Deg-3 coefficients (Q31) (uniform 16 segments, degree 3)
C0_D3 = [8, 29, 101, 351, 1226, 4280, 14943, 52147, 182007, 635274, 2217261, 7738907, 27011122, 94277759, 329061395, 1148536803]
C1_D3 = [5, 18, 63, 216, 767, 2672, 9341, 32603, 113795, 397185, 1386246, 4838570, 16888589, 58947286, 205746563, 718126388]
C2_D3 = [2, 6, 20, 71, 248, 864, 3016, 10528, 36752, 128262, 447678, 1562677, 5453959, 19035870, 66441396, 231902939]
C3_D3 = [0, 1, 4, 15, 51, 179, 623, 2176, 7594, 26507, 92517, 322812, 1126959, 3933792, 13730603, 47924832]


# ---------------- debug knobs ----------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

DEBUG_PATCH  = bool(_env_int("DEBUG_PATCH", 0))
DEBUG_WINDOW = _env_int("DEBUG_WINDOW", -1)
DEBUG_SANITY = bool(_env_int("DEBUG_SANITY", 0))
DEBUG_EXP    = bool(_env_int("DEBUG_EXP", 0))
DEBUG_LIMIT  = _env_int("DEBUG_LIMIT", 4)

ONLY_LAYER_ENV = os.getenv("ONLY_LAYER")
ONLY_HEAD_ENV  = os.getenv("ONLY_HEAD")
ONLY_LAYER = int(ONLY_LAYER_ENV) if ONLY_LAYER_ENV is not None else None
ONLY_HEAD  = int(ONLY_HEAD_ENV)  if ONLY_HEAD_ENV  is not None else None

if ONLY_HEAD is not None and ONLY_LAYER is None:
    print("[warn] ONLY_HEAD is set but ONLY_LAYER is not set. ONLY_HEAD will be ignored.")
    ONLY_HEAD = None


# ---------------- Encodings ----------------
def _load_encodings(path: str) -> dict:
    with open(path, "r") as f:
        enc = json.load(f)
    return enc["activation_encodings"] if "activation_encodings" in enc else enc

def _get_enc(enc_act: dict, key: str) -> dict:
    if key not in enc_act:
        raise KeyError(f"Encoding key not found: {key}")
    e0 = enc_act[key][0]
    return {"scale": float(e0["scale"]), "offset": int(e0["offset"])}

def get_layer_add2_encoding(enc: dict, layer_idx: int) -> dict:
    return _get_enc(enc, f"/model_layers_{layer_idx}_self_attn_Add_2/Add_output_0")

def get_layer_softmax_encoding(enc: dict, layer_idx: int) -> dict:
    return _get_enc(enc, f"/model_layers_{layer_idx}_self_attn_Softmax/Softmax_output_0")


# ---------------- My bank format (coeffs_tree Lxx/Hyy) ----------------
@dataclass
class MyApproxParamsTorch:
    S: int
    b_q30: torch.Tensor        # [S+1] int64, on device
    mid_q30: torch.Tensor      # [S]   int64
    inv_half_q48: torch.Tensor # [S]   int64
    c0: torch.Tensor           # [S] int64
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    c5: torch.Tensor

def _make_inv_half_q48(half_q30: List[int]) -> List[int]:
    out = []
    for h in half_q30:
        out.append((1 << 48) // h if h > 0 else 0)
    return out

def _load_my_params_obj(cfg: dict, device: torch.device) -> MyApproxParamsTorch:
    boundaries = cfg["boundaries"]
    if len(boundaries) < 2:
        raise ValueError("boundaries too short")

    b_q30 = [int(round(b * (1 << Q30))) for b in boundaries]
    S = len(b_q30) - 1

    lo = b_q30[:-1]
    hi = b_q30[1:]
    mid_q30 = [(l + h) // 2 for l, h in zip(lo, hi)]
    half_q30 = [(h - l) // 2 for l, h in zip(lo, hi)]
    inv_half_q48 = _make_inv_half_q48(half_q30)

    coeffs_q31 = cfg["coeffs_q31"]
    if len(coeffs_q31) != S:
        raise ValueError("coeffs_q31 length mismatch vs boundaries")

    c0 = [0]*S; c1=[0]*S; c2=[0]*S; c3=[0]*S; c4=[0]*S; c5=[0]*S
    for s in range(S):
        cs = list(coeffs_q31[s])
        c0[s] = int(cs[0])
        if len(cs) > 1: c1[s] = int(cs[1])
        if len(cs) > 2: c2[s] = int(cs[2])
        if len(cs) > 3: c3[s] = int(cs[3])
        if len(cs) > 4: c4[s] = int(cs[4])
        if len(cs) > 5: c5[s] = int(cs[5])

    t = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
    return MyApproxParamsTorch(
        S=S,
        b_q30=t(b_q30),
        mid_q30=t(mid_q30),
        inv_half_q48=t(inv_half_q48),
        c0=t(c0), c1=t(c1), c2=t(c2), c3=t(c3), c4=t(c4), c5=t(c5)
    )

def load_my_bank_coeffs_tree(path: str, device: torch.device) -> Dict[Tuple[int,int], MyApproxParamsTorch]:
    with open(path, "r") as f:
        root = json.load(f)
    tree = root["coeffs_tree"]
    bank: Dict[Tuple[int,int], MyApproxParamsTorch] = {}
    for Lk, heads in tree.items():
        if not Lk.startswith("L"):
            continue
        li = int(Lk[1:])
        for Hk, cfg in heads.items():
            if not Hk.startswith("H"):
                continue
            hi = int(Hk[1:])
            bank[(li, hi)] = _load_my_params_obj(cfg, device=device)
    if not bank:
        raise RuntimeError("Loaded empty my_bank")
    return bank

def validate_my_bank(my_bank: Dict[Tuple[int,int], MyApproxParamsTorch], name="my_bank"):
    bad = 0
    min_width = None
    for (L,H), P in sorted(my_bank.items()):
        b = P.b_q30.detach().cpu().tolist()
        for i in range(len(b)-1):
            w = b[i+1] - b[i]
            if min_width is None or w < min_width:
                min_width = w
            if w <= 0:
                print(f"[{name}][BAD_BOUNDARY] L{L:02d} H{H:02d}: b[{i}]={b[i]} b[{i+1}]={b[i+1]} width={w}")
                bad += 1
                break
        invh = P.inv_half_q48.detach().cpu().tolist()
        if any(v == 0 for v in invh):
            zc = sum(1 for v in invh if v == 0)
            print(f"[{name}][ZERO_INV_HALF] L{L:02d} H{H:02d}: {zc}/{len(invh)} segments have inv_half=0")
            bad += 1
        if b[0] > X_MIN_Q30 or b[-1] < 0:
            print(f"[{name}][RANGE_WARN] L{L:02d} H{H:02d}: range_q30=[{b[0]},{b[-1]}] expected cover [{X_MIN_Q30},0]")
    print(f"[{name}] validate done. bad_flags={bad}, min_segment_width_q30={min_width}")


# ---------------- integer math helpers (torch) ----------------
def _sat_int32_t(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, INT32_MIN, INT32_MAX)

def _mul_q31_q15_rnd_sat32_t(y_q31: torch.Tensor, u_q15: torch.Tensor) -> torch.Tensor:
    prod = y_q31 * u_q15  # Q46
    rnd = (1 << 14)
    prod = prod + torch.where(prod >= 0, rnd, -rnd)
    out = prod >> 15
    return _sat_int32_t(out)

def _gather_1d(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return table.index_select(0, idx.reshape(-1)).reshape(idx.shape)


# ---------------- exp approximation (qcom uniform 16 seg) torch ----------------
def _qcom16_common_u_q15_and_seg(dq: torch.Tensor, scale_q30: int, X_MID: torch.Tensor):
    # dq is int64 <=0, clamp already applied by caller
    x_q30 = dq * int(scale_q30)
    x_q30 = torch.clamp(x_q30, X_MIN_Q30, 0)

    x_shift = x_q30 - X_MIN_Q30
    seg = (x_shift * int(16 * INV_XRNG_Q48)) >> 48
    seg = torch.clamp(seg, 0, 15).to(torch.int64)

    mid = _gather_1d(X_MID, seg)
    num = (x_q30 - mid) << Q15
    abs_num = torch.abs(num)
    u_mag = (abs_num * int(INV_HALF_Q48)) >> 48
    u_q15 = torch.where(num < 0, -u_mag, u_mag)
    u_q15 = torch.clamp(u_q15, -32768, 32767).to(torch.int64)
    return seg, u_q15

def exp_qcom16_deg4_poly_q31_from_dq_t(
    dq: torch.Tensor, scale_q30: int,
    X_MID: torch.Tensor, C0t: torch.Tensor, C1t: torch.Tensor, C2t: torch.Tensor,
    C3t: torch.Tensor, C4t: torch.Tensor,
) -> torch.Tensor:
    seg, u_q15 = _qcom16_common_u_q15_and_seg(dq, scale_q30, X_MID)

    c0 = _gather_1d(C0t, seg)
    c1 = _gather_1d(C1t, seg)
    c2 = _gather_1d(C2t, seg)
    c3 = _gather_1d(C3t, seg)
    c4 = _gather_1d(C4t, seg)

    y = c4
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c3)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c2)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c1)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c0)

    y = torch.clamp(y, 0, INT32_MAX)
    return y

def exp_qcom16_deg3_poly_q31_from_dq_t(
    dq: torch.Tensor, scale_q30: int,
    X_MID: torch.Tensor, C0t: torch.Tensor, C1t: torch.Tensor, C2t: torch.Tensor,
    C3t: torch.Tensor,
) -> torch.Tensor:
    seg, u_q15 = _qcom16_common_u_q15_and_seg(dq, scale_q30, X_MID)

    c0 = _gather_1d(C0t, seg)
    c1 = _gather_1d(C1t, seg)
    c2 = _gather_1d(C2t, seg)
    c3 = _gather_1d(C3t, seg)

    # Horner deg-3: (((c3*u)+c2)*u + c1)*u + c0
    y = c3
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c2)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c1)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c0)

    y = torch.clamp(y, 0, INT32_MAX)
    return y


# ---------------- exp approximation (my) torch ----------------
def exp_my_poly_q31_from_dq_t(dq: torch.Tensor, scale_q30: int, P: MyApproxParamsTorch,
                             dbg: Optional[dict] = None) -> torch.Tensor:
    x_q30 = dq * int(scale_q30)
    x_q30 = torch.clamp(x_q30, P.b_q30[0].item(), P.b_q30[-1].item())

    seg = torch.searchsorted(P.b_q30, x_q30, right=True) - 1
    seg = torch.clamp(seg, 0, P.S - 1).to(torch.int64)

    mid = _gather_1d(P.mid_q30, seg)
    num = (x_q30 - mid) << Q15
    abs_num = torch.abs(num)
    invh = _gather_1d(P.inv_half_q48, seg)
    u_mag = (abs_num * invh) >> 48
    u_q15 = torch.where(num < 0, -u_mag, u_mag)
    u_q15 = torch.clamp(u_q15, -32768, 32767).to(torch.int64)

    c0 = _gather_1d(P.c0, seg)
    c1 = _gather_1d(P.c1, seg)
    c2 = _gather_1d(P.c2, seg)
    c3 = _gather_1d(P.c3, seg)
    c4 = _gather_1d(P.c4, seg)
    c5 = _gather_1d(P.c5, seg)

    y = c5
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c4)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c3)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c2)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c1)
    y = _sat_int32_t(_mul_q31_q15_rnd_sat32_t(y, u_q15) + c0)

    y = torch.clamp(y, 0, INT32_MAX)

    if dbg is not None and dbg.get("do_print", False) and DEBUG_EXP:
        sat = int((y == INT32_MAX).sum().item())
        zer = int((y == 0).sum().item())
        if sat or zer:
            print(f"[MY_EXP] win={dbg['win']} L={dbg['L']:02d} H={dbg['H']:02d} sat={sat} zero={zer} shape={tuple(y.shape)}")

    return y


# ---------------- approx attention softmax -> float probs (torch) ----------------
def quant_to_u16_torch(x_real: torch.Tensor, scale: float, offset: int) -> torch.Tensor:
    lo = (0 + offset) * scale
    hi = (65535 + offset) * scale
    x = torch.clamp(x_real, lo, hi)
    q = torch.round(x / float(scale) - float(offset)).to(torch.int64)
    q = torch.clamp(q, 0, 65535).to(torch.int64)
    return q

def probs_from_expq31_t(exp_q31: torch.Tensor, softmax_scale: float, softmax_offset: int,
                        dbg: Optional[dict] = None) -> torch.Tensor:
    sum_q31 = torch.sum(exp_q31, dim=-1, keepdim=True)
    sum_q31 = torch.clamp(sum_q31, min=1)

    scale_inv = int(round(1.0 / float(softmax_scale)))
    num = exp_q31 * scale_inv
    prob_u16 = (num + (sum_q31 // 2)) // sum_q31

    if softmax_offset != 0:
        prob_u16 = prob_u16 - int(softmax_offset)

    prob_u16 = torch.clamp(prob_u16, 0, 65535).to(torch.int64)

    p = (prob_u16 + int(softmax_offset)) * float(softmax_scale)
    p = p.to(torch.float32)
    p = p / torch.clamp(torch.sum(p, dim=-1, keepdim=True), min=1e-12)

    if dbg is not None and dbg.get("do_print", False) and DEBUG_SANITY:
        if not torch.isfinite(p).all():
            print(f"[PROB][NONFINITE] win={dbg['win']} L={dbg['L']:02d} H={dbg.get('H',-1):02d}")
        rs = p.sum(dim=-1)
        print(f"[PROB][ROWSUM] win={dbg['win']} L={dbg['L']:02d} H={dbg.get('H',-1):02d} min={float(rs.min()):.6f} max={float(rs.max()):.6f}")

    return p


def approx_attention_probs_gpu(
    x: torch.Tensor,
    layer_idx: int,
    method: str,
    enc: dict,
    my_bank: Optional[Dict[Tuple[int,int], MyApproxParamsTorch]],
    tables: dict,
    window_idx: int,
) -> torch.Tensor:
    """
    x: [B,H,Q,K] float attention scores (pre-softmax)
    returns: [B,H,Q,K] float probs
    """
    B, Hh, Q, K = x.shape
    dev = x.device

    e_add2 = get_layer_add2_encoding(enc, layer_idx)
    e_smx  = get_layer_softmax_encoding(enc, layer_idx)

    add2_scale = float(e_add2["scale"])
    add2_offset = int(e_add2["offset"])
    smx_scale = float(e_smx["scale"])
    smx_offset = int(e_smx["offset"])

    q = quant_to_u16_torch(x, add2_scale, add2_offset)
    qmax = torch.max(q, dim=-1, keepdim=True).values
    dq = q - qmax  # <= 0

    dq_min = int(math.ceil(CLAMP_MIN / add2_scale))
    dq = torch.maximum(dq, torch.tensor(dq_min, device=dev, dtype=torch.int64))

    scale_q30 = int(round(add2_scale * (1 << Q30)))

    do_print = DEBUG_PATCH and (DEBUG_WINDOW < 0 or window_idx == DEBUG_WINDOW)
    dbg_common = {"win": window_idx, "L": layer_idx, "do_print": do_print}

    # Head ablation support (debug-only): patch only head H, use fp softmax for others
    only_head = ONLY_HEAD if (ONLY_LAYER is not None and layer_idx == ONLY_LAYER) else None

    if method == "qcom":
        exp_q31 = exp_qcom16_deg4_poly_q31_from_dq_t(
            dq=dq, scale_q30=scale_q30,
            X_MID=tables["X_MID_Q30"],
            C0t=tables["C0_D4"], C1t=tables["C1_D4"], C2t=tables["C2_D4"], C3t=tables["C3_D4"], C4t=tables["C4_D4"]
        )
        probs = probs_from_expq31_t(exp_q31, smx_scale, smx_offset, dbg=dbg_common).to(x.dtype)
        return probs

    if method == "qcom3":
        exp_q31 = exp_qcom16_deg3_poly_q31_from_dq_t(
            dq=dq, scale_q30=scale_q30,
            X_MID=tables["X_MID_Q30"],
            C0t=tables["C0_D3"], C1t=tables["C1_D3"], C2t=tables["C2_D3"], C3t=tables["C3_D3"]
        )
        probs = probs_from_expq31_t(exp_q31, smx_scale, smx_offset, dbg=dbg_common).to(x.dtype)
        return probs

    # method == "my"
    assert my_bank is not None

    if only_head is not None:
        probs = torch.softmax(x, dim=-1)
        h = only_head
        P = my_bank[(layer_idx, h)]
        dq_h = dq[:, h, :, :]  # [B,Q,K]
        dbg_h = dict(dbg_common); dbg_h["H"] = h
        exp_q31_h = exp_my_poly_q31_from_dq_t(dq_h, scale_q30=scale_q30, P=P, dbg=dbg_h)
        p_h = probs_from_expq31_t(exp_q31_h, smx_scale, smx_offset, dbg=dbg_h).to(x.dtype)
        probs[:, h, :, :] = p_h
        return probs

    probs_heads = []
    for h in range(Hh):
        P = my_bank[(layer_idx, h)]
        dq_h = dq[:, h, :, :]
        dbg_h = dict(dbg_common); dbg_h["H"] = h
        exp_q31_h = exp_my_poly_q31_from_dq_t(dq_h, scale_q30=scale_q30, P=P, dbg=dbg_h)
        p_h = probs_from_expq31_t(exp_q31_h, smx_scale, smx_offset, dbg=dbg_h).to(x.dtype)
        probs_heads.append(p_h.unsqueeze(1))
    return torch.cat(probs_heads, dim=1)


# ---------------- Softmax patcher ----------------
class AttentionSoftmaxPatcherGPU:
    """
    Monkey-patches torch.nn.functional.softmax.
    Only patches attention tensors [B,H,Q,K] with Q==chunk_len and K==context_len and dim=-1.
    Uses CURRENT_LAYER hook to map patched call -> layer_idx.
    """
    def __init__(self, enc: dict, method: str, my_bank, context_len: int, chunk_len: int, tables: dict):
        self.enc = enc
        self.method = method
        self.my_bank = my_bank
        self.context_len = context_len
        self.chunk_len = chunk_len
        self.tables = tables
        self._orig = None

        self.layer_counter = 0
        self.current_window = -1

        self.calls_total = 0
        self.calls_layer_minus1 = 0
        self.calls_per_layer = defaultdict(int)
        self.calls_per_shape = defaultdict(int)
        self._dbg_prints_per_layer = defaultdict(int)

    def reset(self):
        self.layer_counter = 0
        self.calls_total = 0
        self.calls_layer_minus1 = 0
        self.calls_per_layer.clear()
        self.calls_per_shape.clear()
        self._dbg_prints_per_layer.clear()

    def _should_print(self, layer_idx: int) -> bool:
        if not DEBUG_PATCH:
            return False
        if DEBUG_WINDOW >= 0 and self.current_window != DEBUG_WINDOW:
            return False
        key = (self.current_window, layer_idx)
        if self._dbg_prints_per_layer[key] >= DEBUG_LIMIT:
            return False
        self._dbg_prints_per_layer[key] += 1
        return True

    def _patched(self, x, dim=None, _stacklevel=3, dtype=None):
        if isinstance(x, torch.Tensor) and x.dim() == 4 and (dim is None or dim == -1):
            B, H, Q, K = x.shape
            if Q == self.chunk_len and K == self.context_len:
                layer_idx = CURRENT_LAYER["idx"]
                self.calls_total += 1
                self.calls_per_shape[(B, H, Q, K)] += 1

                if layer_idx < 0:
                    self.calls_layer_minus1 += 1
                    layer_idx = self.layer_counter

                if ONLY_LAYER is not None and layer_idx != ONLY_LAYER:
                    return self._orig(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

                self.layer_counter += 1
                self.calls_per_layer[layer_idx] += 1

                if self._should_print(layer_idx):
                    print(f"[PATCH] win={self.current_window} method={self.method} L={layer_idx} shape={(B,H,Q,K)} "
                          f"counter={self.layer_counter-1} layer_minus1={self.calls_layer_minus1}")

                return approx_attention_probs_gpu(
                    x=x,
                    layer_idx=layer_idx,
                    method=self.method,
                    enc=self.enc,
                    my_bank=self.my_bank,
                    tables=self.tables,
                    window_idx=self.current_window,
                )

        return self._orig(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    def __enter__(self):
        self._orig = F.softmax
        F.softmax = self._patched
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        F.softmax = self._orig
        if DEBUG_PATCH and (DEBUG_WINDOW < 0 or self.current_window == DEBUG_WINDOW):
            print(f"[PATCH_SUMMARY] method={self.method} total_calls={self.calls_total} layer_minus1={self.calls_layer_minus1}")
            for L in sorted(self.calls_per_layer.keys()):
                print(f"  layer {L}: calls={self.calls_per_layer[L]}")
            for shp, v in sorted(self.calls_per_shape.items(), key=lambda kv: kv[1], reverse=True)[:8]:
                print(f"  shape {shp}: calls={v}")


# ---------------- dataset tokens ----------------
def load_wikitext2_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if max_tokens is not None and ids.numel() > max_tokens:
        ids = ids[:max_tokens]
    return ids

def force_eager_attention(model):
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

def register_layer_hooks(model):
    """
    Sets CURRENT_LAYER["idx"] before each layer's self_attn forward, resets after.
    """
    handles = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        def make_pre(ii):
            def pre_hook(mod, inp):
                CURRENT_LAYER["idx"] = ii
            return pre_hook

        def post_hook(mod, inp, out):
            CURRENT_LAYER["idx"] = -1

        handles.append(layer.self_attn.register_forward_pre_hook(make_pre(i)))
        handles.append(layer.self_attn.register_forward_hook(post_hook))
    print(f"[info] registered layer hooks: {len(handles)}")
    return handles


# ---------------- perplexity eval ----------------
def compute_ppl_prefix_chunk(
    model,
    token_ids_1d: torch.Tensor,
    context_len: int,
    chunk_len: int,
    device: torch.device,
    patcher: Optional[AttentionSoftmaxPatcherGPU],
    max_windows: int,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
):
    prefix_len = context_len - chunk_len
    ids = token_ids_1d.to(device)

    needed = context_len + 1
    if ids.numel() < needed:
        print(f"[warn] Not enough tokens: have {ids.numel()} need {needed}")
        return 1.0, 0.0, 0, []

    stride = chunk_len
    last_start = ids.numel() - needed
    start_positions = list(range(0, last_start + 1, stride))
    if max_windows is not None:
        start_positions = start_positions[:max_windows]

    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    total_nll = 0.0
    total_tokens = 0
    rows = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if (use_autocast and device.type == "cuda") else
        torch.autocast(device_type="cpu", enabled=False)
    )

    for wi, st in enumerate(start_positions):
        window_plus1 = ids[st : st + needed]
        if window_plus1.numel() != needed:
            break

        window_in  = window_plus1[:-1]
        window_tgt = window_plus1[1:]

        prefix_ids = window_in[:prefix_len].unsqueeze(0)
        chunk_ids  = window_in[prefix_len:prefix_len + chunk_len].unsqueeze(0)
        labels     = window_tgt[prefix_len:prefix_len + chunk_len].unsqueeze(0)

        prefix_mask = torch.ones_like(prefix_ids, dtype=torch.long, device=device)
        full_mask   = torch.ones((1, context_len), dtype=torch.long, device=device)

        # prefix (no patch)
        with autocast_ctx:
            out1 = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True)
        past = out1.past_key_values

        # chunk (patched)
        if patcher is not None:
            patcher.reset()
            patcher.current_window = wi
            with patcher:
                with autocast_ctx:
                    out2 = model(input_ids=chunk_ids, attention_mask=full_mask, past_key_values=past, use_cache=True)
        else:
            with autocast_ctx:
                out2 = model(input_ids=chunk_ids, attention_mask=full_mask, past_key_values=past, use_cache=True)

        logits = out2.logits  # [1, chunk_len, vocab]
        vocab = logits.size(-1)

        loss = ce(logits.reshape(-1, vocab), labels.reshape(-1))
        n_toks = int(labels.numel())

        total_nll += float(loss.item())
        total_tokens += n_toks

        rows.append({
            "window_idx": wi,
            "start_pos": st,
            "loss_sum": float(loss.item()),
            "tokens": n_toks,
            "loss_per_token": float(loss.item()) / max(1, n_toks),
        })

    ppl = math.exp(total_nll / max(1, total_tokens))
    avg_nll = total_nll / max(1, total_tokens)
    return ppl, avg_nll, total_tokens, rows


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--encoding-json", required=True)
    ap.add_argument("--my-approx-json", type=str, default=None)
    ap.add_argument("--methods", type=str, default="fp,qcom,qcom3,my")  # fp,qcom,qcom3,my
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--chunk-len", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=200000)
    ap.add_argument("--outdir", default="./workdir/perplexity")
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--no-autocast", action="store_true")
    args = ap.parse_args()

    _load_runtime_dependencies()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.set_num_interop_threads(2)

    os.makedirs(args.outdir, exist_ok=True)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("fp","qcom","qcom3","my"):
            raise ValueError("methods must be any of: fp,qcom,qcom3,my")

    device = torch.device("cuda" if args.device == "cuda" else "cpu")

    if args.dtype == "fp16":
        model_dtype = torch.float16
    elif args.dtype == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    enc = _load_encodings(args.encoding_json)

    # Preload tables on device (int64)
    tables = {
        "X_MID_Q30": torch.tensor(X_MID_Q30, dtype=torch.int64, device=device),

        "C0_D4": torch.tensor(C0_D4, dtype=torch.int64, device=device),
        "C1_D4": torch.tensor(C1_D4, dtype=torch.int64, device=device),
        "C2_D4": torch.tensor(C2_D4, dtype=torch.int64, device=device),
        "C3_D4": torch.tensor(C3_D4, dtype=torch.int64, device=device),
        "C4_D4": torch.tensor(C4_D4, dtype=torch.int64, device=device),

        "C0_D3": torch.tensor(C0_D3, dtype=torch.int64, device=device),
        "C1_D3": torch.tensor(C1_D3, dtype=torch.int64, device=device),
        "C2_D3": torch.tensor(C2_D3, dtype=torch.int64, device=device),
        "C3_D3": torch.tensor(C3_D3, dtype=torch.int64, device=device),
    }

    my_bank = None
    if "my" in methods:
        if args.my_approx_json is None:
            raise ValueError("--my-approx-json required when using my")
        my_bank = load_my_bank_coeffs_tree(args.my_approx_json, device=device)
        validate_my_bank(my_bank, name=os.path.basename(args.my_approx_json))
        print(f"[info] loaded my_bank entries: {len(my_bank)}")

    print("[info] loading model:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    force_eager_attention(model)
    _ = register_layer_hooks(model)

    token_ids = load_wikitext2_tokens(tok, max_tokens=args.max_tokens)
    print(f"[info] token stream length: {token_ids.numel()}")

    use_autocast = (not args.no_autocast) and (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))

    results = []
    for method in methods:
        if method == "fp":
            patcher = None
        else:
            patcher = AttentionSoftmaxPatcherGPU(
                enc=enc,
                method=method,
                my_bank=my_bank if method == "my" else None,
                context_len=args.context_len,
                chunk_len=args.chunk_len,
                tables=tables,
            )

        with torch.no_grad():
            ppl, avg_nll, ntok, rows = compute_ppl_prefix_chunk(
                model=model,
                token_ids_1d=token_ids,
                context_len=args.context_len,
                chunk_len=args.chunk_len,
                device=device,
                patcher=patcher,
                max_windows=args.max_windows,
                use_autocast=use_autocast,
                autocast_dtype=model_dtype if device.type == "cuda" else torch.float32,
            )

        print(f"[PPL:{method}] ppl={ppl:.6f}  avg_nll={avg_nll:.8f}  tokens={ntok}")
        results.append({"method": method, "ppl": ppl, "avg_nll": avg_nll, "tokens": ntok})

        if args.save_csv:
            out_csv = os.path.join(args.outdir, f"ppl_windows_{method}.csv")
            with open(out_csv, "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                else:
                    w = csv.writer(f)
                    w.writerow(["window_idx","start_pos","loss_sum","tokens","loss_per_token"])
            print("[write]", out_csv)

    out_sum = os.path.join(args.outdir, "ppl_summary.csv")
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","ppl","avg_nll","tokens"])
        w.writeheader()
        w.writerows(results)
    print("[write]", out_sum)


if __name__ == "__main__":
    main()
