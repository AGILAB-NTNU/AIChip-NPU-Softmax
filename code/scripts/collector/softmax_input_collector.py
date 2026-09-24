#!/usr/bin/env python3
from __future__ import annotations

"""
Collect stabilized attention-softmax inputs and optionally export u16 tensors for
the downstream PSO workflow.
"""

import os, json, argparse
import numpy as np

torch = None
F = None
AutoTokenizer = None
AutoModelForCausalLM = None
plt = None


def _load_runtime_dependencies():
    global torch, F, AutoTokenizer, AutoModelForCausalLM, plt

    try:
        import torch as torch_mod
        import torch.nn.functional as F_mod
        from transformers import AutoModelForCausalLM as auto_model_for_causal_lm_mod
        from transformers import AutoTokenizer as auto_tokenizer_mod
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing collector dependencies. Install runtime dependencies with "
            "'python3 -m pip install -r code/requirements.txt'."
        ) from exc

    torch = torch_mod
    F = F_mod
    AutoTokenizer = auto_tokenizer_mod
    AutoModelForCausalLM = auto_model_for_causal_lm_mod
    plt = plt_mod


# ---------------- Encodings helpers ----------------
def load_encodings_activation(path: str) -> dict:
    with open(path, "r") as f:
        enc = json.load(f)
    if "activation_encodings" in enc:
        return enc["activation_encodings"]
    return enc

def get_layer_add2_encoding(enc_act: dict, layer_idx: int) -> dict:
    key = f"/model_layers_{layer_idx}_self_attn_Add_2/Add_output_0"
    if key not in enc_act:
        raise KeyError(f"Missing Add_2 encoding key: {key}")
    e0 = enc_act[key][0]
    return {"scale": float(e0["scale"]), "offset": int(e0["offset"])}


# ---------------- Quant helpers ----------------
def quant_to_u16_torch(x_real: torch.Tensor, scale: float, offset: int) -> torch.Tensor:
    # q = round(x/scale - offset), clamp to [0,65535]
    q = torch.round(x_real / float(scale) - float(offset)).to(torch.int64)
    q = torch.clamp(q, 0, 65535).to(torch.uint16)
    return q


# ---------------- Parsing helpers ----------------
def parse_int_list(s: str):
    # e.g. "0,1,2,10" or "0-3,8"
    s = s.strip()
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))

def parse_int_set(s: str):
    if not s:
        return None
    lst = parse_int_list(s)
    return set(lst) if lst is not None else None


# ---------------- Token windowing (NO repeating by default) ----------------
def build_ctx_from_ids(
    ids: list,
    total_len: int,
    pad_id: int,
    mode: str,
    truncate_side: str,
    pad_side: str,
    rng: np.random.Generator,
):
    """
    ids: token ids for one prompt
    Returns (full_ids, full_mask) length total_len.
      - mode="truncate": if ids longer, take head/tail; if shorter, pad
      - mode="pad": if ids longer, take head/tail; if shorter, pad
      - mode="random_window": if ids longer, choose random contiguous window of total_len
                              if shorter, pad
      - mode="repeat": old behavior (discouraged), repeats ids to fill
    """
    assert mode in ["truncate", "pad", "random_window", "repeat"]
    assert truncate_side in ["head", "tail"]
    assert pad_side in ["left", "right"]

    if len(ids) == 0:
        ids = [pad_id]

    if mode == "repeat":
        # old behavior (kept only if you explicitly ask)
        if len(ids) >= total_len:
            ids2 = ids[:total_len] if truncate_side == "head" else ids[-total_len:]
        else:
            reps = (total_len + len(ids) - 1) // len(ids)
            ids2 = (ids * reps)[:total_len]
        mask = [1] * total_len
        return ids2, mask

    # from here: no repeating
    if len(ids) >= total_len:
        if mode == "random_window":
            start = int(rng.integers(0, len(ids) - total_len + 1))
            ids2 = ids[start:start + total_len]
        else:
            ids2 = ids[:total_len] if truncate_side == "head" else ids[-total_len:]
        mask = [1] * total_len
        return ids2, mask

    # shorter -> pad
    n_pad = total_len - len(ids)
    if pad_side == "left":
        ids2 = [pad_id] * n_pad + ids
        mask = [0] * n_pad + [1] * len(ids)
    else:
        ids2 = ids + [pad_id] * n_pad
        mask = [1] * len(ids) + [0] * n_pad
    return ids2, mask


def force_eager_attention(model):
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"


# ---------------- Distribution accumulator ----------------
class LayerHeadDistributions:
    """
    For one layer: stores hist counts per head and samples per head for boxplot.
    z is in real domain, clipped to [clamp_min, 0].
    """
    def __init__(self, n_heads: int, bins: int, clamp_min: float, sample_per_head: int, seed: int = 0):
        self.n_heads = n_heads
        self.bins = bins
        self.clamp_min = float(clamp_min)
        self.sample_per_head = int(sample_per_head)
        self.edges = np.linspace(self.clamp_min, 0.0, self.bins + 1, dtype=np.float32)
        self.hist = np.zeros((n_heads, bins), dtype=np.int64)
        self.samples = [[] for _ in range(n_heads)]
        self.rng = np.random.default_rng(seed)

        # extra: count how much got clipped at clamp_min
        self.clip_count = np.zeros((n_heads,), dtype=np.int64)
        self.total_count = np.zeros((n_heads,), dtype=np.int64)

    def update_head(self, head_idx: int, z_flat: np.ndarray, clipped_mask: np.ndarray = None):
        # histogram
        h, _ = np.histogram(z_flat, bins=self.edges)
        self.hist[head_idx] += h

        # clipping stats
        self.total_count[head_idx] += z_flat.size
        if clipped_mask is not None:
            self.clip_count[head_idx] += int(clipped_mask.sum())

        # sampling for boxplot (cap size)
        if self.sample_per_head <= 0:
            return
        n = z_flat.size
        if n <= self.sample_per_head:
            take = z_flat
        else:
            idx = self.rng.choice(n, size=self.sample_per_head, replace=False)
            take = z_flat[idx]

        take = np.atleast_1d(take).astype(np.float32)
        self.samples[head_idx].append(take)

    def finalize_samples(self):
        out = []
        for h in range(self.n_heads):
            if isinstance(self.samples[h], np.ndarray):
                out.append(self.samples[h])
                continue
            if len(self.samples[h]) == 0:
                out.append(np.empty((0,), dtype=np.float32))
            else:
                arrs = [np.atleast_1d(a) for a in self.samples[h]]
                out.append(np.concatenate(arrs, axis=0).astype(np.float32))
        self.samples = out

    def plot_hist_cdf(self, out_png: str, title: str):
        self.finalize_samples()
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.22)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])

        for h in range(self.n_heads):
            counts = self.hist[h].astype(np.float64)
            s = counts.sum()
            if s <= 0:
                continue
            pdf = counts / s
            ax1.plot(centers, pdf, linewidth=1, alpha=0.9, label=f"h{h}")
            cdf = np.cumsum(pdf)
            ax2.plot(centers, cdf, linewidth=1, alpha=0.9, label=f"h{h}")

        ax1.set_title(title + " | Histogram per head (z in [-20,0])")
        ax1.set_xlim(self.clamp_min, 0.0)
        ax1.set_ylabel("Normalized count")
        ax1.grid(True, linewidth=0.3, alpha=0.5)

        ax2.set_title(title + " | CDF per head (z in [-20,0])")
        ax2.set_xlim(self.clamp_min, 0.0)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel("z = (q - qmax) * scale")
        ax2.set_ylabel("CDF")
        ax2.grid(True, linewidth=0.3, alpha=0.5)

        ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1, frameon=False)

        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def plot_boxplot(self, out_png: str, title: str):
        self.finalize_samples()
        data = [self.samples[h] for h in range(self.n_heads)]
        fig = plt.figure(figsize=(16, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(data, showfliers=False)
        ax.set_title(title + " | Boxplot per head (sampled z in [-20,0])")
        ax.set_xlabel("Head index")
        ax.set_ylabel("z")
        ax.set_ylim(self.clamp_min, 0.0)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def write_clip_stats(self, out_json: str):
        clip_frac = {}
        for h in range(self.n_heads):
            tot = int(self.total_count[h])
            clp = int(self.clip_count[h])
            clip_frac[f"h{h}"] = {
                "total": tot,
                "clipped": clp,
                "clipped_frac": (clp / tot) if tot > 0 else 0.0
            }
        with open(out_json, "w") as f:
            json.dump(clip_frac, f, indent=2)


# ---------------- Softmax catcher ----------------
class SoftmaxInputCollectorHook:
    """
    Intercepts torch.nn.functional.softmax calls.
    Captures attention softmax inputs [1,H,Q,K] for Q=chunk_len, K=context_len.

    Features:
      - external aggregation dict (layer_dists)
      - optional per-sample u16 saving into outdir/bins/sampleXXXX/...
      - selective saving by sample/layer/head
    """
    def __init__(
        self,
        target_q: int,
        target_k: int,
        max_layers: int,
        enc_act: dict,
        layer_dists: dict,
        outdir: str,
        sample_idx: int,
        clamp_min: float = -20.0,
        bins: int = 240,
        sample_per_head: int = 50000,
        save_u16: bool = True,
        save_samples: set = None,   # which sample indices to save
        save_layers: set = None,    # which layers to save (None = all)
        save_heads: set = None,     # which heads to save  (None = all)
        debug: bool = False,
    ):
        self.target_q = int(target_q)
        self.target_k = int(target_k)
        self.max_layers = int(max_layers)
        self.enc_act = enc_act
        self.layer_dists = layer_dists
        self.outdir = outdir
        self.sample_idx = int(sample_idx)

        self.clamp_min = float(clamp_min)
        self.bins = int(bins)
        self.sample_per_head = int(sample_per_head)

        self.save_u16 = bool(save_u16)
        self.save_samples = save_samples
        self.save_layers = save_layers
        self.save_heads = save_heads
        self.debug = bool(debug)

        self._orig = None
        self.layer_idx = 0
        self.n_heads = None

        # u16 output root
        self.u16_root = os.path.join(outdir, "bins")
        if self.save_u16:
            os.makedirs(self.u16_root, exist_ok=True)

    def _should_save_this(self, layer_idx: int, head_idx: int) -> bool:
        if not self.save_u16:
            return False
        if self.save_samples is not None and self.sample_idx not in self.save_samples:
            return False
        if self.save_layers is not None and layer_idx not in self.save_layers:
            return False
        if self.save_heads is not None and head_idx not in self.save_heads:
            return False
        return True

    def _patched(self, x, dim=None, _stacklevel=3, dtype=None):
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            B, H, Q, K = x.shape
            if self.debug:
                print(f"[debug] saw softmax input shape={tuple(x.shape)} dim={dim} dtype={x.dtype}")

            if (B == 1) and (Q == self.target_q) and (K == self.target_k) and (self.layer_idx < self.max_layers):
                if self.n_heads is None:
                    self.n_heads = H

                e_add2 = get_layer_add2_encoding(self.enc_act, self.layer_idx)
                scale = e_add2["scale"]
                offset = e_add2["offset"]

                x_cpu = x.detach().to(torch.float32).cpu()
                q_u16 = quant_to_u16_torch(x_cpu, scale=scale, offset=offset).numpy()  # uint16

                if self.layer_idx not in self.layer_dists:
                    self.layer_dists[self.layer_idx] = LayerHeadDistributions(
                        n_heads=H, bins=self.bins, clamp_min=self.clamp_min,
                        sample_per_head=self.sample_per_head,
                        seed=1234 + self.layer_idx
                    )

                # Save dir per sample
                if self.save_u16 and (self.save_samples is None or self.sample_idx in self.save_samples):
                    sample_dir = os.path.join(self.u16_root, f"sample{self.sample_idx:04d}")
                    os.makedirs(sample_dir, exist_ok=True)

                for h in range(H):
                    qh = q_u16[0, h, :, :]  # [Q,K] uint16

                    if self._should_save_this(self.layer_idx, h):
                        sample_dir = os.path.join(self.u16_root, f"sample{self.sample_idx:04d}")
                        out_bin = os.path.join(sample_dir, f"layer{self.layer_idx:02d}_head{h:02d}_add2_u16.bin")
                        qh.tofile(out_bin)

                    # dq = q - qmax per row
                    q_int = qh.astype(np.int32)
                    qmax = np.max(q_int, axis=-1, keepdims=True)
                    dq = q_int - qmax  # <= 0

                    dq_min = int(np.ceil(self.clamp_min / float(scale)))  # negative
                    clipped = (dq < dq_min)
                    dq = np.maximum(dq, dq_min)

                    z = dq.astype(np.float32) * float(scale)  # in [clamp_min, 0]
                    z_flat = z.reshape(-1)
                    clipped_flat = clipped.reshape(-1)

                    self.layer_dists[self.layer_idx].update_head(h, z_flat, clipped_mask=clipped_flat)

                if self.debug:
                    print(f"[capture] sample {self.sample_idx} layer {self.layer_idx} scale={scale} offset={offset}")

                self.layer_idx += 1

        return self._orig(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    def __enter__(self):
        self._orig = F.softmax
        F.softmax = self._patched
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        F.softmax = self._orig


# ---------------- Prompt sources ----------------
def load_prompts(args):
    prompts = []

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts.append(f.read())

    if args.prompt_list:
        # one prompt per line
        with open(args.prompt_list, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    prompts.append(t)

    if args.prompt:
        prompts.append(args.prompt)

    if not prompts:
        prompts = ["Hello! Please summarize the following text and explain key points."]

    # optionally limit
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]

    return prompts


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--encoding-json", required=True)
    ap.add_argument("--outdir", default="./workdir/collector")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    ap.add_argument("--context-len", type=int, default=2048)
    ap.add_argument("--chunk-len", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=32)

    ap.add_argument("--clamp-min", type=float, default=-20.0)
    ap.add_argument("--bins", type=int, default=240)
    ap.add_argument("--sample-per-head", type=int, default=50000)

    # prompt sources
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt-file", type=str, default=None)
    ap.add_argument("--prompt-list", type=str, default=None)  # one prompt per line
    ap.add_argument("--max-prompts", type=int, default=0, help="If >0, limit number of prompts loaded from sources.")

    # windowing / fill behavior
    ap.add_argument("--mode", type=str, default="pad",
                    choices=["pad", "truncate", "random_window", "repeat"],
                    help="How to form exactly context-len tokens from each prompt.")
    ap.add_argument("--truncate-side", type=str, default="tail", choices=["head", "tail"])
    ap.add_argument("--pad-side", type=str, default="left", choices=["left", "right"])
    ap.add_argument("--windows-per-prompt", type=int, default=1,
                    help="For long texts, how many windows to sample per prompt (mode=random_window).")
    ap.add_argument("--seed", type=int, default=1234)

    # u16 saving controls
    ap.add_argument("--save-u16", action="store_true", help="Enable saving u16 bin files.")
    ap.add_argument("--save-samples", type=str, default=None,
                    help="Which sample indices to save, e.g. '0,1,5' or '0-9'. If omitted, saves all (NOT recommended).")
    ap.add_argument("--save-layers", type=str, default=None,
                    help="Which layers to save u16 for, e.g. '21' or '0-3,10'. If omitted, saves all layers.")
    ap.add_argument("--save-heads", type=str, default=None,
                    help="Which heads to save u16 for, e.g. '0-7' or '21'. If omitted, saves all heads.")
    ap.add_argument("--no-plots", action="store_true", help="Skip plotting (faster).")
    ap.add_argument("--layers", type=str, default=None,
                    help="Which layers to plot, e.g. '0-3,10,21'. If omitted, plots all captured layers.")

    ap.add_argument("--debug-softmax-shapes", action="store_true")
    args = ap.parse_args()
    _load_runtime_dependencies()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # load encodings
    enc_act = load_encodings_activation(args.encoding_json)

    # load model/tokenizer
    print("[info] loading:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

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

    # prompts
    prompts = load_prompts(args)
    print(f"[info] loaded {len(prompts)} prompt(s). mode={args.mode}, windows_per_prompt={args.windows_per_prompt}")

    # saving controls
    save_samples = parse_int_set(args.save_samples) if args.save_samples else None
    save_layers = parse_int_set(args.save_layers) if args.save_layers else None
    save_heads  = parse_int_set(args.save_heads)  if args.save_heads  else None

    # aggregation dict across all samples/windows
    layer_dists = {}

    # sample loop
    sample_idx = 0
    captured_any = False

    with torch.no_grad():
        for p_i, prompt_text in enumerate(prompts):
            ids = tok.encode(prompt_text, add_special_tokens=False)
            n_windows = args.windows_per_prompt if args.mode == "random_window" else 1
            n_windows = max(1, int(n_windows))

            for w in range(n_windows):
                full_ids, full_mask = build_ctx_from_ids(
                    ids=ids,
                    total_len=ctx,
                    pad_id=tok.pad_token_id,
                    mode=args.mode,
                    truncate_side=args.truncate_side,
                    pad_side=args.pad_side,
                    rng=rng,
                )
                input_ids = torch.tensor(full_ids, dtype=torch.long, device=args.device).unsqueeze(0)
                attn_mask = torch.tensor(full_mask, dtype=torch.long, device=args.device).unsqueeze(0)

                prefix_ids = input_ids[:, :prefix_len]
                chunk_ids  = input_ids[:, prefix_len:ctx]
                prefix_mask = attn_mask[:, :prefix_len]
                full_mask_tensor = attn_mask  # [1, ctx]

                # prefix pass builds cache
                out1 = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True)
                past = out1.past_key_values

                # capture pass on chunk
                with SoftmaxInputCollectorHook(
                    target_q=qlen, target_k=ctx, max_layers=args.max_layers,
                    enc_act=enc_act,
                    layer_dists=layer_dists,
                    outdir=args.outdir,
                    sample_idx=sample_idx,
                    clamp_min=args.clamp_min, bins=args.bins, sample_per_head=args.sample_per_head,
                    save_u16=args.save_u16,
                    save_samples=save_samples,
                    save_layers=save_layers,
                    save_heads=save_heads,
                    debug=args.debug_softmax_shapes
                ) as catcher:
                    _ = model(input_ids=chunk_ids, attention_mask=full_mask_tensor, past_key_values=past, use_cache=True)

                if catcher.layer_idx > 0:
                    captured_any = True

                if (sample_idx % 10) == 0:
                    print(f"[info] processed sample {sample_idx} (prompt {p_i}, window {w})")

                sample_idx += 1

    if not captured_any:
        raise RuntimeError("Captured 0 attention softmax calls. Try --debug-softmax-shapes.")

    captured_layers = sorted(layer_dists.keys())
    print(f"[info] captured layers: {captured_layers}")

    # plotting
    if not args.no_plots:
        wanted = parse_int_list(args.layers) if args.layers else None
        if wanted is None:
            wanted = captured_layers
        else:
            wanted = [i for i in wanted if i in layer_dists]

        plot_dir = os.path.join(args.outdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        for layer_idx in wanted:
            dist = layer_dists[layer_idx]
            title = f"Layer {layer_idx} Aggregate Distribution (Q={qlen}, K={ctx}, samples={sample_idx})"

            out_histcdf = os.path.join(plot_dir, f"layer{layer_idx:02d}_hist_cdf.png")
            out_box = os.path.join(plot_dir, f"layer{layer_idx:02d}_boxplot.png")
            out_clip = os.path.join(plot_dir, f"layer{layer_idx:02d}_clip_stats.json")

            dist.plot_hist_cdf(out_histcdf, title=title)
            dist.plot_boxplot(out_box, title=title)
            dist.write_clip_stats(out_clip)

            print(f"[write] {out_histcdf}")
            print(f"[write] {out_box}")
            print(f"[write] {out_clip}")

    # write spec
    spec = {
        "model": args.model,
        "encoding_json": args.encoding_json,
        "context_len": ctx,
        "chunk_len": qlen,
        "prefix_len": prefix_len,
        "clamp_min": args.clamp_min,
        "bins": args.bins,
        "sample_per_head": args.sample_per_head,
        "mode": args.mode,
        "truncate_side": args.truncate_side,
        "pad_side": args.pad_side,
        "windows_per_prompt": args.windows_per_prompt,
        "seed": args.seed,
        "num_samples_total": sample_idx,
        "save_u16": args.save_u16,
        "save_samples": args.save_samples,
        "save_layers": args.save_layers,
        "save_heads": args.save_heads,
        "captured_layers": captured_layers,
        "prompt_sources": {
            "prompt": bool(args.prompt),
            "prompt_file": bool(args.prompt_file),
            "prompt_list": bool(args.prompt_list),
            "num_prompts_loaded": len(prompts),
        },
    }
    with open(os.path.join(args.outdir, "collector_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[write] {os.path.join(args.outdir, 'collector_spec.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
