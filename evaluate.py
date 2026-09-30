import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EPS = 1e-8
ACTIVE_THRESHOLD = 1e-4
DCOORD_NUM_SP = 5
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
SC_TYPES = ["entity", "attribute", "relation", "negation", "quantity", "temporal", "intent"]
SP_STYLES = ["formal", "casual", "question", "technical",
             "simplified", "verbose", "passive", "imperative"]


def topk_sparsify(scores, k):
    idx = scores.abs().topk(k, dim=-1).indices
    return scores * torch.zeros_like(scores).scatter_(-1, idx, 1.0)


class TopKEncoder(nn.Module):
    def __init__(self, dim, num_atoms, k):
        super().__init__()
        self.proj = nn.Linear(dim, num_atoms)
        self.k = k

    def forward(self, dz):
        return topk_sparsify(self.proj(dz), self.k)


class GlobalFrame(nn.Module):
    def __init__(self, dim, num_atoms, k):
        super().__init__()
        self.A0 = nn.Parameter(torch.zeros(dim, num_atoms))
        self.encoder = TopKEncoder(dim, num_atoms, k)

    def frame(self, z_anchor):
        return self.A0


class LocalFrame(nn.Module):
    def __init__(self, dim, num_atoms, k, hidden=512):
        super().__init__()
        self.A0 = nn.Parameter(torch.zeros(dim, num_atoms))
        self.atom_mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, num_atoms),
        )
        self.encoder = TopKEncoder(dim, num_atoms, k)

    def frame(self, z_anchor):
        r = torch.tanh(self.atom_mlp(z_anchor.unsqueeze(0))).squeeze(0)
        return self.A0 * (1 + r)


def load_frame(path, k, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    dim, num_atoms = state["A0"].shape
    if any(key.startswith("atom_mlp.") for key in state):
        frame, kind = LocalFrame(dim, num_atoms, k, state["atom_mlp.0.weight"].shape[0]), "Local"
    else:
        frame, kind = GlobalFrame(dim, num_atoms, k), "Global"
    frame.load_state_dict(state, strict=True)
    return frame.to(device).eval(), kind


def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    texts, layout = [], []
    for g in data["groups"]:
        layout.append({
            "start": len(texts), "n_sp": len(g["sp"]), "n_sc": len(g["sc"]),
            "sp_styles": [m["style"] for m in g.get("sp_metadata", [])],
            "sc_types": [m["edit_type"] for m in g.get("sc_metadata", [])],
        })
        texts += [g["anchor"]] + g["sp"] + g["sc"]
    return texts, layout, data.get("split") == "eval"


@torch.no_grad()
def extract_hidden_states(model_name, layer, texts, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPES[args.dtype], output_hidden_states=True,
        trust_remote_code=True,
    ).to(args.device).eval()

    if args.hidden_dim and model.config.hidden_size != args.hidden_dim:
        raise SystemExit(f"{model_name} hidden size is {model.config.hidden_size}, "
                         f"not {args.hidden_dim}")
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers and not 1 <= layer <= n_layers:
        raise SystemExit(f"{model_name} has {n_layers} layers; got --layer {layer}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    chunks = []
    for i in range(0, len(texts), args.batch_size):
        batch = tokenizer(texts[i:i + args.batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to(args.device)
        chunks.append(model(**batch).hidden_states[layer].float()[:, -1, :].cpu())

    features = torch.cat(chunks)
    if not torch.isfinite(features).all():
        raise SystemExit("non-finite hidden states; try --dtype bfloat16")
    return features


def build_groups(features, layout, held_out):
    groups = []
    for g in layout:
        s, n_sp, n_sc = g["start"], g["n_sp"], g["n_sc"]
        z_sp = features[s + 1:s + 1 + n_sp]
        z_sc = features[s + 1 + n_sp:s + 1 + n_sp + n_sc]
        sp_styles, sc_types = g["sp_styles"], g["sc_types"]
        if held_out:
            z_sp, z_sc = z_sp[1::2], z_sc[1::2]
            sp_styles, sc_types = sp_styles[1::2], sc_types[1::2]
        groups.append({"z_anchor": features[s], "z_sp": z_sp, "z_sc": z_sc,
                       "sp_styles": sp_styles, "sc_types": sc_types})
    return groups


def prepare_features(args):
    if args.features:
        data = torch.load(args.features, map_location="cpu", weights_only=False)
        return data["groups"], data["dim"]

    texts, layout, is_eval_split = load_dataset(args.data)
    print(f"extracting layer {args.layer} from {args.model} ({len(texts)} texts)")
    raw = extract_hidden_states(args.model, args.layer, texts, args)
    mean, std = raw.mean(dim=0), raw.std(dim=0).clamp(min=1e-6)
    groups = build_groups((raw - mean) / std, layout, held_out=not is_eval_split)

    if args.save_features:
        os.makedirs(os.path.dirname(args.save_features) or ".", exist_ok=True)
        torch.save({"groups": groups, "dim": raw.shape[1]}, args.save_features)
        print(f"saved features: {args.save_features}")
    return groups, raw.shape[1]


def label(labels, i):
    return labels[i] if i < len(labels) else "unknown"


@torch.no_grad()
def evaluate(frame, groups, device):
    sc_values = defaultdict(lambda: defaultdict(list))
    sp_energy = defaultdict(list)

    for g in groups:
        if len(g["z_sp"]) == 0 or len(g["z_sc"]) == 0:
            continue
        z = g["z_anchor"].to(device)
        z_sp, z_sc = g["z_sp"].to(device), g["z_sc"].to(device)

        A = frame.frame(z)
        dz_sem, dz_nui = z_sc - z, z_sp - z
        alpha = frame.encoder(dz_sem)
        recon = alpha @ A.T

        coords = dz_sem @ A
        sem = coords.pow(2).sum(dim=1)
        nui = (dz_nui @ A).pow(2).sum(dim=1)

        n_ref = min(DCOORD_NUM_SP, len(z_sp))
        drift = torch.stack([(coords - (z_sc - z_sp[m]) @ A).pow(2).sum(dim=1)
                             for m in range(n_ref)]).mean(dim=0)
        mag = alpha.abs()

        metrics = {
            "E_rec": (dz_sem - recon).norm(dim=1) / (dz_sem.norm(dim=1) + EPS),
            "Cos": F.cosine_similarity(dz_sem, recon, dim=1),
            "S/T": sem / (nui.mean() + EPS),
            "D_coord": drift / (sem + EPS),
            "EffNum": mag.sum(dim=1).pow(2) / (mag.pow(2).sum(dim=1) + EPS),
            "Active": (mag > ACTIVE_THRESHOLD).float().sum(dim=1),
        }
        for n in range(len(z_sc)):
            bucket = sc_values[label(g["sc_types"], n)]
            for name, values in metrics.items():
                bucket[name].append(values[n].item())
        for m in range(len(z_sp)):
            sp_energy[label(g["sp_styles"], m)].append(nui[m].item())

    def stats(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    per_type, pooled = {}, defaultdict(list)
    for sc_type, metrics in sc_values.items():
        per_type[sc_type] = {"count": len(metrics["E_rec"]),
                             **{k: stats(v) for k, v in metrics.items()}}
        for k, v in metrics.items():
            pooled[k] += v
    return {
        "overall": {"count": len(pooled["E_rec"]), **{k: stats(v) for k, v in pooled.items()}},
        "per_sc_type": per_type,
        "per_sp_style": {s: {"count": len(v), "nui_energy": stats(v)} for s, v in sp_energy.items()},
    }


def print_results(rows, breakdown):
    head = f"{'frame':<8}{'E_rec':>8}{'Cos':>8}{'S/T':>8}{'D_coord':>9}{'EffNum':>8}"
    print("\n" + head + "\n" + "-" * len(head))
    for r in rows:
        o = r["results"]["overall"]
        print(f"{r['frame']:<8}{o['E_rec']['mean']:>8.3f}{o['Cos']['mean']:>8.3f}"
              f"{o['S/T']['mean']:>8.2f}{o['D_coord']['mean']:>9.2f}{o['EffNum']['mean']:>8.1f}")
    if not breakdown:
        return
    for r in rows:
        res = r["results"]
        print(f"\n{r['frame']}: S/T by SC type")
        for t in SC_TYPES + [t for t in res["per_sc_type"] if t not in SC_TYPES]:
            if t in res["per_sc_type"]:
                v = res["per_sc_type"][t]
                print(f"  {t:<12} n={v['count']:<6} {v['S/T']['mean']:.2f}")
        print(f"{r['frame']}: nuisance energy by SP style")
        for s in SP_STYLES + [s for s in res["per_sp_style"] if s not in SP_STYLES]:
            if s in res["per_sp_style"]:
                v = res["per_sp_style"][s]
                print(f"  {s:<12} n={v['count']:<6} {v['nui_energy']['mean']:.2f}")


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Global and Local invariant-atom frames.")
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--layer", type=int, required=True, help="primary layer (index into hidden_states)")
    p.add_argument("--hidden-dim", type=int, help="expected hidden size of the model")
    p.add_argument("--global-ckpt", help="Global frame checkpoint")
    p.add_argument("--local-ckpt", help="Local frame checkpoint")
    p.add_argument("--data", default="data/eval.json", help="evaluation data")
    p.add_argument("--features", help="features saved with --save-features; skips extraction")
    p.add_argument("--save-features", help="write extracted features to this path")
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--dtype", default="float16", choices=DTYPES)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--device", default=default_device())
    p.add_argument("--breakdown", action="store_true",
                   help="print S/T per SC type and nuisance energy per SP style")
    p.add_argument("--output", help="write full results to this JSON file")
    args = p.parse_args()
    if not (args.global_ckpt or args.local_ckpt):
        p.error("give --global-ckpt and/or --local-ckpt")
    return args


def main():
    args = parse_args()
    groups, dim = prepare_features(args)

    rows = []
    for path in (args.global_ckpt, args.local_ckpt):
        if not path:
            continue
        frame, kind = load_frame(path, args.topk, args.device)
        if frame.A0.shape[0] != dim:
            raise SystemExit(f"{path}: frame dim {frame.A0.shape[0]} != feature dim {dim}")
        rows.append({"model": args.model, "layer": args.layer, "frame": kind,
                     "results": evaluate(frame, groups, args.device)})

    print_results(rows, args.breakdown)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nresults written to {args.output}")


if __name__ == "__main__":
    main()
