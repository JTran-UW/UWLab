# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline encoder benchmark on a collect_vision_probe_dataset.py dataset: regress peg / hole / peg-minus-hole positions
(wrist frame, the aux-loss targets) from side + wrist RGB + joint positions, and report test error in mm.

  preprocess:  vision_probe_benchmark.py preprocess --data <dataset dir> [--size 128]
               -> <dataset dir>/data<size>.pt (uint8 images resized with antialiasing, labels, keep mask applied)
  train:       vision_probe_benchmark.py train --data <dataset dir>/data128.pt --encoder <name> --out <dir>
               [--train_episodes_frac 1.0] [--steps 30000]

Encoders: cnn_small (RL CNN, width 0.55), cnn_wide (width 2), resnet18 (scratch), resnet18_imagenet (fine-tuned),
dino_map (frozen DINOv3 ViT-S/16 + MAP), dino_ft2 (last 2 blocks trainable + MAP), hybrid (frozen DINO + MAP and a
trainable wrist CNN), vit_tiny_p16 / vit_tiny_p8 (untrained ViT-Tiny, 16 / 8 px patches, shared by both cameras, + MAP). Splits are by episode (80/10/10, fixed seed); training uses OmniResetImageAugV2.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = {"peg": slice(0, 3), "hole": slice(18, 21), "peg_minus_hole": slice(39, 42)}
DIST_BINS = [(-1.0, 0.02, "seated (<2 cm)"), (0.02, 0.05, "2-5 cm"), (0.05, 0.15, "5-15 cm"), (0.15, 9.0, ">15 cm")]


def preprocess(args):
    shards = sorted(glob.glob(os.path.join(args.data, "shard_*.pt")))
    keep_path = os.path.join(args.data, "keep_mask.pt")
    keep = torch.load(keep_path)["keep"] if os.path.exists(keep_path) else None
    out, offset = {}, 0
    for path in shards:
        d = torch.load(path)
        n = d["side"].shape[0]
        k = keep[offset : offset + n] if keep is not None else torch.ones(n, dtype=torch.bool)
        offset += n
        for cam in ("side", "wrist"):
            if f"{cam}_mean" not in d:
                d[f"{cam}_mean"] = d[cam].float().mean((1, 2, 3))
        for cam in ("side", "wrist"):
            x = d.pop(cam)[k].permute(0, 3, 1, 2).float()
            x = torch.cat([F.interpolate(x[i : i + 1024], size=args.size, mode="bilinear", antialias=True, align_corners=False)
                           for i in range(0, x.shape[0], 1024)])
            out.setdefault(cam, []).append(x.round().clamp(0, 255).to(torch.uint8))
        for key, v in d.items():
            out.setdefault(key, []).append(v[k])
        print(f"[prep] {path}: kept {int(k.sum())}/{n}", flush=True)
    out = {k: torch.cat(v) for k, v in out.items()}
    path = os.path.join(args.data, f"data{args.size}.pt")
    torch.save(out, path)
    print(f"[prep] wrote {path}: {out['side'].shape[0]} frames, images {tuple(out['side'].shape[1:])}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
class ConvTrunk(nn.Module):
    """The RL CNN per camera (rgb.PixelStateEncoder conv stack), flattened."""

    def __init__(self, size: int, width: float):
        super().__init__()
        layers, cin = [], 3
        for i, c in enumerate(max(1, round(c * width)) for c in (16, 32, 64, 64)):
            layers += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(max(size // 32, 1) if i == 0 else 2)]
            cin = c
        self.net = nn.Sequential(*layers, nn.Conv2d(cin, max(1, round(64 * width)), 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 3, size, size)).shape[-1]

    def forward(self, x):
        return self.net(x)


class ResNetTrunk(nn.Module):
    def __init__(self, pretrained_path: str | None):
        super().__init__()
        from torchvision.models import resnet18

        net = resnet18(weights=None)
        if pretrained_path:
            net.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        net.fc = nn.Identity()
        self.net, self.out_dim, self.pretrained = net, 512, bool(pretrained_path)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class DinoTokens(nn.Module):
    """DINOv3 ViT-S/16 patch tokens (B, T, 384); ``trainable_blocks`` last blocks (+ final norm) get gradients."""

    def __init__(self, model_dir: str, trainable_blocks: int):
        super().__init__()
        timm_dir = os.environ.get("UWLAB_TIMM_DIR", "/workspace/uwlab/_timm")
        if os.path.isdir(timm_dir) and timm_dir not in sys.path:
            sys.path.insert(0, timm_dir)
        import timm

        arch = os.path.basename(model_dir.rstrip("/")).split(".")[0]
        self.model = timm.create_model(arch, pretrained=True, num_classes=0,
                                       pretrained_cfg_overlay=dict(file=os.path.join(model_dir, "model.safetensors")))
        self.model.requires_grad_(False)
        if trainable_blocks > 0:
            for blk in self.model.blocks[-trainable_blocks:]:
                blk.requires_grad_(True)
            self.model.norm.requires_grad_(True)
        self.frozen = trainable_blocks == 0
        cfg = self.model.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            h = self.model.forward_features((x - self.mean) / self.std)
        return h[:, self.model.num_prefix_tokens :]


class VitScratchTokens(nn.Module):
    """Untrained timm ViT-Tiny (12 blocks, width 192) with ``patch``-px patches; returns patch tokens (B, T, 192)."""

    def __init__(self, size: int, patch: int):
        super().__init__()
        timm_dir = os.environ.get("UWLAB_TIMM_DIR", "/workspace/uwlab/_timm")
        if os.path.isdir(timm_dir) and timm_dir not in sys.path:
            sys.path.insert(0, timm_dir)
        import timm

        self.model = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=size, patch_size=patch, num_classes=0)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model.forward_features((x - self.mean) / self.std)[:, self.model.num_prefix_tokens :]


class ProbeModel(nn.Module):
    """Per-camera image features + projected joints -> MLP -> normalized targets."""

    def __init__(self, encoder: str, size: int, state_dim: int, out_dim: int, dino_dir: str, resnet_weights: str):
        super().__init__()
        from uwlab_rl.flashsac.rgb import MAPHead

        self.encoder, self.cnn_size = encoder, 96
        self.state_fc = nn.Linear(state_dim, 64)
        if encoder in ("cnn_small", "cnn_wide"):
            w = 0.55 if encoder == "cnn_small" else 2.0
            self.trunks = nn.ModuleList(ConvTrunk(self.cnn_size, w) for _ in range(2))
            feat = sum(t.out_dim for t in self.trunks)
        elif encoder in ("resnet18", "resnet18_imagenet"):
            self.trunks = nn.ModuleList(ResNetTrunk(resnet_weights if encoder.endswith("imagenet") else None) for _ in range(2))
            feat = 1024
        elif encoder.startswith("vit_tiny"):
            patch = int(encoder.split("_p")[-1])
            self.vit = VitScratchTokens(size, patch)
            self.maps = nn.ModuleList(MAPHead((size // patch) ** 2, 192, 128, 8, 4) for _ in range(2))
            feat = 2 * 8 * 128
        else:
            self.dino = DinoTokens(dino_dir, 2 if encoder == "dino_ft2" else 0)
            t = (size // 16) ** 2
            self.maps = nn.ModuleList(MAPHead(t, 384, 128, 8, 4) for _ in range(2))
            feat = 2 * 8 * 128
            if encoder == "hybrid":
                self.wrist_cnn = ConvTrunk(self.cnn_size, 0.55)
                feat += self.wrist_cnn.out_dim
        self.img_fc = nn.Linear(feat, 256)
        self.head = nn.Sequential(nn.LayerNorm(256 + 64), nn.Linear(256 + 64, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, out_dim))

    def param_groups(self, lr):
        slow = [p for n, p in self.named_parameters() if p.requires_grad and (n.startswith("dino.") or
                (self.encoder == "resnet18_imagenet" and n.startswith("trunks.")))]
        ids = {id(p) for p in slow}
        fast = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return [{"params": fast, "lr": lr}] + ([{"params": slow, "lr": lr * 0.1}] if slow else [])

    def forward(self, side, wrist, state):
        """side / wrist: (B, 3, H, W) in 0..1 at the stored size."""
        small = lambda x: F.interpolate(x, size=self.cnn_size, mode="bilinear", antialias=True, align_corners=False)
        if self.encoder.startswith("cnn"):
            f = torch.cat([t(small(x)) for t, x in zip(self.trunks, (side, wrist))], -1)
        elif self.encoder.startswith("resnet"):
            f = torch.cat([t(x) for t, x in zip(self.trunks, (side, wrist))], -1)
        elif self.encoder.startswith("vit_tiny"):
            tok = self.vit(torch.cat([side, wrist])).chunk(2)
            f = torch.cat([m(t.float()) for m, t in zip(self.maps, tok)], -1)
        else:
            tok = self.dino(torch.cat([side, wrist])).chunk(2)
            f = torch.cat([m(t.float()) for m, t in zip(self.maps, tok)], -1)
            if self.encoder == "hybrid":
                f = torch.cat([f, self.wrist_cnn(small(wrist))], -1)
        return self.head(torch.cat([self.img_fc(f), self.state_fc(state)], -1))


# ---------------------------------------------------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(args.data)
    targets = torch.cat([d["aux_state"][:, s] for s in TARGETS.values()], 1)
    state = d["joint_pos"].float()
    ep = (d["env_id"] * 100000 + d["episode"]).numpy()
    uniq = np.unique(ep)
    perm = np.random.default_rng(0).permutation(uniq)
    n_tr, n_va = int(0.8 * len(uniq)), int(0.1 * len(uniq))
    tr_eps = perm[:n_tr][: max(1, int(round(args.train_episodes_frac * n_tr)))]
    split = {"train": np.isin(ep, tr_eps), "val": np.isin(ep, perm[n_tr : n_tr + n_va]), "test": np.isin(ep, perm[n_tr + n_va :])}
    idx = {k: torch.from_numpy(np.flatnonzero(v)) for k, v in split.items()}
    mu, sd = targets[idx["train"]].mean(0), targets[idx["train"]].std(0).clamp_min(1e-4)
    smu, ssd = state[idx["train"]].mean(0), state[idx["train"]].std(0).clamp_min(1e-4)
    print(f"[bench] {args.encoder}: frames train {len(idx['train'])} ({len(tr_eps)} episodes) val {len(idx['val'])} test {len(idx['test'])}", flush=True)
    side, wrist = d["side"].to(dev), d["wrist"].to(dev)  # uint8 on the GPU
    targets_n, state_n = ((targets - mu) / sd).to(dev), ((state - smu) / ssd).to(dev)
    size = side.shape[-1]
    model = ProbeModel(args.encoder, size, state.shape[1], targets.shape[1], args.dino_dir, args.resnet_weights).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.param_groups(args.lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))))
    from uwlab_rl.flashsac.rgb import OmniResetImageAugV2

    aug = OmniResetImageAugV2(blur_sigma=(0.1, 0.86))
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    def batch(ids, augment):
        s, w = side[ids].float(), wrist[ids].float()
        if augment:
            s, w = aug(s).clamp(0, 255), aug(w).clamp(0, 255)
        return s / 255.0, w / 255.0, state_n[ids], targets_n[ids]

    @torch.no_grad()
    def evaluate(which):
        model.eval()
        preds = []
        ids_all = idx[which].to(dev)
        for i in range(0, len(ids_all), 1024):
            s, w, st, _ = batch(ids_all[i : i + 1024], False)
            with torch.autocast(**amp):
                preds.append(model(s, w, st).float())
        model.train()
        p = torch.cat(preds) * sd.to(dev) + mu.to(dev)
        return p.cpu(), targets[idx[which]]

    def errors(p, t, sub):
        """mm errors per target (3-D norm) on rows ``sub``."""
        out = {}
        for k, (name, s) in enumerate(TARGETS.items()):
            e = (p[sub, 3 * k : 3 * k + 3] - t[sub, 3 * k : 3 * k + 3]).norm(dim=1) * 1000
            out[name] = {"median_mm": round(float(e.median()), 2), "p90_mm": round(float(torch.quantile(e, 0.9)), 2)} if len(e) else None
        return out

    # cost: forward + backward at batch 2048 (an RL update's obs + next obs of 1024)
    cost_ms = None
    if dev == "cuda":
        try:
            ids = idx["train"][torch.randint(len(idx["train"]), (min(2048, len(idx["train"])),))].to(dev)
            s, w, st, y = batch(ids, False)
            for k in range(8):
                if k == 3:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast(**amp):
                    F.mse_loss(model(s, w, st).float(), y).backward()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            cost_ms = round((time.perf_counter() - t0) / 5 * 1000, 1)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(f"[bench] fwd+bwd at batch 2048: {cost_ms} ms; trainable params {n_params / 1e6:.2f}M", flush=True)

    best, best_state, curve, t_start = float("inf"), None, [], time.time()
    tr = idx["train"].to(dev)
    for step in range(1, args.steps + 1):
        ids = tr[torch.randint(len(tr), (args.batch,), device=dev)]
        s, w, st, y = batch(ids, True)
        with torch.autocast(**amp):
            loss = F.mse_loss(model(s, w, st).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            p, t = evaluate("val")
            e = errors(p, t, torch.arange(len(t)))
            v = e["peg_minus_hole"]["median_mm"]
            curve.append({"step": step, "train_loss": round(float(loss), 4), "val_peg_minus_hole_median_mm": v, "minutes": round((time.time() - t_start) / 60, 1)})
            print(f"[bench] step {step} loss {float(loss):.4f} val peg-hole median {v:.2f} mm ({curve[-1]['minutes']} min)", flush=True)
            if v < best:
                best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    p, t = evaluate("test")
    test_rows = idx["test"]
    dist = (d["peg_pose"][test_rows, :3] - d["hole_pose"][test_rows, :3]).norm(dim=1)
    dist = dist - 0.015  # peg root sits 15 mm above the hole root when seated
    res = {"encoder": args.encoder, "train_episodes_frac": args.train_episodes_frac, "train_frames": len(idx["train"]),
           "steps": args.steps, "trainable_params_M": round(n_params / 1e6, 3), "fwd_bwd_ms_batch2048": cost_ms,
           "best_val_peg_minus_hole_median_mm": best, "test_all": errors(p, t, torch.arange(len(t))), "test_by_distance": {},
           "test_by_appearance": {}, "curve": curve, "minutes": round((time.time() - t_start) / 60, 1)}
    for lo, hi, name in DIST_BINS:
        sub = torch.nonzero((dist >= lo) & (dist < hi)).flatten()
        res["test_by_distance"][name] = {"frames": len(sub), **(errors(p, t, sub) if len(sub) else {})}
    peg_app = d["appearance_insertive_object"][test_rows]
    wrist_dark = d["wrist_mean"][test_rows] < 60
    for name, m in {"peg textured": peg_app[:, 3] >= 0, "peg plain": peg_app[:, 3] < 0, "peg metallic>0.7": peg_app[:, 5] > 0.7,
                    "dark wrist (mean<60)": wrist_dark}.items():
        sub = torch.nonzero(m).flatten()
        res["test_by_appearance"][name] = {"frames": len(sub), **(errors(p, t, sub) if len(sub) else {})}
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.encoder}_frac{args.train_episodes_frac:g}"
    np.savez(os.path.join(args.out, f"{tag}_test.npz"), pred=p.numpy(), target=t.numpy(), dist=dist.numpy(),
             rows=test_rows.numpy(), target_names=np.array(list(TARGETS)))
    json.dump(res, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=1)
    print(f"[bench] RESULT {tag}: test peg-hole median {res['test_all']['peg_minus_hole']['median_mm']} mm, "
          f"2-5 cm bin {res['test_by_distance']['2-5 cm'].get('peg_minus_hole')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preprocess")
    p.add_argument("--data", required=True)
    p.add_argument("--size", type=int, default=128)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--encoder", required=True, choices=["cnn_small", "cnn_wide", "resnet18", "resnet18_imagenet", "dino_map", "dino_ft2", "hybrid", "vit_tiny_p16", "vit_tiny_p8"])
    t.add_argument("--out", required=True)
    t.add_argument("--train_episodes_frac", type=float, default=1.0)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--eval_every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dino_dir", default="/workspace/uwlab/checkpoints/dinov3/vit_small_patch16_dinov3.lvd1689m")
    t.add_argument("--resnet_weights", default="/workspace/uwlab/checkpoints/resnet18-f37072fd.pth")
    a = ap.parse_args()
    preprocess(a) if a.cmd == "preprocess" else train(a)
