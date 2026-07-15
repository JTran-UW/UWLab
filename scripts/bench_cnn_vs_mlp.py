"""Synthetic microbenchmark: DepthCNN+MLP (vision student) vs MLP (state student).

Times forward + backward on random inputs matching each student's obs shape,
so we can compare per-iter network cost to the ~250 ms/iter env rollout cost
already measured at 256 envs.

No Isaac, no env — just pytorch timing. Sanity check for "is the CNN the
bottleneck during training or is it rendering/physics?".
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _conv_out(hw, k, s):
    return ((hw[0] - k) // s + 1, (hw[1] - k) // s + 1)


class DepthCNN(nn.Module):
    def __init__(self, in_channels: int, h: int, w: int, embed: int = 128):
        super().__init__()
        h1, w1 = _conv_out((h, w), 6, 2)
        h2, w2 = _conv_out((h1, w1), 4, 2)
        h3, w3 = _conv_out((h2, w2), 4, 2)
        h4, w4 = _conv_out((h3, w3), 4, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 6, 2),
            nn.ReLU(),
            nn.LayerNorm([16, h1, w1]),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.LayerNorm([32, h2, w2]),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.LayerNorm([64, h3, w3]),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.LayerNorm([128, h4, w4]),
        )
        self.flat = 128 * h4 * w4
        self.head = nn.Sequential(nn.Linear(self.flat, embed), nn.ReLU())

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class DepthCNN_NoLN_GAP(nn.Module):
    """DepthCNN variant: drop LayerNorm, GAP instead of flatten→Linear head."""

    def __init__(self, in_ch: int, embed: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 6, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
        )
        self.proj = nn.Linear(128, embed)

    def forward(self, x):
        f = self.conv(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return F.relu(self.proj(f))


class DepthCNN_Slim(nn.Module):
    """Smaller channels (8→16→32→64) + GAP. ~10× fewer params than baseline."""

    def __init__(self, in_ch: int, embed: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, 6, 2), nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        )
        self.proj = nn.Linear(64, embed)

    def forward(self, x):
        f = self.conv(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return F.relu(self.proj(f))


class DepthCNN_SlimFlat(nn.Module):
    """Baseline structure, but halve channels (8→16→32→64). Keeps flatten→Linear
    so spatial info is preserved; the Linear head shrinks from 18432→9216."""

    def __init__(self, in_ch: int, embed: int = 128):
        super().__init__()
        h, w = 224, 224
        h1, w1 = _conv_out((h, w), 6, 2)
        h2, w2 = _conv_out((h1, w1), 4, 2)
        h3, w3 = _conv_out((h2, w2), 4, 2)
        h4, w4 = _conv_out((h3, w3), 4, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, 6, 2), nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        )
        self.flat_dim = 64 * h4 * w4
        self.head = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.ReLU())

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class DepthCNN_NoLN_Flat(nn.Module):
    """Current channels (16→32→64→128), drop LayerNorm, keep flatten→Linear."""

    def __init__(self, in_ch: int, embed: int = 128):
        super().__init__()
        h, w = 224, 224
        h1, w1 = _conv_out((h, w), 6, 2)
        h2, w2 = _conv_out((h1, w1), 4, 2)
        h3, w3 = _conv_out((h2, w2), 4, 2)
        h4, w4 = _conv_out((h3, w3), 4, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 6, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
        )
        self.flat_dim = 128 * h4 * w4
        self.head = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.ReLU())

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class DepthCNN_SpatialPool(nn.Module):
    """Current channels, pool to 4x4 (keeps coarse spatial), then Linear.
    Head Linear is 128*16=2048 → 128 instead of 18432→128."""

    def __init__(self, in_ch: int, embed: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 6, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(128 * 16, embed), nn.ReLU())

    def forward(self, x):
        f = self.conv(x)
        f = F.adaptive_avg_pool2d(f, 4).flatten(1)
        return self.head(f)


class DepthCNN_AggressiveStride(nn.Module):
    """Aggressive first-conv stride-4 to shrink spatial fast + GAP."""

    def __init__(self, in_ch: int, embed: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 8, 4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
        )
        self.proj = nn.Linear(128, embed)

    def forward(self, x):
        f = self.conv(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return F.relu(self.proj(f))


def wrap_cnn_with_mlp(cnn, proprio_dim: int, num_actions: int, embed: int = 128):
    """Wrap a CNN encoder with the same MLP head used by VisionStudent."""

    class Wrapper(nn.Module):
        def __init__(self, cnn):
            super().__init__()
            self.cnn = cnn
            self.mlp = nn.Sequential(
                nn.Linear(embed + proprio_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, num_actions),
            )

        def forward(self, img, proprio):
            return self.mlp(torch.cat([self.cnn(img), proprio], dim=-1))

    return Wrapper(cnn)


class VisionStudent(nn.Module):
    """DepthCNN on image, concat with proprio history, MLP head to actions."""

    def __init__(self, proprio_dim: int, num_actions: int, in_ch: int, h: int, w: int):
        super().__init__()
        self.cnn = DepthCNN(in_ch, h, w)
        self.mlp = nn.Sequential(
            nn.Linear(128 + proprio_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, img, proprio):
        return self.mlp(torch.cat([self.cnn(img), proprio], dim=-1))


class ResNet18Student(nn.Module):
    """ImageNet-style ResNet18 encoder → 128d, concat proprio, MLP head."""

    def __init__(self, proprio_dim: int, num_actions: int, in_ch: int):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        # Replace first conv to accept arbitrary in_ch; keep the rest.
        if in_ch != 3:
            backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Drop the ImageNet classifier; the final feature is 512-d.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.proj = nn.Linear(512, 128)
        self.mlp = nn.Sequential(
            nn.Linear(128 + proprio_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, img, proprio):
        f = self.backbone(img).flatten(1)
        f = F.relu(self.proj(f))
        return self.mlp(torch.cat([f, proprio], dim=-1))


class StateStudent(nn.Module):
    """MLP on flattened 215d state history (43 per step × 5 history)."""

    def __init__(self, in_dim: int, num_actions: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.mlp(x)


def time_it(fn, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms/iter


def main():
    device = torch.device("cuda:0")
    B = 256  # match depth benchmark batch / num_envs

    # ---- Vision student ----
    vision = VisionStudent(proprio_dim=105, num_actions=7, in_ch=1, h=224, w=224).to(device)
    vis_opt = torch.optim.Adam(vision.parameters(), lr=1e-4)
    img = torch.randn(B, 1, 224, 224, device=device)
    proprio = torch.randn(B, 105, device=device)
    target = torch.randn(B, 7, device=device)

    def vis_fwd():
        vision(img, proprio)

    def vis_fwd_bwd():
        vis_opt.zero_grad()
        pred = vision(img, proprio)
        F.mse_loss(pred, target).backward()
        vis_opt.step()

    # ---- State student ----
    state = StateStudent(in_dim=215, num_actions=7).to(device)
    st_opt = torch.optim.Adam(state.parameters(), lr=1e-4)
    x = torch.randn(B, 215, device=device)

    def st_fwd():
        state(x)

    def st_fwd_bwd():
        st_opt.zero_grad()
        pred = state(x)
        F.mse_loss(pred, target).backward()
        st_opt.step()

    # Also: RGB 3-channel CNN variant.
    vision_rgb = VisionStudent(proprio_dim=105, num_actions=7, in_ch=3, h=224, w=224).to(device)
    rgb_opt = torch.optim.Adam(vision_rgb.parameters(), lr=1e-4)
    rgb = torch.randn(B, 3, 224, 224, device=device)

    def rgb_fwd():
        vision_rgb(rgb, proprio)

    def rgb_fwd_bwd():
        rgb_opt.zero_grad()
        pred = vision_rgb(rgb, proprio)
        F.mse_loss(pred, target).backward()
        rgb_opt.step()

    # ---- Run ----
    # ResNet18 variants (depth 1ch and rgb 3ch).
    rn_d = ResNet18Student(proprio_dim=105, num_actions=7, in_ch=1).to(device)
    rn_d_opt = torch.optim.Adam(rn_d.parameters(), lr=1e-4)

    def rn_d_fwd():
        rn_d(img, proprio)

    def rn_d_fwd_bwd():
        rn_d_opt.zero_grad()
        pred = rn_d(img, proprio)
        F.mse_loss(pred, target).backward()
        rn_d_opt.step()

    rn_rgb = ResNet18Student(proprio_dim=105, num_actions=7, in_ch=3).to(device)
    rn_rgb_opt = torch.optim.Adam(rn_rgb.parameters(), lr=1e-4)

    def rn_rgb_fwd():
        rn_rgb(rgb, proprio)

    def rn_rgb_fwd_bwd():
        rn_rgb_opt.zero_grad()
        pred = rn_rgb(rgb, proprio)
        F.mse_loss(pred, target).backward()
        rn_rgb_opt.step()

    # Lean variants (all 1ch depth, same proprio+MLP head for fairness).
    lean_variants = {
        "noLN+GAP (no spatial)":   DepthCNN_NoLN_GAP(1).to(device),
        "slim+GAP (no spatial)":   DepthCNN_Slim(1).to(device),
        "stride4+GAP (no spatial)": DepthCNN_AggressiveStride(1).to(device),
        "slim+flatten (spatial)":  DepthCNN_SlimFlat(1).to(device),
        "noLN+flatten (spatial)":  DepthCNN_NoLN_Flat(1).to(device),
        "pool4x4+Linear (coarse sp)": DepthCNN_SpatialPool(1).to(device),
    }
    lean_results = {}
    lean_params = {}
    for name, cnn in lean_variants.items():
        model = wrap_cnn_with_mlp(cnn, 105, 7).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        def fwd(m=model):
            m(img, proprio)

        def fwd_bwd(m=model, o=opt):
            o.zero_grad()
            p = m(img, proprio)
            F.mse_loss(p, target).backward()
            o.step()

        lean_results[name] = (time_it(fwd), time_it(fwd_bwd))
        lean_params[name] = sum(p.numel() for p in model.parameters())

    results = []
    results.append(("state MLP fwd",          time_it(st_fwd)))
    results.append(("state MLP fwd+bwd",      time_it(st_fwd_bwd)))
    results.append(("depth CNN+MLP fwd",      time_it(vis_fwd)))
    results.append(("depth CNN+MLP fwd+bwd",  time_it(vis_fwd_bwd)))
    results.append(("rgb CNN+MLP fwd",        time_it(rgb_fwd)))
    results.append(("rgb CNN+MLP fwd+bwd",    time_it(rgb_fwd_bwd)))
    results.append(("depth ResNet18 fwd",     time_it(rn_d_fwd)))
    results.append(("depth ResNet18 fwd+bwd", time_it(rn_d_fwd_bwd)))
    results.append(("rgb   ResNet18 fwd",     time_it(rn_rgb_fwd)))
    results.append(("rgb   ResNet18 fwd+bwd", time_it(rn_rgb_fwd_bwd)))

    n_state = sum(p.numel() for p in state.parameters())
    n_vision = sum(p.numel() for p in vision.parameters())
    n_rgb = sum(p.numel() for p in vision_rgb.parameters())
    n_rn_d = sum(p.numel() for p in rn_d.parameters())
    n_rn_rgb = sum(p.numel() for p in rn_rgb.parameters())

    print(f"\nBatch size B = {B}, image = 224x224\n")
    print(f"  state MLP params         = {n_state/1e6:.2f} M")
    print(f"  depth CNN+MLP params     = {n_vision/1e6:.2f} M")
    print(f"  rgb   CNN+MLP params     = {n_rgb/1e6:.2f} M")
    print(f"  depth ResNet18+MLP params = {n_rn_d/1e6:.2f} M")
    print(f"  rgb   ResNet18+MLP params = {n_rn_rgb/1e6:.2f} M\n")
    for name, ms in results:
        print(f"  {name:28s} = {ms:7.2f} ms")

    print("\n  -- Lean depth-CNN variants (224x224, 1ch, same MLP head) --")
    for name, (fwd, bwd) in lean_results.items():
        print(f"  {name:20s} params = {lean_params[name]/1e6:5.2f} M  fwd = {fwd:6.2f} ms  fwd+bwd = {bwd:6.2f} ms")

    print("\nReference: depth benchmark ran at ~4 iters/s = ~250 ms/iter wall (env + render + net)")


if __name__ == "__main__":
    main()
