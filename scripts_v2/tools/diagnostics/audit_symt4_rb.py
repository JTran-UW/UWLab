"""Audit a recorded SymT4 expert buffer: metadata, shapes, success-episode fraction, and internal consistency
of the recorded tensor features (partial-trace identity T_ijkk = delta_ij - d_i d_j  =>  trace = 2, |d| = 1),
plus the layout constants (peg terms at dims 0:90 and 187:277; prev_actions 90:97)."""
import sys, torch
src = sys.argv[1]
P = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
bt, meta = P["buffer_tensors"], P["metadata"]
E = 256
o = bt["observations"][:E].float(); c = bt["critic_observations"][:E].float()
print(f"task: {meta.get('task')} | n_obs {meta['n_obs']} n_critic_obs {meta.get('n_critic_obs')} | actor {tuple(bt['observations'].shape)} critic {tuple(bt['critic_observations'].shape)} | n_step {meta.get('n_step', meta.get('n_steps'))}")
d = bt["dones"].bool(); t = bt["truncations"].bool()
ends = d | t
n_ep = int(ends.sum()); n_tr = int(t.sum())
print(f"episodes {n_ep}; ended by truncation (success-truncated or time-out) {n_tr} ({n_tr / max(n_ep, 1):.1%}); terminal (abnormal etc.) {n_ep - n_tr}")
# success-truncated = truncation with a positive reward at the end step is not stored; use truncations w/o done as the succ proxy like before
r = bt["rewards"][:E]
print(f"reward at truncation steps: mean {r[t[:E]].mean():.3f}, frac>0 {(r[t[:E]] > 0).float().mean():.3f}")
fails = []
ok = lambda cnd, m: fails.append(m) if not cnd else None
ok(meta["n_obs"] == 307 and meta.get("n_critic_obs") == 307, "n_obs/n_critic_obs != 307")
ok(torch.equal(o, c), "actor and critic obs differ (should be identical groups)")
NAMES = ["xxxx", "xxxy", "xxxz", "xxyy", "xxyz", "xxzz", "xyyy", "xyyz", "xyzz", "xzzz", "yyyy", "yyyz", "yyzz", "yzzz", "zzzz"]
def check_block(x, start, label):
    blk = x[..., start:start + 90].reshape(*x.shape[:-1], 5, 18)
    T = blk[..., 3:]; comp = {n: T[..., i] for i, n in enumerate(NAMES)}
    t4 = lambda i, j, k, l: comp["".join(sorted("xyz"[m] for m in (i, j, k, l)))]
    M = torch.stack([torch.stack([sum(t4(i, j, k, k) for k in range(3)) for j in range(3)], -1) for i in range(3)], -2)  # I - d d^T
    tr = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    dd = torch.eye(3) - M  # d d^T
    ev = torch.linalg.eigvalsh(dd)  # should be (0, 0, 1)
    print(f"[{label}] trace(I - dd^T) mean {tr.mean():.4f} (expect 2) max|err| {(tr - 2).abs().max():.2e}; dd^T eigen max|err| {(ev - torch.tensor([0., 0., 1.])).abs().max():.2e}; T range [{T.min():.3f},{T.max():.3f}]")
    ok((tr - 2).abs().max() < 1e-3 and (ev - torch.tensor([0., 0., 1.])).abs().max() < 1e-3, f"{label}: tensor block inconsistent")
check_block(o, 0, "peg_in_hole_t4"); check_block(o, 187, "peg_in_gripper_t4")
pa = o[..., 90:97]; act = bt["actions"][:E].float()
print(f"prev_actions[t+1] == actions[t] (within episodes) max err {(pa[:, 1:] - act[:, :-1])[~ends[:E, :-1]].abs().max():.2e}")
print("RESULT:", "PASS" if not fails else "FAIL: " + " | ".join(fails))
