"""Pure-torch unit test of tensor4 features: 8-element invariance, 45-deg control, partial-trace identity, ordering."""
import torch, math, itertools, numpy as np, importlib.util
import isaaclab.utils.math as mu
spec = importlib.util.spec_from_file_location("so", "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/symmetric_obs.py")
so = importlib.util.module_from_spec(spec); spec.loader.exec_module(so)
torch.manual_seed(0); N = 2000
q = mu.random_orientation(N, "cpu"); p = torch.randn(N, 3); fp = torch.randn(N, 3); fq = mu.random_orientation(N, "cpu")
f0 = so.tensor4_pose_features(p, q, fp, fq)
worst = 0.0
for g in so.symmetry_group_local():
    worst = max(worst, (so.tensor4_pose_features(p, mu.quat_mul(q, g.expand(N, 4)), fp, fq) - f0).abs().max().item())
a = torch.tensor([0., 0., math.pi / 4]); q45 = mu.quat_mul(q, mu.quat_from_angle_axis(a.norm().view(1), (a / a.norm()).view(1, 3)).expand(N, 4))
d45 = (so.tensor4_pose_features(p, q45, fp, fq) - f0).abs().amax(-1)
T = f0[:, 3:]; comp = {n: T[:, i] for i, n in enumerate(so.TENSOR4_NAMES)}
t4 = lambda i, j, k, l: comp["".join(sorted("xyz"[m] for m in (i, j, k, l)))]
_, rq = mu.subtract_frame_transforms(fp, fq, p, q); d = mu.quat_apply(rq, torch.tensor([0., 0, 1]).expand(N, 3))
err = max((sum(t4(i, j, k, k) for k in range(3)) - ((1.0 if i == j else 0.0) - d[:, i] * d[:, j])).abs().max().item() for i in range(3) for j in range(3))
R = mu.matrix_from_quat(rq[:5]).numpy(); IDX4 = list(itertools.combinations_with_replacement(range(3), 4))
ref = np.array([[np.prod(R[n][:, 0][list(c)]) + np.prod(R[n][:, 1][list(c)]) for c in IDX4] for n in range(5)])
oerr = np.abs(ref - T[:5].numpy()).max()
print("names:", so.TENSOR4_NAMES)
print(f"invariance over 8 group elements: max |dF| = {worst:.2e}")
print(f"45-deg yaw control: min |dF| over samples = {d45.min():.3f} (must be > 0)")
print(f"partial-trace identity T_ijkk = delta_ij - d_i d_j: max err {err:.2e}")
print(f"match with video-script numpy ordering: max err {oerr:.2e}")
print("RESULT:", "PASS" if worst < 1e-5 and d45.min() > 1e-2 and err < 1e-5 and oerr < 1e-5 else "FAIL")
