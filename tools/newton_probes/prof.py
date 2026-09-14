import argparse, time
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--reset_type", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, os
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x
def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck["actor_state_dict"]; W = [sd[f"mlp.{i}.weight"] for i in (0, 2, 4, 6, 8)]; b = [sd[f"mlp.{i}.bias"] for i in (0, 2, 4, 6, 8)]
    mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
    def policy(obs):
        x = (obs - mean) / (std + 1e-2)
        for i, (Wi, bi) in enumerate(zip(W, b)):
            x = x @ Wi.T + bi
            if i < 4: x = torch.nn.functional.elu(x)
        return x
    return policy

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = "./Datasets/OmniReset_patched"
    if args_cli.reset_type:
        env_cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]; env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    obs, _ = env.reset()
    policy = load_policy("expert_seed0_rslrl52.pt", env.device)
    # instrument: wrap sim.step and action apply
    import isaaclab.sim as sim_utils
    sim = env.sim; t_phys = [0.0]; t_act = [0.0]; t_evt = [0.0]
    orig_step = sim.step
    def timed_step(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = orig_step(*a, **k); torch.cuda.synchronize(); t_phys[0] += time.perf_counter() - t0; return r
    sim.step = timed_step
    am = env.action_manager; orig_apply = am.apply_action
    def timed_apply(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = orig_apply(*a, **k); torch.cuda.synchronize(); t_act[0] += time.perf_counter() - t0; return r
    am.apply_action = timed_apply
    em = env.event_manager; orig_reset_idx = env._reset_idx
    def timed_reset(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = orig_reset_idx(*a, **k); torch.cuda.synchronize(); t_evt[0] += time.perf_counter() - t0; return r
    env._reset_idx = timed_reset
    # split collide vs solver (requires use_cuda_graph=False)
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    t_col = [0.0]; t_sol = [0.0]
    if NM._collision_pipeline is not None:
        cp = NM._collision_pipeline; oc = cp.collide
        def tc(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter(); r = oc(*a, **k); torch.cuda.synchronize(); t_col[0] += time.perf_counter() - t0; return r
        cp.collide = tc
    orss = NM._run_solver_substeps
    def trs(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = orss(*a, **k); torch.cuda.synchronize(); t_sol[0] += time.perf_counter() - t0; return r
    NM._run_solver_substeps = trs
    for _ in range(3): obs, *_ = env.step(policy(obs["policy"]))  # warmup
    t_col[0] = t_sol[0] = 0.0
    t_phys[0] = t_act[0] = t_evt[0] = 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter(); n_resets = 0
    for t in range(args_cli.steps):
        obs, r, term, trunc, info = env.step(policy(obs["policy"]) + 0.5 * torch.randn(env.num_envs, 7, device=env.device))
        n_resets += int((term | trunc).sum())
    torch.cuda.synchronize(); tot = time.perf_counter() - t0
    print(f"PROFILE envs={env.num_envs} steps={args_cli.steps}: total {tot:.2f}s = {tot/args_cli.steps*1000:.0f} ms/step | physics(sim.step) {t_phys[0]:.2f}s | action apply {t_act[0]:.2f}s | reset_idx {t_evt[0]:.2f}s | other {tot-t_phys[0]-t_act[0]-t_evt[0]:.2f}s | collide {t_col[0]:.2f}s solver {t_sol[0]:.2f}s | resets {n_resets} | clamp={os.environ.get('UWLAB_OSC_VEL_CLAMP','auto')} notify={os.environ.get('UWLAB_NEWTON_ROOT_NOTIFY','on')}")
    try:
        import numpy as np
        opt = NM._solver.mjw_model.opt
        ni = NM._solver.mjw_data.solver_niter.numpy()
        print(f"SOLVER opt.iterations={opt.iterations} ls_iterations={opt.ls_iterations} tolerance={opt.tolerance} ls_tolerance={getattr(opt,'ls_tolerance',None)} solver={opt.solver} cone={opt.cone} impratio={getattr(opt,'impratio',None)} | niter: max={ni.max()} mean={ni.mean():.1f} p50={np.percentile(ni,50):.0f} p90={np.percentile(ni,90):.0f} p99={np.percentile(ni,99):.0f} n_at_cap={(ni>=opt.iterations).sum()}/{ni.size}")
        d = NM._solver.mjw_data
        for name in ("nefc", "ncon", "nacon", "njmax", "naconmax"):
            v = getattr(d, name, None)
            if v is None: continue
            try:
                a = v.numpy(); print(f"SOLVER {name}: max={a.max()} mean={a.mean():.1f} p90={np.percentile(a,90):.0f}")
            except Exception:
                print(f"SOLVER {name}: {v}")
        c = NM._contacts
        if c is not None and hasattr(c, "rigid_contact_count"): print(f"NEWTON contacts total={int(c.rigid_contact_count.numpy()[0])} per env={c.rigid_contact_count.numpy()[0]/env.num_envs:.1f} reset_type={args_cli.reset_type}")
    except Exception as ex:
        print("SOLVER stats unavailable:", ex)
    env.close()
main(); app.close()
