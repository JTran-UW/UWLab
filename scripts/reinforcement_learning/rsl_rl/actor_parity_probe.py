"""Compare rsl_rl 5.2 actor forward vs a manual MLP on identical weights/obs."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--expert2x", default="peg_state_rl_expert_seed0.pt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
import sys; sys.argv = [sys.argv[0]] + remaining
app = AppLauncher(args_cli).app
import gymnasium as gym, torch
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from uwlab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli); agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = "./Datasets/OmniReset"
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False})
    dev = env.unwrapped.device
    ld0 = torch.load(args_cli.checkpoint, map_location=dev, weights_only=False)["actor_state_dict"]
    A = runner.alg._raw_actor
    w = lambda: A.mlp._modules["0"].weight
    print("\n== load isolation")
    print("  after runner.load: mlp.0 |W|=", round(w().abs().mean().item(), 4), " equals ckpt:", torch.equal(w(), ld0["mlp.0.weight"]), " normalizer count:", A.obs_normalizer.count.item())
    missing = A.load_state_dict(ld0, strict=True)
    print("  after direct A.load_state_dict:", missing, " mlp.0 |W|=", round(w().abs().mean().item(), 4), " equals ckpt:", torch.equal(w(), ld0["mlp.0.weight"]))
    print("  A.state_dict keys:", [k for k in A.state_dict().keys()])
    print("  runner.alg.actor is _raw_actor:", runner.alg.actor is A, "| type(alg.actor):", type(runner.alg.actor).__name__)
    sd = torch.load(args_cli.expert2x, map_location=dev, weights_only=False)["model_state_dict"]
    Ws = [sd[f"actor.{i}.weight"] for i in (0, 2, 4, 6, 8)]; bs = [sd[f"actor.{i}.bias"] for i in (0, 2, 4, 6, 8)]
    mean, std = sd["actor_obs_normalizer._mean"], sd["actor_obs_normalizer._std"]
    def manual(o):
        x = (o - mean) / (std + 1e-8)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            x = x @ W.T + b
            if i < 4: x = torch.nn.functional.elu(x)
        return x
    obs = env.get_observations()
    print("\nobs type:", type(obs).__name__, "keys:", list(obs.keys()) if hasattr(obs, "keys") else None)
    pol = obs["policy"] if hasattr(obs, "keys") else obs
    print("policy obs shape:", tuple(pol.shape), "finite:", bool(torch.isfinite(pol).all()))
    actor = runner.alg._raw_actor if hasattr(runner.alg, "_raw_actor") else runner.alg.policy
    print("actor class:", type(actor).__name__, "| modules:", [type(m).__name__ for m in actor.children()])
    print("actor mlp:", getattr(actor, "mlp", None))
    n = actor.obs_normalizer
    print("normalizer:", type(n).__name__, "training:", n.training, "| _mean match:", torch.allclose(n._mean, mean), "_std match:", torch.allclose(n._std, std), "count:", n.count.item() if hasattr(n, "count") else None)
    with torch.inference_mode():
        a_manual = manual(pol)
        infer = runner.get_inference_policy(device=dev)
        a_runner = infer(obs)
        # also the raw actor on the policy tensor, with and without normalizer
        try:
            a_raw = actor(pol)
        except Exception as e:
            a_raw = None; print("actor(pol) failed:", e)
        try:
            a_raw_td = actor(obs)
        except Exception as e:
            a_raw_td = None; print("actor(obs td) failed:", e)
    def cmp(name, a):
        if a is None: return
        a = a if torch.is_tensor(a) else a.get("action", a)
        print(f"  {name:28s} shape={tuple(a.shape)}  max|diff vs manual|={(a - a_manual).abs().max().item():.4g}  mean|a|={a.abs().mean().item():.3f}  gripper mean={a[:, -1].mean().item():+.3f}")
    print(f"  {'manual MLP (2.x expert)':28s} shape={tuple(a_manual.shape)}  mean|a|={a_manual.abs().mean().item():.3f}  gripper mean={a_manual[:, -1].mean().item():+.3f}")
    cmp("runner inference policy", a_runner); cmp("actor(policy tensor)", a_raw); cmp("actor(obs tensordict)", a_raw_td)
    # normalizer applied?
    with torch.inference_mode():
        x = n(pol) if callable(n) else None
        if x is not None: print("  normalizer(pol) vs manual norm max|diff|:", ((x - (pol - mean) / (std + 1e-8)).abs().max().item()))
        # feed the normalized obs directly through the mlp
        try:
            y = actor.mlp(x); print("  mlp(normalized) vs manual max|diff|:", (y - a_manual).abs().max().item())
        except Exception as e: print("  mlp(normalized) failed:", e)
    print("\n== get_latent diagnosis")
    print("  actor.obs_groups:", actor.obs_groups, "| obs_dim:", actor.obs_dim)
    with torch.inference_mode():
        lat = actor.get_latent(obs)
        print("  get_latent shape:", tuple(lat.shape), " vs normalizer(pol) max|diff|:", (lat - n(pol)).abs().max().item())
        raw_cat = torch.cat([obs[g] for g in actor.obs_groups], dim=-1)
        print("  raw concat shape:", tuple(raw_cat.shape), " equals pol:", torch.equal(raw_cat, pol), " max|diff|:", (raw_cat - pol).abs().max().item() if raw_cat.shape == pol.shape else "shape mismatch")
        print("  obs['policy'][:2,:6]:", pol[:2, :6].tolist())
        print("  raw_cat[:2,:6]      :", raw_cat[:2, :6].tolist())
        print("  obs batch_size:", obs.batch_size, " obs['policy'] is view of td:", obs["policy"].data_ptr() == pol.data_ptr())
    print("\n== mlp children walk")
    ch = list(actor.mlp.children()); print("  children:", [type(c).__name__ + (f"({c.in_features}->{c.out_features})" if hasattr(c, "in_features") else "") for c in ch])
    print("  named_children:", [k for k, _ in actor.mlp.named_children()])
    print("  mlp forward source:", type(actor.mlp).__mro__[:3])
    with torch.inference_mode():
        f = lat
        for layer in ch[:-1]: f = layer(f)
        walk_out = ch[-1](f)
        seq_out = actor.mlp(lat)
        print("  walk vs mlp(lat) max|diff|:", (walk_out - seq_out).abs().max().item(), " walk mean|a|:", walk_out.abs().mean().item(), " seq mean|a|:", seq_out.abs().mean().item())
        det = actor.distribution.deterministic_output(walk_out)
        print("  deterministic_output(walk) mean|a|:", det.abs().mean().item(), " vs walk max|diff|:", (det - walk_out).abs().max().item())
        full = actor(obs)
        print("  actor(obs) mean|a|:", full.abs().mean().item(), " vs walk max|diff|:", (full - walk_out).abs().max().item())
    print("\n== eps + activation-drop effect on the trained expert")
    with torch.inference_mode():
        for eps in (1e-8, 1e-2):
            x = (pol - mean) / (std + eps)
            for i, (W, b) in enumerate(zip(Ws, bs)):
                x = x @ W.T + b
                if i < 4: x = torch.nn.functional.elu(x)
            print(f"  manual eps={eps:g}: mean|a|={x.abs().mean().item():.3f}  vs seq(lat) max|diff|={(x - seq_out).abs().max().item():.4f}  gripper mean={x[:, -1].mean().item():+.3f}")
        # fraction of hidden pre-activations that are negative (where dropping ELU matters)
        h = lat; negs = []
        for i, layer in enumerate(list(actor.mlp._modules.values())):
            h = layer(h)
            if isinstance(layer, torch.nn.Linear) and layer.out_features != 7: negs.append((h < 0).float().mean().item())
        print("  fraction negative pre-activations per hidden layer (seq path):", [round(v, 3) for v in negs])
        # random-init network: activation drop effect
        torch.manual_seed(0)
        rnd = type(actor.mlp)(215, 7, [512, 256, 128, 64], activation="elu").to(dev)
        z = torch.randn(64, 215, device=dev)
        f = z
        chr_ = list(rnd.children())
        for layer in chr_[:-1]: f = layer(f)
        walk_r = chr_[-1](f); seq_r = rnd(z)
        print(f"  random-init MLP: walk vs seq max|diff|={(walk_r - seq_r).abs().max().item():.4f}  seq mean|a|={seq_r.abs().mean().item():.3f} walk mean|a|={walk_r.abs().mean().item():.3f}")
    print("\n== weight equality runner actor vs 2.x checkpoint")
    mods = dict(actor.mlp.named_children())
    for i in (0, 2, 4, 6, 8):
        W = mods[str(i)].weight; Wc = sd[f"actor.{i}.weight"]
        print(f"  layer {i}: shape {tuple(W.shape)} equal={torch.equal(W, Wc)} max|diff|={(W - Wc).abs().max().item():.4g}  |W| mean={W.abs().mean().item():.4f} |Wc| mean={Wc.abs().mean().item():.4f}")
    ld = torch.load(args_cli.checkpoint, map_location=dev, weights_only=False)["actor_state_dict"]
    print("  converted ckpt mlp.0.weight equals sd actor.0.weight:", torch.equal(ld["mlp.0.weight"], sd["actor.0.weight"]))
    print("  runner actor state_dict keys:", list(actor.state_dict().keys())[:6])
    with torch.inference_mode():
        h = lat
        for k, layer in actor.mlp._modules.items():
            h = layer(h); print(f"    after {k} {type(layer).__name__}: mean|h|={h.abs().mean().item():.4f}")
        h2 = (pol - mean) / (std + 1e-2)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            h2 = h2 @ W.T + b
            if i < 4: h2 = torch.nn.functional.elu(h2)
            print(f"    manual after layer {i}: mean|h|={h2.abs().mean().item():.4f}")
    env.close()
main(); app.close()
