"""Force fixed dataset reset states on env 0..N-1, dump policy obs per term + expert action, and step a few times."""
import argparse, json
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--checkpoint", default="expert_seed0_rslrl52.pt")
parser.add_argument("--reset_type", default="ObjectAnywhereEEGrasped")
parser.add_argument("--indices", default="0,1,2,3")
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, sys, os
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_from_nested_dict
sys.path.insert(0, os.path.join(os.getcwd(), "scripts_v2/tools/diagnostics"))
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
    idxs = [int(v) for v in args_cli.indices.split(",")]
    env_cfg.scene.num_envs = len(idxs); env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = [args_cli.reset_type]; p["probs"] = [1.0]
    # kill randomization so both backends see nominal dynamics
    for n in ("randomize_robot_mass", "randomize_insertive_object_mass", "randomize_receptive_object_mass", "randomize_table_mass",
              "robot_material", "insertive_object_material", "receptive_object_material", "table_material", "randomize_gripper_actuator_parameters"):
        if hasattr(env_cfg.events, n): setattr(env_cfg.events, n, None)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    ids = torch.tensor(idxs, device=env.device)
    states = sample_from_nested_dict(term.datasets[0], ids)
    term._reset_to(states["initial_state"], env_ids=torch.arange(len(idxs), device=env.device), is_relative=True)
    ds = states["initial_state"]["rigid_object"]["insertive_object"]
    peg = env.scene["insertive_object"]
    print("DATASET peg root_pose env0:", [round(v,4) for v in ds["root_pose"][0].tolist()], "env_origin:", [round(v,3) for v in env.scene.env_origins[0].tolist()])
    print("AFTER _reset_to (no step) peg root_pos_w:", [round(v,4) for v in T(peg.data.root_pos_w)[0].tolist()], "quat:", [round(v,3) for v in T(peg.data.root_quat_w)[0].tolist()])
    if hasattr(peg.data, "root_link_pos_w"): print("   root_link_pos_w:", [round(v,4) for v in T(peg.data.root_link_pos_w)[0].tolist()], "root_com_pos_w:", [round(v,4) for v in T(peg.data.root_com_pos_w)[0].tolist()] if hasattr(peg.data,"root_com_pos_w") else None)
    print("   robot root_pos_w:", [round(v,4) for v in T(env.scene["robot"].data.root_pos_w)[0].tolist()])
    env.action_manager.reset(); env.observation_manager.reset()
    for tname in ("rel_cartesian_osc", "arm", "gripper"):
        pass
    for at in env.action_manager._terms.values(): at.reset(None)
    env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
    print("AFTER 1 physics step peg root_pos_w:", [round(v,4) for v in T(peg.data.root_pos_w)[0].tolist()])
    policy = load_policy(args_cli.checkpoint, env.device)
    names = env.observation_manager.active_terms["policy"]; dims = env.observation_manager.group_obs_term_dim["policy"]
    robot = env.scene["robot"]; jn = list(robot.joint_names)
    rec = {"backend": type(env_cfg.sim.physics).__name__, "joint_names": jn, "terms": names, "dims": [int(d[0]) for d in dims], "steps": []}
    import isaaclab.utils.math as mu
    bn = list(robot.body_names); base = bn.index("robotiq_base_link")
    bp, bq = T(robot.data.body_pos_w), T(robot.data.body_quat_w)
    rec["bodies"] = {}
    rec["body_world"] = {b: {"pos": bp[:, i].cpu().tolist(), "quat": bq[:, i].cpu().tolist()} for i, b in enumerate(bn)}
    rec["joint_pos_reset"] = T(robot.data.joint_pos).cpu().tolist()
    for b in ("left_inner_finger", "right_inner_finger", "left_inner_knuckle", "right_inner_knuckle", "left_outer_finger", "right_outer_finger", "wrist_3_link"):
        i = bn.index(b)
        pb, qb = mu.subtract_frame_transforms(bp[:, base], bq[:, base], bp[:, i], bq[:, i])
        rec["bodies"][b] = {"pos_in_base": pb.cpu().tolist(), "quat_in_base": qb.cpu().tolist()}
    pp, pq = mu.subtract_frame_transforms(bp[:, base], bq[:, base], T(env.scene["insertive_object"].data.root_pos_w), T(env.scene["insertive_object"].data.root_quat_w))
    rec["bodies"]["peg"] = {"pos_in_base": pp.cpu().tolist(), "quat_in_base": pq.cpu().tolist()}
    act = torch.zeros(env.action_space.shape, device=env.device)
    for t in range(args_cli.steps):
        obs = env.observation_manager.compute()["policy"]
        a = policy(obs)
        rec["steps"].append({"t": t, "obs": obs.cpu().tolist(), "action": a.cpu().tolist(),
                             "joint_pos": T(robot.data.joint_pos).cpu().tolist(),
                             "peg_pos": T(env.scene["insertive_object"].data.root_pos_w).cpu().tolist()})
        env.step(a)
    json.dump(rec, open(args_cli.out, "w"))
    print("wrote", args_cli.out, "terms", list(zip(names, rec["dims"])))
    env.close()
main(); app.close()
