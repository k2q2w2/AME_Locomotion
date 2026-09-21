"""Paired first-episode finetune terrain evaluation of AME and GLAD checkpoints.

Run with Isaac Lab's Python and --headless. No training, optimizer load or task
configuration edits. Checkpoints load strictly. JSON records every trial before
Isaac Lab auto-reset, plus pairing checks and terrain hashes.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--variants', type=int, default=4)
parser.add_argument('--replicas', type=int, default=4)
parser.add_argument('--seeds', type=int, nargs='+', default=[1001,1002,1003])
parser.add_argument('--difficulties', type=float, nargs='+', default=[0.2,0.5,0.8])
parser.add_argument('--terrain_seed', type=int, default=20260922)
parser.add_argument('--half_width', type=float, default=1.0, help='Allowed lateral deviation from terrain center (m).')
parser.add_argument('--lsio_checkpoint', default='logs/rsl_rl/g1_ame_lsio/2026-09-18_13-12-26/model_19998.pt')
parser.add_argument('--glad_checkpoint', default='logs/rsl_rl/g1_ame_glad/2026-09-20_15-19-10_finetune_from10000_2gpu/model_19999.pt')
parser.add_argument('--clean_checkpoint', default='logs/rsl_rl/g1_ame_glad_critic_clean_stop_grad/2026-09-21_04-24-21_finetune_from10000_auto_2gpu/model_19998.pt')
parser.add_argument('--models', nargs='+', choices=['AME-LSIO','AME1','AME2','GLAD','GLAD-CleanStopGrad'], default=['AME-LSIO','AME1','AME2'])
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.variants < 1 or args.replicas < 1 or not all(0 <= d <= 1 for d in args.difficulties):
    parser.error('Positive variants/replicas and difficulties in [0,1] required.')
if not 0 < args.half_width <= 3.5:
    parser.error('half_width must be within (0, 3.5] to remain inside the terrain tile.')
if len(set(args.models)) != len(args.models):
    parser.error('Each model must occur only once.')
app_launcher = AppLauncher(args)
app = app_launcher.app

import copy
import hashlib
import importlib
import json
import time
from unittest.mock import patch
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains import TerrainGenerator
from isaaclab.utils.io import dump_yaml
from rsl_rl.modules import ActorCriticEncoder, ActorCriticEncoderLSIO, ActorCriticEncoderGLAD
from terrain_eval_metrics import traversal_outcome, OUTCOMES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(args.output).resolve()
OUT.mkdir(parents=True, exist_ok=True)
terrain_manifest = []


class FixedTerrainGenerator(TerrainGenerator):
    """Exact difficulty per row, equal terrain coverage, seeded custom RNGs."""
    def _generate_curriculum_terrains(self):
        terrains = list(self.cfg.sub_terrains.items())
        original_rng = np.random.default_rng
        for col in range(self.cfg.num_cols):
            name, cfg = terrains[col // args.variants]
            for row, difficulty in enumerate(args.difficulties):
                mesh_seed = args.terrain_seed + col * 100 + row
                state = np.random.get_state()
                np.random.seed(mesh_seed)
                # Project heightfield functions call default_rng() without a seed.
                # Scope the override to one mesh; leave training generators intact.
                def seeded_rng(seed=None):
                    return original_rng(mesh_seed if seed is None else seed)
                try:
                    with patch.object(np.random, 'default_rng', seeded_rng):
                        mesh, origin = self._get_terrain_mesh(difficulty, copy.deepcopy(cfg))
                finally:
                    np.random.set_state(state)
                digest = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()
                terrain_manifest.append(dict(terrain=name, row=row, col=col, difficulty=difficulty,
                                             variant=col % args.variants, mesh_seed=mesh_seed, sha256=digest))
                self._add_sub_terrain(mesh, origin, row, col, cfg)


class TrialRecorder:
    def __init__(self, env, cells):
        self.env = env
        self.cells = cells
        self.feet, _ = env.scene['contact_forces'].find_bodies('.*_ankle_roll_link')
        assert len(self.feet) == 2
        self.start()

    def start(self):
        n, device = self.env.num_envs, self.env.device
        self.active = torch.ones(n, dtype=torch.bool, device=device)
        self.outcome = torch.zeros(n, dtype=torch.long, device=device)
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.error = torch.zeros(n, device=device)
        self.max_x = torch.full((n,), -100., device=device)
        self.final_xyz = torch.zeros(n, 3, device=device)
        self.final_error = torch.zeros(n, device=device)
        self.final_steps = torch.zeros(n, dtype=torch.long, device=device)

    def step(self):
        e = self.env
        robot = e.scene['robot']
        relative = robot.data.root_pos_w - e.scene.env_origins
        contact = e.termination_manager.get_term('base_contact')
        timeout = e.termination_manager.get_term('time_out')
        forces = e.scene['contact_forces'].data.net_forces_w[:, self.feet, :]
        supported = (forces.norm(dim=-1) > 20.).any(dim=-1)
        upright = robot.data.projected_gravity_b[:, 2] < -0.5
        code = traversal_outcome(relative, contact, timeout, supported, upright, half_width=args.half_width)
        self.steps += self.active.long()
        velocity = robot.data.root_lin_vel_b[:, :2]
        target = e.command_manager.get_command('base_velocity')[:, :2]
        self.error += (velocity - target).norm(dim=-1) * self.active
        self.max_x = torch.where(self.active, torch.maximum(self.max_x, relative[:, 0]), self.max_x)
        done = self.active & (code != 0)
        self.outcome[done] = code[done]
        self.final_xyz[done] = relative[done]
        self.final_error[done] = self.error[done] / self.steps[done]
        self.final_steps[done] = self.steps[done]
        self.active[done] = False
        return done

    def records(self, model, seed):
        assert not self.active.any()
        records = []
        for i, cell in enumerate(self.cells):
            code = int(self.outcome[i])
            records.append(dict(model=model, seed=seed, env_id=i, **cell, outcome=OUTCOMES[code],
                                success=code == 1, seconds=float(self.final_steps[i]) * self.env.step_dt,
                                max_forward_m=float(self.max_x[i]), final_xyz_m=self.final_xyz[i].tolist(),
                                mean_velocity_error_m_s=float(self.final_error[i])))
        return records


def record_termination(env):
    if not hasattr(env, 'terrain_eval_recorder'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.terrain_eval_recorder.step()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    module = importlib.import_module('ame_locomotion.tasks.manager_based.ame_locomotion.29dof.velocity_env_cfg_29dof')
    history_module = importlib.import_module('ame_locomotion.tasks.manager_based.ame_locomotion.29dof.lsio_observations')
    terrain_module = importlib.import_module('ame_locomotion.tasks.manager_based.ame_locomotion.terrains.finetune_terrain_cfg')
    cfg = module.G1RoughEnvCfg()
    cfg.seed = args.terrain_seed
    cfg.sim.device = args.device or 'cuda:0'
    cfg.scene.terrain.terrain_generator = copy.deepcopy(terrain_module.FINETUNE_ROUGH_TERRAINS_CFG)
    gen = cfg.scene.terrain.terrain_generator
    gen.class_type = FixedTerrainGenerator
    gen.seed = args.terrain_seed
    gen.curriculum = True  # Fixed generator uses this entry point; no adaptive env curriculum.
    gen.num_rows = len(args.difficulties)
    gen.num_cols = len(gen.sub_terrains) * args.variants
    gen.border_width = 5.
    for sub in gen.sub_terrains.values():
        sub.proportion = 1. / len(gen.sub_terrains)
    cfg.scene.num_envs = gen.num_rows * gen.num_cols * args.replicas
    cfg.scene.terrain.num_envs = cfg.scene.num_envs
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.episode_length_s = 10.
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.height_scan.params['noise'] = False
    history = history_module.make_lsio_observations(cfg.observations)
    cfg.observations = {'policy': cfg.observations.policy, 'critic': cfg.observations.critic,
                        'lsio_policy': history.policy, 'proprio_history': history.proprio_history}
    for name in ['physics_material','add_base_mass','base_com','base_external_force_torque','push_robot']:
        setattr(cfg.events, name, None)
    cfg.events.reset_base.params['pose_range'] = {'x':(-.025,.025),'y':(-.05,.05),'yaw':(-.03,.03)}
    cfg.events.reset_robot_joints.params['position_range'] = (1.,1.)
    cfg.events.reset_robot_joints.params['velocity_range'] = (-.1,.1)
    command = cfg.commands.base_velocity
    command.debug_vis = False
    command.resampling_time_range = (100.,100.)
    command.ranges.lin_vel_x = (1.,1.)
    command.ranges.lin_vel_y = (0.,0.)
    command.ranges.ang_vel_z = (-1.,1.)
    command.ranges.heading = (0.,0.)
    command.heading_command = True
    command.rel_heading_envs = 1.
    command.rel_standing_envs = 0.
    cfg.terminations.evaluation = TerminationTermCfg(func=record_termination)
    dump_yaml(str(OUT / 'environment.yaml'), cfg)
    env = ManagerBasedRLEnv(cfg)
    try:
        # Pair every model on identical fixed geometry and initial state samples.
        cells = []
        for row, difficulty in enumerate(args.difficulties):
            for col in range(gen.num_cols):
                for replica in range(args.replicas):
                    cells.append(dict(terrain=list(gen.sub_terrains)[col // args.variants], difficulty=difficulty,
                                      row=row, col=col, variant=col % args.variants, replica=replica))
        rows = torch.tensor([c['row'] for c in cells], device=env.device)
        cols = torch.tensor([c['col'] for c in cells], device=env.device)
        terrain = env.scene.terrain
        terrain.terrain_levels[:] = rows
        terrain.terrain_types[:] = cols
        terrain.env_origins[:] = terrain.terrain_origins[rows, cols]
        env.scene.env_origins[:] = terrain.env_origins
        write_json(OUT / 'terrain_manifest.json', terrain_manifest)
        obs, _ = env.reset(seed=args.seeds[0])
        policies = {}
        checkpoints = {}
        paths = {'AME-LSIO': ROOT / args.lsio_checkpoint, 'AME1': ROOT / 'pretrained/ame1.pt', 'AME2': ROOT / 'pretrained/ame2.pt',
                 'GLAD': ROOT / args.glad_checkpoint, 'GLAD-CleanStopGrad': ROOT / args.clean_checkpoint}
        for name in args.models:
            path = paths[name]
            state = torch.load(path, map_location=env.device, weights_only=False)
            stop_grad = name == 'GLAD-CleanStopGrad'
            if state.get('critic_encoder_stop_grad', False) != stop_grad:
                raise ValueError(f'{name}: checkpoint critic stop-grad mode mismatch.')
            if stop_grad and state.get('critic_feature_source') != 'critic':
                raise ValueError('CleanStopGrad requires a Critic-source checkpoint, not Actor-source StopGrad.')
            uses_history = name in ('AME-LSIO', 'GLAD', 'GLAD-CleanStopGrad')
            groups = {'policy':['lsio_policy' if uses_history else 'policy'], 'critic':['critic']}
            cls = ActorCriticEncoder
            if uses_history:
                groups['history'] = ['proprio_history']
                cls = ActorCriticEncoderLSIO
            if name.startswith('GLAD'):
                cls = ActorCriticEncoderGLAD
            policy = cls(obs, groups, env.action_manager.total_action_dim,
                         attach_global=name in ('AME2', 'GLAD', 'GLAD-CleanStopGrad'),
                         critic_encoder_stop_grad=stop_grad, critic_feature_source='critic').to(env.device)
            policy.load_state_dict(state['model_state_dict'], strict=True)
            policy.eval()
            policies[name] = policy
            checkpoints[name] = dict(path=str(path), iter=state.get('iter'), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     critic_encoder_stop_grad=stop_grad, critic_feature_source='critic',
                                     architecture=state['model_state_dict'].get('_extra_state'))
        recorder = TrialRecorder(env, cells)
        env.terrain_eval_recorder = recorder
        all_records, pairing = [], []
        protocol = dict(difficulties=args.difficulties, variants=args.variants, replicas=args.replicas,
                        seeds=args.seeds, terrain_seed=args.terrain_seed, num_envs=env.num_envs,
                        half_width=args.half_width,
                        command='vx=1 m/s, vy=0; heading=0 with existing heading controller',
                        success=f'x >= 3.5 m from terrain center, |y| <= {args.half_width} m, x <= 3.9 m, gravity_z < -0.5, at least one foot contact >20 N; no illegal contact',
                        deadline_seconds=10., checkpoint_metadata=checkpoints,
                        note='One first episode per environment/model/seed; no observation noise, pushes or randomized physical properties. Shared layouts imply correlated trials.')
        write_json(OUT / 'protocol.json', protocol)
        for seed_index, seed in enumerate(args.seeds):
            reference = None
            reference_obs = None
            order = list(policies)
            offset = seed_index % len(order)
            order = order[offset:] + order[:offset]
            for name in order:
                obs, _ = env.reset(seed=seed)
                recorder.start()
                initial = torch.cat((env.scene['robot'].data.root_state_w, env.scene['robot'].data.joint_pos,
                                     env.scene['robot'].data.joint_vel), dim=-1).clone()
                if reference is None:
                    reference = initial
                    reference_obs = {key: value.clone() for key, value in obs.items()}
                difference = (initial - reference).abs().max().item()
                obs_difference = max((obs[key] - value).abs().max().item() for key, value in reference_obs.items())
                if difference > 1e-5:
                    raise RuntimeError(f'Initial states not paired: {name} seed={seed} delta={difference}')
                if obs_difference > 1e-4:
                    raise RuntimeError(f'Initial observations not paired: {name} seed={seed} delta={obs_difference}')
                pairing.append(dict(model=name, seed=seed, max_initial_state_delta=difference,
                                    max_initial_observation_delta=obs_difference,
                                    initial_state_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest()))
                began = time.monotonic()
                # Env buffers survive this block and are mutated by the next
                # reset; no_grad avoids creating inference-only tensors.
                with torch.no_grad():
                    for step in range(env.max_episode_length + 2):
                        actions, _ = policies[name].act_inference(obs)
                        if not torch.isfinite(actions).all():
                            raise RuntimeError('Nonfinite policy action')
                        obs, _, _, _, _ = env.step(actions)
                        if not recorder.active.any():
                            break
                records = recorder.records(name, seed)
                all_records.extend(records)
                write_json(OUT / 'trials.json', all_records)
                write_json(OUT / 'pairing.json', pairing)
                success = sum(r['success'] for r in records)
                print(f'EVAL_RESULT model={name} seed={seed} success={success}/{len(records)} wall_seconds={time.monotonic()-began:.1f}', flush=True)
        print(f'EVAL_COMPLETE trials={len(all_records)} output={OUT}', flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
