"""Train the formation-hold stage using upstream MAPPO PPO updates."""
import argparse
from collections import Counter
import hashlib
import shutil
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.mission_env import MissionEnv
from swarm_core.alignment_env import AlignmentEnv
from swarm_core.curriculum import HeadingCurriculum
from swarm_core.distance_curriculum import DistanceCurriculum
from swarm_core.learning.mappo_adapter import (
    CONTRACT, pack, make_policy, critic_input, hidden_inputs, gae)

def load(path):
    return json.loads(Path(path).read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-lr', type=float)
    parser.add_argument('--communication-mixture')
    parser.add_argument('--snapshot-interval', type=int, default=0)
    parser.add_argument('--wall-seconds', type=float, default=None)
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--communication', default='configs/communication/nominal.json')
    parser.add_argument('--task', default='configs/tasks/formation_v1.json')
    parser.add_argument('--reward', default='configs/training/formation_reward_v1.json')
    parser.add_argument('--final-std', type=float)
    parser.add_argument('--std-decay-steps', type=int, default=20000)
    parser.add_argument('--initial-std', type=float, default=1.0)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--initial-actor')
    parser.add_argument('--sampling-seed', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--output', required=True)
    cli = parser.parse_args()
    if min(cli.steps, cli.rollout, cli.epochs) < 1 or not 0 < cli.gamma <= 1 or not 0 <= cli.gae_lambda <= 1:
        parser.error('Invalid training settings.')
    if cli.wall_seconds is not None and (not np.isfinite(cli.wall_seconds) or cli.wall_seconds <= 0):
        parser.error('Wall-clock limit must be positive and finite.')
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(cli.seed)
    rng = np.random.default_rng(cli.seed)
    task = load(cli.task)
    comm = load(cli.communication)
    weights = load(cli.reward)
    if cli.curriculum and task.get('training_curriculum'):
        parser.error('Choose time-based or mastery-based curriculum, not both.')
    if cli.resume and task.get('training_curriculum'):
        parser.error('Mastery curriculum resume requires saved curriculum state; start a fresh run.')
    env_class={"mission":MissionEnv,"alignment":AlignmentEnv}.get(task.get("environment"),FormationEnv)
    mixture=load(cli.communication_mixture) if cli.communication_mixture else None
    if mixture and (cli.curriculum or task.get('training_curriculum') or task.get('mission_distance_curriculum')):
        parser.error('Communication mixture requires a fixed task curriculum.')
    if mixture:
        from swarm_core.communication.profile_sampler import ProfileSampler
        sampler=ProfileSampler(mixture)
        comm=sampler.sample(cli.seed)
    env = env_class(task, comm, weights)
    local_dim = 16 if task.get("environment") in ("mission","alignment") else 9
    critic_dim = task["team"]["num_agents"] * (12 if local_dim==16 else 5) + 2
    contract = task.get("observation_contract", CONTRACT)
    neighbor_dim = 9 if contract in ("formation_geometry_v2","formation_geometry_critic_v3","formation_relational_v4","mission_relational_v2","mission_local_goal_v3","mission_local_goal_time_aligned_v4") else 7
    if contract == 'formation_history_v5': neighbor_dim = 11
    if contract == 'formation_history_risk_v6': neighbor_dim = 13
    if contract in ('formation_geometry_critic_v3','formation_relational_v4','formation_history_v5','formation_history_risk_v6'):
        count=task['team']['num_agents']
        critic_dim=6*count+count*(count-1)//2+2
    if task.get('critic_contract') == 'mission_relative_v1':
        critic_dim=11*task['team']['num_agents']+task['team']['num_agents']*(task['team']['num_agents']-1)//2+3
    hold_std_scale=task.get('exploration',{}).get('hold_std_scale')
    args, policy, trainer = make_policy(task['team']['num_agents'], cli.device, local_dim=local_dim, critic_dim=critic_dim, neighbor_dim=neighbor_dim, geometry_summary=contract in ('formation_history_risk_v6','formation_history_v5','formation_relational_v4','mission_relational_v2','mission_local_goal_v3','mission_local_goal_time_aligned_v4'),hold_std_scale=hold_std_scale,local_only=task.get('actor_local_only',False))
    if not np.isfinite(cli.initial_std) or cli.initial_std <= 0 or not np.isfinite(cli.entropy_coef) or cli.entropy_coef < 0:
        parser.error('Invalid exploration parameters.')
    args.entropy_coef = cli.entropy_coef
    trainer.entropy_coef = cli.entropy_coef
    if cli.final_std is not None and (not np.isfinite(cli.final_std) or cli.final_std <= 0 or cli.std_decay_steps <= 0):
        parser.error('Invalid exploration schedule.')
    if cli.final_std is not None:
        policy.actor.act.action_out.logstd._bias.requires_grad_(False)
    if cli.initial_actor and cli.resume:
        parser.error('Choose initial actor or complete checkpoint resume.')
    if cli.initial_actor:
        initial = torch.load(cli.initial_actor, map_location=cli.device, weights_only=True)
        policy.actor.load_state_dict(initial['actor'])
    # Actor initialization must not override the requested exploration scale.
    # Full checkpoint resume retains its saved scale unless a schedule is selected.
    if not cli.resume:
        with torch.no_grad():
            policy.actor.act.action_out.logstd._bias.fill_(float(np.log(cli.initial_std)))
    if cli.sampling_seed is not None:
        torch.manual_seed(cli.sampling_seed)
    if cli.resume:
        checkpoint = torch.load(cli.resume, map_location=cli.device, weights_only=True)
        if checkpoint['contract'] != contract:
            raise ValueError('Checkpoint observation contract mismatch.')
        if checkpoint.get('local_only',False)!=task.get('actor_local_only',False):raise ValueError('Checkpoint actor information architecture mismatch; use explicit actor initialization for adaptation.')
        policy.actor.load_state_dict(checkpoint['actor'])
        policy.critic.load_state_dict(checkpoint['critic'])
        policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    if cli.actor_lr is not None:
        if not np.isfinite(cli.actor_lr) or cli.actor_lr <= 0:
            parser.error('Actor learning rate must be positive and finite.')
        for group in policy.actor_optimizer.param_groups:
            group['lr'] = cli.actor_lr
    manifest = dict(settings=vars(cli), local_only=task.get('actor_local_only',False), hold_std_scale=hold_std_scale, task=task, communication=comm,
        reward=weights, contract=contract, local_dim=local_dim, critic_dim=critic_dim, neighbor_dim=neighbor_dim, geometry_summary=contract in ('formation_history_risk_v6','formation_history_v5','formation_relational_v4','mission_relational_v2','mission_local_goal_v3','mission_local_goal_time_aligned_v4'), torch=torch.__version__,
        upstream_commit=subprocess.check_output(['git','-C','external/on-policy','rev-parse','HEAD'], text=True).strip(),
        project_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()))
    manifest['communication_mixture'] = mixture
    manifest['source_sha256'] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('src/swarm_core', 'scripts', 'learning')
        for path in sorted(Path(folder).rglob('*.py'))
    }
    manifest['resume_checkpoint_sha256'] = hashlib.sha256(Path(cli.resume).read_bytes()).hexdigest() if cli.resume else None
    manifest['resume_source_step'] = checkpoint.get('step',0) if cli.resume else 0
    manifest['initial_actor_sha256'] = hashlib.sha256(Path(cli.initial_actor).read_bytes()).hexdigest() if cli.initial_actor else None
    manifest['starting_action_std'] = policy.actor.act.action_out.logstd._bias.detach().exp().cpu().reshape(-1).tolist()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    curriculum_start = dict(task['initialization'].get('near_formation', {}))
    episode = 0
    outcomes = Counter()
    mastery=HeadingCurriculum(task['training_curriculum'],seed=cli.seed+11003) if task.get('training_curriculum') else None
    if mastery: mastery.apply(env.simulation)
    distance_curriculum=DistanceCurriculum(task['mission_distance_curriculum'],cli.seed+21101) if task.get('mission_distance_curriculum') else None
    if distance_curriculum: distance_curriculum.apply(env)
    obs, _ = env.reset(cli.seed)
    n = len(obs)
    hidden, masks = hidden_inputs(n, args)
    encoder_before = [p.detach().clone() for p in policy.actor.base.neighbors.parameters()]
    head_before = [p.detach().clone() for p in policy.actor.base.head.parameters()]
    completed = 0
    next_snapshot = cli.snapshot_interval
    started = time.monotonic()
    deadline = started + cli.wall_seconds if cli.wall_seconds else float("inf")
    clipped_samples = 0
    action_components = 0
    while completed < cli.steps and time.monotonic() < deadline:
        if cli.final_std is not None:
            fraction=min(1.,completed/cli.std_decay_steps)
            scheduled_std=cli.initial_std+(cli.final_std-cli.initial_std)*fraction
            with torch.no_grad():
                policy.actor.act.action_out.logstd._bias.fill_(float(np.log(scheduled_std)))
        rows = []
        trainer.prep_rollout()
        for _ in range(min(cli.rollout, cli.steps-completed)):
            if time.monotonic() >= deadline and rows:
                break
            raw, central = pack(obs), critic_input(env)
            with torch.no_grad():
                value, action, logp, _, _ = policy.get_actions(central, raw, hidden, hidden, masks)
            raw_action = action.cpu().numpy()
            clipped_samples += int((np.abs(raw_action)>1).sum())
            action_components += raw_action.size
            next_obs, reward, term, trunc, info = env.step(raw_action)
            with torch.no_grad():
                nv = policy.get_values(critic_input(env), hidden, masks).cpu().numpy()
            rows.append((central, raw, action.cpu().numpy(), value.cpu().numpy(),
                logp.cpu().numpy(), reward[:,None], nv, float(term), float(term or trunc)))
            completed += 1
            obs = next_obs
            if term or trunc:
                outcomes[info['reason']] += 1
                with (out/'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=completed, episode=episode,
                        communication_profile=env.simulation.communication.get('profile_name'), curriculum_episode=mastery.status() if mastery else None, distance_curriculum=distance_curriculum.status() if distance_curriculum else None, mission_distance_m=env.task['mission']['translation_distance_m'], **info))+'\n')
                episode += 1
                if mastery:
                    mastery.record(info['reason']=='success')
                    mastery.apply(env.simulation)
                if distance_curriculum:
                    distance_curriculum.record(info['reason']=='success')
                    distance_curriculum.apply(env)
                if cli.curriculum:
                    elapsed_fraction = (time.monotonic()-started) / cli.wall_seconds if cli.wall_seconds else completed/cli.steps
                    fraction = min(1.,elapsed_fraction)
                    curriculum = env.simulation.task['initialization'].setdefault('near_formation', {})
                    start_probability = curriculum_start.get('probability', 0.95)
                    start_jitter = curriculum_start.get('position_jitter_m', 0.05)
                    curriculum['probability'] = start_probability + (min(0.30, start_probability)-start_probability)*fraction
                    curriculum['position_jitter_m'] = start_jitter + (max(0.25, start_jitter)-start_jitter)*fraction
                if mixture:
                    env=env_class(task,sampler.sample(cli.seed+episode),weights)
                obs, _ = env.reset(cli.seed + episode)
        arrays = [np.asarray(x) for x in zip(*rows)]
        central, raw, actions, values, logps, rewards, nv, term, end = arrays
        advantages, returns = gae(rewards, values, nv, term[:,None,None],
                                  end[:,None,None], cli.gamma, cli.gae_lambda)
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-5)
        flatten = lambda x: x.reshape((-1, x.shape[-1]))
        central, raw, actions, values, logps, returns, advantages = map(flatten,
            (central, raw, actions, values, logps, returns, advantages))
        trainer.prep_training()
        losses = []
        ppo_diagnostics = []
        for _ in range(cli.epochs):
            for idx in np.array_split(rng.permutation(len(raw)), min(4,len(raw))):
                h, m = hidden_inputs(len(idx), args)
                sample = (central[idx], raw[idx], h, h.copy(), actions[idx],
                    values[idx], returns[idx], m, m.copy(), logps[idx], advantages[idx], None)
                result = trainer.ppo_update(sample)
                if not all(torch.isfinite(torch.as_tensor(x)).all() for x in result):
                    raise RuntimeError('Nonfinite PPO update.')
                losses.append(float(result[0].detach()))
                ratios = result[5].detach().clamp_min(1e-12)
                ppo_diagnostics.append([
                    float(result[2].detach()), float(result[4]), float(result[1]),
                    float(((ratios - 1.) - ratios.log()).mean()),
                    float(((ratios - 1.).abs() > trainer.clip_param).float().mean()),
                ])
        record = dict(step=completed, value_loss=float(np.mean(losses)), episodes=episode,outcomes=dict(outcomes),
            elapsed_s=time.monotonic()-started,steps_per_second=completed/max(time.monotonic()-started,1e-9),
            clipped_action_fraction=clipped_samples/max(action_components,1),
            action_std=policy.actor.act.action_out.logstd._bias.detach().exp().cpu().reshape(-1).tolist(),
            curriculum=env.simulation.task['initialization'].get('near_formation'),
            mastery_curriculum=mastery.status() if mastery else None, distance_curriculum=distance_curriculum.status() if distance_curriculum else None)
        diagnostics = np.mean(ppo_diagnostics, axis=0)
        record.update(dict(zip(
            ('policy_loss', 'actor_grad_norm', 'critic_grad_norm',
             'approx_kl', 'ppo_clip_fraction'), diagnostics.tolist())))
        with (out/'updates.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        torch.save(dict(contract=contract, local_only=task.get('actor_local_only',False), hold_std_scale=hold_std_scale, local_dim=local_dim, critic_dim=critic_dim, neighbor_dim=neighbor_dim, geometry_summary=contract in ('formation_history_risk_v6','formation_history_v5','formation_relational_v4','mission_relational_v2','mission_local_goal_v3','mission_local_goal_time_aligned_v4'), num_agents=n, args=vars(args),
            actor=policy.actor.state_dict(), critic=policy.critic.state_dict(),
            actor_optimizer=policy.actor_optimizer.state_dict(),
            critic_optimizer=policy.critic_optimizer.state_dict(), step=completed), out/'checkpoint.tmp')
        (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
        if cli.snapshot_interval > 0 and completed >= next_snapshot:
            shutil.copy2(out/'checkpoint.pt', out/f'checkpoint_{completed:07d}.pt')
            next_snapshot += cli.snapshot_interval
    if task.get('actor_local_only',False):assert any(not torch.equal(a,b) for a,b in zip(head_before,policy.actor.base.head.parameters()))
    else:assert any(not torch.equal(a,b) for a,b in zip(encoder_before,policy.actor.base.neighbors.parameters()))
    summary=dict(completed_steps=completed,elapsed_s=time.monotonic()-started,
        episodes=episode,outcomes=dict(outcomes),mastery_curriculum=mastery.status() if mastery else None,distance_curriculum=distance_curriculum.status() if distance_curriculum else None,stop_reason='wall_limit' if time.monotonic()>=deadline else 'step_limit')
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print('Local-only actor optimization: PASSED' if task.get('actor_local_only',False) else 'Joint encoder optimization: PASSED', flush=True)
    print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    main()
