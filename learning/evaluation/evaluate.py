"""Deterministic, seed-separated evaluation; export actor-only artifact."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.mission_env import MissionEnv
from swarm_core.alignment_env import AlignmentEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import CONTRACT, pack, make_policy, hidden_inputs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--episodes',type=int,default=5)
    parser.add_argument('--seed',type=int,default=100000)
    parser.add_argument('--task')
    parser.add_argument('--communication')
    parser.add_argument('--mode', choices=['policy','stationary'], default='policy')
    cli=parser.parse_args()
    if cli.episodes < 1: parser.error('Episodes must be positive.')
    torch.set_num_threads(1)
    run=Path(cli.run)
    manifest=json.loads((run/'manifest.json').read_text())
    if cli.task: manifest['task']=json.loads(Path(cli.task).read_text())
    # Load only locally produced, trusted checkpoint files.
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    if checkpoint['contract'] != manifest.get('contract', CONTRACT): raise ValueError('Contract mismatch.')
    if checkpoint['contract'] != manifest['task'].get('observation_contract',CONTRACT):
        raise ValueError('Evaluation task observation contract mismatch.')
    if checkpoint['num_agents'] != manifest['task']['team']['num_agents']:
        raise ValueError('Evaluation team size differs from trained checkpoint.')
    args,policy,_=make_policy(checkpoint['num_agents'],local_dim=checkpoint.get('local_dim',9),critic_dim=checkpoint.get('critic_dim'),neighbor_dim=checkpoint.get('neighbor_dim',7),geometry_summary=checkpoint.get('geometry_summary',False),hold_std_scale=checkpoint.get('hold_std_scale'),local_only=checkpoint.get('local_only',False))
    policy.actor.load_state_dict(checkpoint['actor'])
    policy.actor.eval()
    comm=manifest['communication']
    if cli.communication: comm=json.loads(Path(cli.communication).read_text())
    env={'mission':MissionEnv,'alignment':AlignmentEnv}.get(manifest['task'].get('environment'),FormationEnv)(manifest['task'],comm,manifest['reward'])
    results=[]
    for episode in range(cli.episodes):
        obs,_=env.reset(cli.seed+episode)
        h,m=hidden_inputs(len(obs),args)
        initial_geometry=formation_metrics(env.simulation.states,manifest['task']['team']['desired_pair_distance_m'],manifest['task']['robot_defaults']['footprint_radius_m'])
        initial_translation=None
        if hasattr(env,'starts') and all(p is not None for p in env.starts):
            initial_translation=float(np.dot(initial_geometry['centroid_m']-np.mean(env.starts,axis=0),env.direction))
        total=0.; errors=[]; pair_errors=[]; clearance=float('inf'); trajectory=[]
        while True:
            with torch.no_grad(): action,_=policy.act(pack(obs),h,m,deterministic=True)
            if cli.mode=='stationary': action=torch.zeros_like(action)
            obs,reward,term,trunc,info=env.step(action.numpy())
            total+=float(reward[0])
            geometry=formation_metrics(env.simulation.states,manifest['task']['team']['desired_pair_distance_m'],manifest['task']['robot_defaults']['footprint_radius_m'])
            errors.append(geometry['formation_rmse_m'])
            pair_errors.append(geometry['max_pair_error_m'])
            clearance=min(clearance,geometry['minimum_clearance_m'])
            if episode==0:
                trajectory.append([[s.x,s.y,s.heading] for s in env.simulation.states])
            if term or trunc: break
        result=dict(initial_translation_m=initial_translation,episode_started_near_goal=bool(manifest['task'].get('approach_reset_bank')),diagnostics=info.get('diagnostics'),phases=info.get('phases'),translation_m=info.get('translation_m'),goal_errors_m=info.get('goal_errors_m'),seed=cli.seed+episode,reason=info['reason'],return_sum=total,initial_rmse_m=initial_geometry['formation_rmse_m'],
            mean_rmse_m=float(np.mean(errors)),final_rmse_m=errors[-1],minimum_clearance_m=clearance,
            elapsed_time_s=info['time_s'],max_pair_error_m=float(max(pair_errors)),
            formation_compliant_fraction=float(np.mean(np.asarray(pair_errors)<=manifest['task']['mission']['formation_tolerance_m'])),
            final_max_speed_mps=max(abs(s.speed) for s in env.simulation.states),
            final_max_yaw_rate_radps=max(abs(s.yaw_rate) for s in env.simulation.states),
            strict_formation_success=bool(info['reason']=='success' and max(pair_errors)<=manifest['task']['mission']['formation_tolerance_m']))
        print(json.dumps(result),flush=True); results.append(result)
        if episode==0:
            (run/f'trajectory_{cli.mode}_{comm["profile_name"]}_{cli.seed}.json').write_text(json.dumps(trajectory))
            # Preserve task provenance when the same seeds/profile test multiple resets.
            (run/f'trajectory_{cli.mode}_{manifest["task"]["task_name"]}_{comm["profile_name"]}_{cli.seed}.json').write_text(json.dumps(trajectory))
            (run/'trajectory.json').write_text(json.dumps(trajectory))
    summary=dict(contract=checkpoint["contract"],mode=cli.mode,task=manifest['task'],communication=comm,results=results,
        success_rate=sum(r['reason']=='success' for r in results)/len(results))
    report_name=f"evaluation_{cli.mode}_{manifest['task']['task_name']}_{comm['profile_name']}_{cli.seed}_{cli.episodes}.json"
    (run/report_name).write_text(json.dumps(summary,indent=2))
    (run/'evaluation.json').write_text(json.dumps(summary,indent=2))
    torch.save(dict(contract=checkpoint['contract'],local_only=checkpoint.get('local_only',False),hold_std_scale=checkpoint.get('hold_std_scale'),local_dim=checkpoint.get('local_dim',9),critic_dim=checkpoint.get('critic_dim'),neighbor_dim=checkpoint.get('neighbor_dim',7),num_agents=checkpoint['num_agents'],geometry_summary=checkpoint.get('geometry_summary',False),actor=checkpoint['actor']),run/'actor.pt')
    print('Success rate:',summary['success_rate'])

if __name__=='__main__': main()
