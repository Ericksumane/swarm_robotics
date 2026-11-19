import json,sys
from pathlib import Path
import numpy as np
import torch

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swarm_core.formation_env import FormationEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,pack,hidden_inputs

torch.set_num_threads(1)
task=json.loads((root/'configs/tasks/formation_relational_random_evaluation.json').read_text())
comm=json.loads((root/'configs/communication/nominal.json').read_text())
output=root/'experiments/runs/retained_curriculum_001'
results=[]
for label,folder in [('baseline','baseline'),('legacy','legacy_checkpoint_0060000'),('retained','retained_checkpoint_0060000')]:
    run=output/folder
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    manifest=json.loads((run/'manifest.json').read_text())
    args,policy,_=make_policy(3,local_dim=9,critic_dim=23,neighbor_dim=9,geometry_summary=True)
    policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
    env=FormationEnv(task,comm,manifest['reward'])
    velocities=[];projections=[];linear_commands=[];yaw_commands=[];improved=0
    for seed in range(9000000,9000040):
        obs,_=env.reset(seed)
        states=env.simulation.states
        before=formation_metrics(states,1.,.25)['formation_rmse_m']
        h,m=hidden_inputs(3,args)
        with torch.no_grad():action,_=policy.act(pack(obs),h,m,deterministic=True)
        action=action.numpy()
        for i,own in enumerate(states):
            direction=np.zeros(2)
            for j,peer in enumerate(states):
                if i==j:continue
                delta=np.array([peer.x-own.x,peer.y-own.y]);distance=np.linalg.norm(delta)
                direction+=(distance-1.)*delta/max(distance,1e-8)
            projections.append(float(direction@np.array([np.cos(own.heading),np.sin(own.heading)])))
            linear_commands.append(float(action[i,0]));yaw_commands.append(float(action[i,1]))
        env.step(action)
        velocities.extend(s.speed for s in env.simulation.states)
        improved+=formation_metrics(env.simulation.states,1.,.25)['formation_rmse_m']<before
    result=dict(model=label,episodes=40,one_step_rmse_reduced=improved,
        linear_command_mean=float(np.mean(linear_commands)),linear_command_std=float(np.std(linear_commands)),
        yaw_command_mean=float(np.mean(yaw_commands)),yaw_command_std=float(np.std(yaw_commands)),
        initial_command_projection_correlation=float(np.corrcoef(linear_commands,projections)[0,1]),
        mean_initial_abs_speed_mps=float(np.mean(np.abs(velocities))))
    print(json.dumps(result),flush=True);results.append(result)
(output/'initial_action_response_audit.json').write_text(json.dumps(results,indent=2))
print('Initial-state diagnostic only; yaw can have delayed benefit. No target actions used for training.')
