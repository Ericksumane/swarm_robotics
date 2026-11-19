"""Full experimental mission evaluation, with traceable local protocol state."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.staged_policy import BatchedMissionPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--episodes',type=int,default=1);p.add_argument('--profiles',nargs='+',default=['nominal','good','moderate','severe']);p.add_argument('--time-limit',type=float,default=None);a=p.parse_args()
    if a.episodes<1 or (a.time_limit is not None and (not np.isfinite(a.time_limit) or a.time_limit<=0)):p.error('Invalid evaluation size.')
    torch.set_num_threads(1);out=Path(a.output);out.mkdir(exist_ok=False)
    bundle=torch.load(a.bundle,map_location='cpu',weights_only=True)
    result=dict(role='development',bundle=a.bundle,bundle_sha256=hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest(),scope='Python dynamics, complete dispersed FORM/ALIGN/MOVE/HOLD mission; no pose replacement.',profiles={})
    for profile in a.profiles:
        from swarm_core.mission_spec import completion_task
        task=completion_task(bundle['task'],profile)
        if a.time_limit is not None:task['simulation']['episode_time_limit_s']=a.time_limit
        comm=json.load(open('configs/communication/'+profile+'.json'))
        env=MissionEnv(task,comm,json.load(open('configs/training/geometry_control.json')))
        policy=BatchedMissionPolicy(bundle['translation'],bundle['formation'],alignment=bundle['alignment'],acquisition_routing=bundle['acquisition_routing'],parking=bundle.get('parking'),motion_gate=bundle.get('motion_gate'));rows=[]
        for seed in range(a.seed,a.seed+a.episodes):
            obs,_=env.reset(seed);policy.reset();minimum_clearance=1e9;first_move=[None]*3;max_post_error=0.;trace=[]
            initial=[vars(s) for s in env.simulation.states]
            while True:
                actions,modes=policy.act(obs)
                obs,_,term,trunc,info=env.step(actions)
                geometry=formation_metrics(env.simulation.states,1.,.25);minimum_clearance=min(minimum_clearance,geometry['minimum_clearance_m'])
                for i,phase in enumerate(env.phases):
                    if phase>=1 and first_move[i] is None:first_move[i]=info['time_s']
                if any(x is not None for x in first_move):max_post_error=max(max_post_error,geometry['max_pair_error_m'])
                trace.append(dict(time_s=info['time_s'],states=[vars(s) for s in env.simulation.states],actions=actions.tolist(),modes=modes,phases=env.phases.tolist(),pair_error_m=geometry['max_pair_error_m'],minimum_clearance_m=geometry['minimum_clearance_m'],ready=env.simulation.formation_ready.copy(),proposals=env.simulation.start_proposals.copy(),committed=env.simulation.start_committed.copy(),evidence_steps=env.ready_steps.tolist(),history_available=[int(o.get('formation_history',o)['neighbor_mask'].sum()) for o in obs]))
                if term or trunc:break
            row=dict(seed=seed,reason=info['reason'],elapsed_s=info['time_s'],initial_states=initial,first_move_s=first_move,start_skew_s=max(first_move)-min(first_move) if all(x is not None for x in first_move) else None,goal_errors_m=info['goal_errors_m'],final_pair_error_m=geometry['max_pair_error_m'],max_post_move_pair_error_m=max_post_error if any(x is not None for x in first_move) else None,minimum_clearance_m=minimum_clearance)
            rows.append(row)
            (out/(profile+'_'+str(seed)+'.json')).write_text(json.dumps(trace))
            print(json.dumps(dict(profile=profile,**row)),flush=True)
        result['profiles'][profile]=dict(task=task,communication=comm,results=rows,successes=sum(r['reason']=='success' for r in rows),collisions=sum(r['reason']=='collision' for r in rows),started_all=sum(all(x is not None for x in r['first_move_s']) for r in rows))
        (out/'evaluation.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
