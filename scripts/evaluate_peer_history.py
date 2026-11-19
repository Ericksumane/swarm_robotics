"""Matched development comparison of acquisition only; no mission claim."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-directory');p.add_argument('--profiles',nargs='+',default=['nominal','good','moderate','severe']);p.add_argument('--seed',type=int,default=83000000);p.add_argument('--episodes',type=int,default=8);p.add_argument('--checkpoint',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    if a.episodes<1: p.error('Positive episode count required.')
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];out=Path(a.output_directory) if a.output_directory else root/'experiments/runs/bounded_history_001'
    out.mkdir(parents=True,exist_ok=True)
    item=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    history=item['contract'] in ('formation_history_v5','formation_history_risk_v6')
    if item['contract'] not in ('formation_history_risk_v6','formation_history_v5','formation_relational_v4'):raise ValueError('Unsupported contract.')
    task=json.load(open('configs/tasks/dispersed_history_evaluation.json'))
    if not history:
        task['observation_contract']='formation_relational_v4'
        task.pop('peer_history',None)
    if item['contract']=='formation_history_risk_v6':
        task['observation_contract']=item['contract']
        task['peer_history']=json.loads((Path(a.checkpoint).parent/'manifest.json').read_text())['task']['peer_history']
    task['initialization'].pop('near_formation',None)
    reward=json.load(open('configs/training/dispersed_acquisition_reward.json'))
    args,policy,_=make_policy(3,local_dim=9,neighbor_dim=item['neighbor_dim'],critic_dim=item['critic_dim'],geometry_summary=True)
    policy.actor.load_state_dict(item['actor']);policy.actor.eval();h,m=hidden_inputs(3,args)
    result=dict(role='development',checkpoint=a.checkpoint,seed=a.seed,episodes=a.episodes,scope='Acquisition only in Python dynamics; no distributed readiness, translation, Gazebo or hardware validation.',profiles={})
    for profile in a.profiles:
        comm=json.load(open('configs/communication/'+profile+'.json'));env=FormationEnv(task,comm,reward);rows=[]
        for seed in range(a.seed,a.seed+a.episodes):
            obs,_=env.reset(seed);operational_steps=0;operational_time=None;minimum_clearance=float('inf');available=0;fresh=0;reads=0
            while True:
                with torch.no_grad():action,_=policy.act(pack(obs),h,m,deterministic=True)
                obs,_,term,trunc,info=env.step(action.numpy())
                geometry=formation_metrics(env.simulation.states,1.,.25)
                minimum_clearance=min(minimum_clearance,geometry['minimum_clearance_m'])
                stopped=all(abs(s.speed)<=.03 and abs(s.yaw_rate)<=.05 for s in env.simulation.states)
                operational_steps=operational_steps+1 if stopped and geometry['max_pair_error_m']<=.2 else 0
                if operational_steps>=20 and operational_time is None:operational_time=info['time_s']
                for o in obs:
                    available+=int(o['neighbor_mask'].sum());fresh+=int(o.get('neighbor_fresh_mask',o['neighbor_mask']).sum());reads+=len(o['neighbor_mask'])
                if term or trunc:break
            rows.append(dict(seed=seed,reason=info['reason'],elapsed_s=info['time_s'],operational_formation_time_s=operational_time,final_pair_error_m=geometry['max_pair_error_m'],minimum_clearance_m=minimum_clearance,estimate_availability=available/reads,fresh_read_fraction=fresh/reads))
        summary=dict(strict_successes=sum(r['reason']=='success' for r in rows),operational_arrivals=sum(r['operational_formation_time_s'] is not None for r in rows),operational_safe_episodes=sum(r['operational_formation_time_s'] is not None and r['reason'] not in ('collision','outside_area') for r in rows),collisions=sum(r['reason']=='collision' for r in rows),results=rows)
        result['profiles'][profile]=summary
        (out/(a.label+'.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(label=a.label,profile=profile,**{k:v for k,v in summary.items() if k!='results'})),flush=True)

if __name__=='__main__':main()
