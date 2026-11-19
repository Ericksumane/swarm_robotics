"""Evaluate three independent runtimes over an impaired peer-message network."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from swarm_core.dynamics import step
from swarm_core.initialization import sample_initial_states
from swarm_core.communication.network import Network
from swarm_core.communication.message import StateMessage
from swarm_core.deployment.protocol import PeerEnvelope
from swarm_core.deployment.local_runtime import LocalMissionRuntime
from swarm_core.learning.staged_policy import BatchedMissionPolicy
from swarm_core.mission_spec import completion_task
from swarm_core.geometry import formation_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);p.add_argument('--profiles',nargs='+',default=['nominal','good','moderate','severe']);p.add_argument('--episodes',type=int,default=4);p.add_argument('--seed',type=int,default=85000000);p.add_argument('--hold-s',type=float,default=60.);a=p.parse_args()
    if a.episodes<1 or a.hold_s<0:p.error('Invalid evaluation size.')
    torch.set_num_threads(1);path=Path(a.bundle);b=torch.load(path,map_location='cpu',weights_only=True);out=Path(a.output);out.mkdir(exist_ok=False)
    report=dict(scope='Independent local mission cores; Python physical plant; synthetic impaired messages; no ROS or hardware claim',bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=a.seed,hold_s=a.hold_s,profiles={})
    for profile in a.profiles:
        task=completion_task(b['task'],profile);comm=json.load(open('configs/communication/'+profile+'.json'));dt=task['simulation']['time_step_s'];rows=[]
        for seed in range(a.seed,a.seed+a.episodes):
            states=sample_initial_states(task,seed)
            def model():return BatchedMissionPolicy(b['translation'],b['formation'],alignment=b['alignment'],acquisition_routing=b['acquisition_routing'],parking=b.get('parking'),motion_gate=b.get('motion_gate'))
            nodes=[LocalMissionRuntime(task,i,'validation_'+str(seed),'boot_'+str(i),'mission',model()) for i in range(3)]
            network=Network(3,delay_s=comm['delay_s'],loss_probability=comm['loss_probability'],max_age_s=comm['max_information_age_s'],seed=seed,channel_options=comm.get('channel_options',{}))
            delivered={};minimum=1e9;starts=[None]*3;qualifying=0;first_success=None;max_hold_error=0.;reason='time_limit';frames=[]
            last_outputs=None
            for tick in range(math.ceil((task['simulation']['episode_time_limit_s']+a.hold_s+1.)/dt)):
                now=tick*dt;network.deliver(now)
                for i in range(3):
                    for sender,message in network.receivers[i]._latest.items():
                        if delivered.get((i,sender),-1)<message.sequence:
                            envelope=PeerEnvelope('validation_'+str(seed),'boot_'+str(sender),'mission',message)
                            nodes[i].accept_peer(envelope.to_dict(),now);delivered[i,sender]=message.sequence
                outputs=[node.tick(states[i],now,now) for i,node in enumerate(nodes)];last_outputs=outputs
                for output in outputs:network.broadcast(StateMessage(**output['peer']['state']),now)
                for i,node in enumerate(nodes):
                    if node.phase>=1 and starts[i] is None:starts[i]=now
                actions=[];limits=task['robot_defaults']
                for o in outputs:
                    v,w=o['command'];scale=limits['max_forward_speed_mps'] if v>=0 else limits['max_reverse_speed_mps'];actions.append([v/scale,w/limits['max_yaw_rate_radps']])
                states=[step(state,action,limits,dt) for state,action in zip(states,actions)]
                geo=formation_metrics(states,1.,limits['footprint_radius_m']);minimum=min(minimum,geo['minimum_clearance_m'])
                errors=[None if n.goal is None else float(np.linalg.norm(n.goal-[s.x,s.y])) for n,s in zip(nodes,states)]
                stopped=all(abs(s.speed)<=.03 and abs(s.yaw_rate)<=.05 for s in states)
                complete=all(e is not None and e<=.2 for e in errors) and stopped and geo['max_pair_error_m']<=task['mission']['formation_tolerance_m']
                qualifying=qualifying+1 if complete else 0
                if qualifying*dt>=2.-1e-9 and first_success is None:first_success=now+dt
                if first_success is not None:max_hold_error=max(max_hold_error,max(errors))
                frames.append(dict(time_s=now+dt,states=[vars(s) for s in states],phases=[n.phase for n in nodes],modes=[o['mode'] for o in outputs],goal_errors_m=errors,pair_error_m=geo['max_pair_error_m']))
                if geo['collision']:reason='collision';break
                area=task['operating_area'];r=limits['footprint_radius_m']
                if any(not(area['x_bounds_m'][0]+r<=s.x<=area['x_bounds_m'][1]-r and area['y_bounds_m'][0]+r<=s.y<=area['y_bounds_m'][1]-r) for s in states):reason='outside_area';break
                if any(o['fault'] for o in outputs):reason='runtime_fault';break
                if first_success is not None and now+dt-first_success>=a.hold_s-1e-8:
                    reason='success' if complete and max_hold_error<=.2 else 'hold_failure';break
            row=dict(seed=seed,reason=reason,time_s=now+dt,first_success_s=first_success,minimum_clearance_m=minimum,start_s=starts,start_skew_s=max(starts)-min(starts) if all(s is not None for s in starts) else None,final_goal_errors_m=errors,max_hold_goal_error_m=max_hold_error if first_success else None,faults=[o['fault'] for o in last_outputs])
            rows.append(row);(out/(profile+'_'+str(seed)+'.json')).write_text(json.dumps(frames));print(json.dumps(dict(profile=profile,**row)),flush=True)
            report['profiles'][profile]=rows;(out/'evaluation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
