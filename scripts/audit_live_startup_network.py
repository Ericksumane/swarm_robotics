"""Reproduce original logged expiry masks and compare numerical correction."""
from pathlib import Path
import json,math
import numpy as np
from swarm_core.communication.network import Network
from swarm_core.communication.message import StateMessage
root=Path(__file__).resolve().parents[1];dest=root/'experiments/runs/live_communication_diagnosis_001';dest.mkdir(exist_ok=True)
runs={'nominal':'gazebo_live_20261103_205702_151054','moderate':'gazebo_live_20261103_210039_152265','severe':'gazebo_live_20261103_210551_154136'};results={}
for profile,name in runs.items():
 path=root/'experiments/runs'/name;manifest=json.loads((path/'manifest.json').read_text());rows=[json.loads(line) for line in (path/'control.jsonl').read_text().splitlines()];comm=manifest['hello']['communication'];seed=manifest['settings']['seed'];network=Network(3,delay_s=comm['delay_s'],loss_probability=comm['loss_probability'],max_age_s=comm['max_information_age_s'],seed=seed,channel_options=comm.get('channel_options',{}))
 def transmit(states,t,sequence):
  for i,s in enumerate(states):network.broadcast(StateMessage(sender_id=i,sequence=sequence,generated_at_s=t,x=s['x'],y=s['y'],heading=s['heading'],speed=s['speed'],yaw_rate=s['yaw_rate'],payload_size_bytes=comm['payload_size_bytes']),t)
 transmit(manifest['initial_physical_states'],0.,0);network.deliver(0.)
 corrected_valid_count=0;total=valid_count=boundary_false=expired=missing=0;all_empty=0;longest_empty=current_empty=0;fresh_run=[0]*3;max_fresh_run=[0]*3;physical_ready_run=max_physical_ready=0;alignment_heading=[];alignment_speed=[];transitions=[0]*3;previous=None;matched=True
 for k,row in enumerate(rows,1):
  t=k*.1;transmit(row['states'],t,k);network.deliver(t);valids=[]
  for receiver in range(3):
   n=0
   for sender in range(3):
    if receiver==sender:continue
    message,age,corrected=network.read(receiver,sender,t);corrected_valid_count+=int(corrected);valid=message is not None and age<=comm['max_information_age_s'];total+=1;n+=int(valid);valid_count+=int(valid)
    if message is None:missing+=1
    elif not valid:
     expired+=1
     if math.isclose(age,comm['max_information_age_s'],rel_tol=0.,abs_tol=1e-9):boundary_false+=1
   valids.append(n);fresh_run[receiver]=fresh_run[receiver]+1 if n==2 else 0;max_fresh_run[receiver]=max(max_fresh_run[receiver],fresh_run[receiver])
  matched=matched and valids==row['policy']['valid_neighbors']
  if sum(valids)==0:all_empty+=1;current_empty+=1;longest_empty=max(longest_empty,current_empty)
  else:current_empty=0
  angles=[abs(math.atan2(math.sin(s['heading']),math.cos(s['heading']))) for s in row['states']]
  stopped=all(abs(s['speed'])<=.03 and abs(s['yaw_rate'])<=.05 for s in row['states'])
  qualified=stopped and max(angles)<=.05 and row['policy']['max_pair_error_m']<=.1
  physical_ready_run=physical_ready_run+1 if qualified else 0;max_physical_ready=max(max_physical_ready,physical_ready_run)
  modes=row['policy']['modes']
  for i,mode in enumerate(modes):
   if mode=='alignment':alignment_heading.append(angles[i]);alignment_speed.append(abs(row['states'][i]['speed']))
   if previous is not None and mode!=previous[i]:transitions[i]+=1
  previous=modes
 result=dict(run=name,rows=len(rows),recomputed_masks_match_log=matched,valid_link_fraction=valid_count/total,corrected_valid_link_fraction=corrected_valid_count/total,all_robots_no_valid_neighbors_fraction=all_empty/len(rows),longest_all_empty_s=longest_empty*.1,max_consecutive_both_neighbors_fresh_s=[v*.1 for v in max_fresh_run],expired_reads=expired,missing_reads=missing,false_expired_at_exact_age_boundary=boundary_false,max_true_physical_startup_ready_s=max_physical_ready*.1,alignment_mean_abs_heading_error_rad=float(np.mean(alignment_heading)) if alignment_heading else None,alignment_max_abs_heading_error_rad=max(alignment_heading) if alignment_heading else None,alignment_mean_abs_speed_mps=float(np.mean(alignment_speed)) if alignment_speed else None,mode_transitions_per_robot=transitions,first_move_s=next((r['elapsed_s'] for r in rows if any(p>=1 for p in r['policy']['phases'])),None))
 results[profile]=result;print(profile,json.dumps(result),flush=True)
(dest/'diagnosis.json').write_text(json.dumps(results,indent=2))
