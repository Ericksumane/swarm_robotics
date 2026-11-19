"""Audit continuous terminal hold in recorded local-core and Gazebo traces."""
import argparse,json
from pathlib import Path
import torch
from swarm_core.mission_spec import completion_task

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    base=Path(a.root);b=torch.load(base/'candidate_runtime_v5.pt',map_location='cpu',weights_only=True);result={}
    for name in ['runtime_v5_validation','runtime_v5_independent']:
        path=base/name/'evaluation.json'
        if not path.exists():continue
        report=json.loads(path.read_text());rows=[]
        for profile,runs in report['profiles'].items():
            tol=completion_task(b['task'],profile)['mission']['formation_tolerance_m']
            for run in runs:
                frames=json.loads((base/name/(profile+'_'+str(run['seed'])+'.json')).read_text())
                held=[f for f in frames if run['first_success_s'] is not None and f['time_s']>=run['first_success_s']-1e-8]
                violations=sum(any(e is None or e>.2 for e in f['goal_errors_m']) or f['pair_error_m']>tol or any(abs(s['speed'])>.03 or abs(s['yaw_rate'])>.05 for s in f['states']) for f in held)
                rows.append(dict(profile=profile,seed=run['seed'],reason=run['reason'],samples=len(held),violations=violations))
        result[name]=rows
    rows=[]
    for profile in ['nominal','good','moderate','severe']:
        dest=base/('gazebo_v5_hold_'+profile);summary=json.loads((dest/'summary.json').read_text());start=summary['final_policy']['first_success_s'];tol=completion_task(b['task'],profile)['mission']['formation_tolerance_m'];held=[];minimum=1e9
        for line in (dest/'control.jsonl').open():
            f=json.loads(line);policy=f['policy'];minimum=min(minimum,policy['minimum_clearance_m'])
            if start is not None and policy['time_s']>=start-1e-8:held.append(f)
        violations=sum(any(e is None or e>.2 for e in f['policy']['goal_errors_m']) or f['policy']['max_pair_error_m']>tol or any(abs(s['speed'])>.03 or abs(s['yaw_rate'])>.05 for s in f['states']) for f in held)
        rows.append(dict(profile=profile,first_success_s=start,observed_until_s=summary['simulated_s'],hold_duration_s=summary['simulated_s']-start if start is not None else None,hold_samples=len(held),hold_violations=violations,minimum_clearance_m=minimum,final_goal_errors_m=summary['final_policy']['goal_errors_m']))
    result['gazebo']=rows;(base/'final_hold_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
