from pathlib import Path
import json,sys
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swarm_core.formation_env import FormationEnv
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import make_policy,pack,hidden_inputs
torch.set_num_threads(1)
out=root/'experiments/runs/hard_focus_001'
run=out/'checkpoint_0120000'
checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
args,policy,_=make_policy(3,local_dim=9,critic_dim=23,neighbor_dim=9,geometry_summary=True)
policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
task=json.loads((root/'configs/tasks/formation_relational_120s_evaluation.json').read_text())
comm=json.loads((root/'configs/communication/good.json').read_text())
weights=json.loads((run/'manifest.json').read_text())['reward']
previous=json.loads((root/'experiments/runs/extended_horizon_001/summary.json').read_text())
old=next(x for x in previous['results'] if x['label']=='trained_good_heldout')['report']['results']
failed=[x for x in old if x['reason']!='success']
results=[]
for oldrow in failed:
    env=FormationEnv(task,comm,weights);obs,_=env.reset(oldrow['seed']);h,m=hidden_inputs(3,args)
    while True:
        with torch.no_grad():action,_=policy.act(pack(obs),h,m,deterministic=True)
        obs,_,term,trunc,info=env.step(action.numpy())
        if term or trunc:break
    geometry=formation_metrics(env.simulation.states,1.,.25)
    row=dict(seed=oldrow['seed'],previous_reason=oldrow['reason'],new_reason=info['reason'],
        diagnostics=info['diagnostics'],final_pair_distances_m=geometry['pair_distances_m'].tolist())
    print(json.dumps(row),flush=True);results.append(row)
report=dict(label='Historical regression audit: inspected cases, not untouched final tests',
    checkpoint=str(run),cases=len(results),recovered=sum(x['new_reason']=='success' for x in results),results=results)
(out/'old_failure_regression.json').write_text(json.dumps(report,indent=2))
print('Recovered historical failures:',report['recovered'],'/',report['cases'])
