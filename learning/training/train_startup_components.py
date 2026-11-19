"""Train acquisition and alignment separately; preserve translation experts."""
from pathlib import Path
import concurrent.futures,copy,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/dispersed_recovery_001';out.mkdir(exist_ok=False)
 reference=torch.load(root/'experiments/runs/local_network_routing_001/selected_policy_bundle.pt',map_location='cpu',weights_only=True)
 initial=copy.deepcopy(reference['cruise']['actor'])
 initial['act.action_out.fc_mean.weight'].zero_();initial['act.action_out.fc_mean.bias'].zero_()
 actor=out/'alignment_initial_actor.pt';torch.save(dict(actor=initial),actor)
 common=[sys.executable,'scripts/train_formation.py','--actor-lr','.0001','--entropy-coef','0','--communication','configs/communication/nominal.json']
 jobs={
  'acquisition':common+['--steps','50000','--snapshot-interval','25000','--seed','455','--gamma','.999','--gae-lambda','.99','--initial-std','.05','--final-std','.02','--std-decay-steps','25000','--resume','experiments/runs/extended_horizon_001/continued_final/checkpoint.pt','--task','configs/tasks/dispersed_acquisition_training.json','--reward','configs/training/dispersed_acquisition_reward.json','--output',str(out/'acquisition')],
  'alignment':common+['--steps','40000','--snapshot-interval','20000','--seed','456','--gamma','.99','--gae-lambda','.95','--initial-std','.2','--final-std','.02','--std-decay-steps','20000','--initial-actor',str(actor),'--task','configs/tasks/alignment_v1.json','--reward','configs/training/alignment_reward.json','--output',str(out/'alignment')]}
 def run(name):
  with (out/(name+'.log')).open('w') as f:subprocess.run(jobs[name],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  return name,json.loads((out/name/'training_summary.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(run,jobs))
 (out/'training_summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
