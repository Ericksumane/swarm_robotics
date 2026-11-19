"""Bounded own-state alignment trial; does not change the selected default."""
from pathlib import Path
import concurrent.futures,copy,json,subprocess,sys
import torch

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/own_alignment_network_001';out.mkdir(exist_ok=False)
 registry=json.load(open('experiments/selected_startup_candidate.json'));source=torch.load(registry['actor_bundle'],map_location='cpu',weights_only=True)
 initial=out/'initial_actor.pt';torch.save(source['alignment'],initial)
 reward=json.load(open('configs/training/alignment_reward.json'));reward.update(reward_version='own_state_alignment_v2',alignment_linear=20.)
 reward_path=out/'reward.json';reward_path.write_text(json.dumps(reward,indent=2))
 with (out/'training.log').open('w') as f:subprocess.run([sys.executable,'scripts/train_formation.py','--steps','20000','--snapshot-interval','10000','--seed','458','--actor-lr','.0001','--gamma','.99','--initial-std','.08','--final-std','.01','--std-decay-steps','15000','--entropy-coef','0','--initial-actor',str(initial),'--task','configs/tasks/alignment_own_state_v2.json','--reward',str(reward_path),'--output',str(out/'training')],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
 candidates=[];development={}
 for checkpoint in sorted((out/'training').glob('checkpoint_*.pt')):
  dest=out/(checkpoint.stem+'_dev')
  with dest.with_suffix('.log').open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_startup_component.py','--checkpoint',str(checkpoint),'--task','configs/tasks/alignment_own_state_v2.json','--reward',str(reward_path),'--communication','configs/communication/nominal.json','--output',str(dest),'--seed','82000000','--episodes','8','--role','development'],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((dest/'evaluation.json').read_text());score=(-d['collisions'],d['successes'],-sum(r['elapsed_time_s'] for r in d['results']));candidates.append((score,dest/'actor.pt'));development[checkpoint.stem]=dict(successes=d['successes'],collisions=d['collisions'],score=list(score))
 score,actor=max(candidates,key=lambda item:item[0]);b=copy.deepcopy(source);b['alignment']=torch.load(actor,map_location='cpu',weights_only=True);b['architecture']='local_learned_startup_v2_own_alignment';b['status']='unvalidated_own_alignment_candidate';b['adaptation']=dict(type='Own-state-only alignment; other actors unchanged',development=development)
 bundle=out/'candidate_bundle.pt';torch.save(b,bundle);(out/'selection.json').write_text(json.dumps(dict(actor=str(actor),bundle=str(bundle),development=development),indent=2))
 def run(profile):
  dest=out/('gazebo_'+profile)
  with (out/('gazebo_'+profile+'.log')).open('w') as f:subprocess.run(['bash','scripts/run_live_gazebo.sh','--headless','--fast','--seed','42','--communication',profile,'--startup-bundle',str(bundle),'--output',str(dest)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((dest/'summary.json').read_text());return profile,dict(reason=d['reason'],simulated_s=d['simulated_s'],final_policy=d['final_policy'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(run,['nominal','good','moderate','severe']))
 (out/'summary.json').write_text(json.dumps(dict(selection=dict(actor=str(actor),bundle=str(bundle),development=development),physics_diagnostic=results,scope='Single seed 42 diagnostic, not reserved validation; default unchanged'),indent=2));print(json.dumps({p:dict(reason=d['reason'],time_s=d['simulated_s']) for p,d in results.items()}),flush=True)

if __name__=='__main__':main()
