"""Matched frozen original reference on the 71M candidate evaluation starts."""
from pathlib import Path
import concurrent.futures,json,subprocess,sys

def main():
 root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/local_network_reference_001';out.mkdir(exist_ok=False)
 def run(profile):
  with (out/(profile+'.log')).open('w') as f:subprocess.run([sys.executable,'scripts/evaluate_policy_bundle.py','--bundle','experiments/runs/geometry_control_001/selected_policy_bundle.pt','--task','configs/tasks/translate_stop_hold_hysteresis.json','--communication',f'configs/communication/{profile}.json','--output',str(out/profile),'--seed','71000000','--episodes','40','--role','final'],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
  d=json.loads((out/profile/'evaluation.json').read_text());return profile,{k:d[k] for k in ['successes','strict_successes','collisions']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=dict(ex.map(run,['nominal','good','moderate','severe']))
 (out/'summary.json').write_text(json.dumps(dict(results=results,seed=71000000,episodes=40,weights_changed=False),indent=2));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
