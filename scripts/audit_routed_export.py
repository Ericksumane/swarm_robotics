"""Compare source means and process inference across local routing transitions."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
import torch
from swarm_core.learning.routed_policy import RoutedPolicy
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import pack

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--validation-run',default='experiments/runs/approach_full_validation_001');cli=parser.parse_args()
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[1];folder=root/cli.validation_run;path=folder/'policy_bundle.pt'
 bundle=torch.load(path,map_location='cpu',weights_only=True);local=RoutedPolicy(bundle)
 manifest=json.loads((root/bundle['provenance']['cruise']/'manifest.json').read_text());env=MissionEnv(manifest['task'],manifest['communication'],manifest['reward']);env.reset(54000000)
 far=pack(env.observe_all())[0];near=far.copy();near[12:14]=[.3,0.]
 observations=[far,near,far,far];distances=[5.,.3,5.,5.];resets=[True,False,False,True];expected_modes=['cruise','approach','approach','cruise'];requests=[];expected=[]
 for raw,distance,reset,mode in zip(observations,distances,resets,expected_modes):
  if reset:local.reset()
  action,actual_mode=local.act(raw,goal_distance_m=distance);assert actual_mode==mode
  source,h,m=local.models[mode]
  with torch.no_grad():original,_=source.act(raw[None,:],h,m,deterministic=True)
  np.testing.assert_allclose(action,original.numpy()[0],atol=1e-6)
  expected.append(action);requests.append(json.dumps(dict(observation=raw.tolist(),goal_distance_m=distance,reset=reset)))
 result=subprocess.run([sys.executable,'scripts/routed_actor_service.py',str(path)],input='\n'.join(requests)+'\n',text=True,capture_output=True,check=True,timeout=60,cwd=root)
 actual=[json.loads(line) for line in result.stdout.splitlines()]
 assert [r['mode'] for r in actual]==expected_modes
 np.testing.assert_allclose([r['action'] for r in actual],expected,atol=1e-6)
 (folder/'export_parity.json').write_text(json.dumps(dict(passed=True,modes=expected_modes,source_means_match=True,process_inference_matches=True,reset_verified=True),indent=2))
 print('Actor-only bundle, source means and process parity: PASSED')
 print('Local approach latch and new-mission reset parity: PASSED')

if __name__=='__main__':main()
