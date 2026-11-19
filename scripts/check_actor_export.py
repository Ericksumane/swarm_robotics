"""Verify the deployment service matches in-process actor inference."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack
def main():
 parser=argparse.ArgumentParser();parser.add_argument('run');cli=parser.parse_args()
 run=Path(cli.run);manifest=json.loads((run/'manifest.json').read_text())
 artifact=run/'actor.pt'
 checkpoint=torch.load(artifact,map_location='cpu',weights_only=True)
 args,policy,_=make_policy(checkpoint['num_agents'],local_dim=checkpoint.get('local_dim',9),critic_dim=checkpoint.get('critic_dim'),neighbor_dim=checkpoint.get('neighbor_dim',7),geometry_summary=checkpoint.get('geometry_summary',False),hold_std_scale=checkpoint.get('hold_std_scale'));policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
 env=(MissionEnv if manifest['task'].get('environment')=='mission' else FormationEnv)(manifest['task'],manifest['communication'],manifest['reward']);obs,_=env.reset(seed=99991)
 raw=pack(obs)[0:1];h,m=hidden_inputs(1,args)
 with torch.no_grad():expected,_=policy.act(raw,h,m,deterministic=True)
 request=json.dumps({'observation':raw[0].tolist()})+'\n'
 result=subprocess.run([sys.executable,'scripts/actor_service.py',str(artifact)],input=request,text=True,capture_output=True,check=True,timeout=30)
 actual=np.asarray(json.loads(result.stdout)['action'])
 np.testing.assert_allclose(actual,expected.numpy()[0],atol=1e-6)
 print('Actor-only exported inference parity: PASSED')
if __name__=='__main__':main()
