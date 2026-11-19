"""Export only a measured endpoint improvement; preserve strict-scope limits."""
from pathlib import Path
import argparse,json,torch

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--validation-run',default='experiments/runs/approach_full_validation_001');cli=parser.parse_args()
 root=Path(__file__).resolve().parents[1];folder=root/cli.validation_run
 summary=json.loads((folder/'summary.json').read_text());base=summary['results']['cruise'];candidate=summary['results']['composed']
 if candidate['successes']<=base['successes'] or candidate['collisions']>base['collisions']:raise RuntimeError('Candidate did not pass the paired endpoint improvement gate.')
 provenance=summary['provenance']
 bundle=dict(architecture='local_latched_mappo_v2',contract='mission_local_goal_v3',horizon_m=provenance.get('approach_distance_m',.5),observation_horizon_m=.5,epsilon_m=1e-5,provenance=provenance,status='research_only',continuous_formation_validated=candidate['strict_successes']==provenance['episodes'],hardware_validated=False)
 for name in ['cruise','approach']:
  checkpoint=torch.load(root/provenance[name]/'checkpoint.pt',map_location='cpu',weights_only=True)
  bundle[name]={key:checkpoint[key] for key in ['actor','num_agents','local_dim','critic_dim','neighbor_dim','geometry_summary']}
  bundle[name]['hold_std_scale']=checkpoint.get('hold_std_scale')
 torch.save(bundle,folder/'policy_bundle.pt');print(folder/'policy_bundle.pt')

if __name__=='__main__':main()
