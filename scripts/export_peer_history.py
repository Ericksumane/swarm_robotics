"""Export experimental history actor with its required feature construction."""
from pathlib import Path
import argparse,hashlib,json
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.checkpoint);manifest=json.loads((source.parent/'manifest.json').read_text())
    state=torch.load(source,map_location='cpu',weights_only=True)
    if state['contract'] not in ('formation_history_v5','formation_history_risk_v6') or state['neighbor_dim'] not in (11,13):
        raise ValueError('Expected bounded-history formation contract.')
    if manifest['task']['observation_contract']!=state['contract']:raise ValueError('Manifest mismatch.')
    artifact={k:state[k] for k in ['contract','actor','num_agents','local_dim','critic_dim','neighbor_dim','geometry_summary']}
    artifact.update(component='acquisition',status='experimental_not_selected',peer_history=manifest['task']['peer_history'],checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),local_only=False)
    out=Path(a.output);out.mkdir(exist_ok=False)
    torch.save(artifact,out/'actor.pt');(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(str(out/'actor.pt'))

if __name__=='__main__':main()
