"""Assemble an explicit experimental history mission bundle; never select it."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    checkpoint=Path(a.checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    if state['contract'] not in ('formation_history_v5','formation_history_risk_v6') or state['neighbor_dim'] not in (11,13):raise ValueError('History actor required.')
    bundle=torch.load('experiments/runs/own_alignment_network_001/candidate_bundle.pt',map_location='cpu',weights_only=True)
    fresh=bundle['formation']
    bundle['formation']={k:state[k] for k in ['contract','actor','num_agents','local_dim','neighbor_dim','critic_dim','geometry_summary']}
    bundle['formation']['peer_history']=manifest['task']['peer_history']
    bundle['formation']['fresh_actor']=fresh
    alignment_path=Path('experiments/runs/history_coordination_001/alignment_stationary/checkpoint.pt')
    alignment=torch.load(alignment_path,map_location='cpu',weights_only=True)
    bundle['alignment']={k:alignment[k] for k in ['contract','actor','num_agents','local_dim','neighbor_dim','critic_dim','geometry_summary','local_only']}
    bundle['alignment'].update(component='alignment',stationary_alignment=True)
    bundle['task']=json.load(open('configs/tasks/history_full_mission.json'))
    bundle['task']['formation_peer_history']=manifest['task']['peer_history']
    bundle['acquisition_routing']=bundle['task']['acquisition_routing']
    bundle.update(architecture='local_history_fresh_experts_stationary_slot_v2',status='experimental_not_selected',source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    torch.save(bundle,out);print(str(out),flush=True)

if __name__=='__main__':main()
