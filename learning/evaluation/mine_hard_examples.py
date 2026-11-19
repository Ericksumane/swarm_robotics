"""Mine fresh baseline failures for reset variation, never final-test examples."""
import argparse,json,hashlib
from dataclasses import asdict
from pathlib import Path
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.learning.mappo_adapter import make_policy,pack,hidden_inputs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    cli=parser.parse_args();root=Path(__file__).resolve().parents[2]
    source=Path(cli.source);out=Path(cli.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    args,policy,_=make_policy(checkpoint['num_agents'],local_dim=checkpoint.get('local_dim',9),
        critic_dim=checkpoint.get('critic_dim'),neighbor_dim=checkpoint.get('neighbor_dim',7),geometry_summary=checkpoint.get('geometry_summary',False),hold_std_scale=checkpoint.get('hold_std_scale'))
    policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
    manifest=json.loads((source/'manifest.json').read_text())
    communication=json.loads((root/'configs/communication/good.json').read_text())
    entries=[];rows=[]
    for kind,taskfile,start,count in [('near','formation_relational_120s_evaluation.json',20000000,40),
                                      ('broad','formation_relational_broad_evaluation.json',21000000,20)]:
        task=json.loads((root/'configs/tasks'/taskfile).read_text())
        env=FormationEnv(task,communication,manifest['reward'])
        for seed in range(start,start+count):
            obs,_=env.reset(seed);initial=[asdict(s) for s in env.simulation.states]
            h,m=hidden_inputs(len(obs),args)
            while True:
                with torch.no_grad():action,_=policy.act(pack(obs),h,m,deterministic=True)
                obs,_,term,trunc,info=env.step(action.numpy())
                if term or trunc:break
            row=dict(kind=kind,seed=seed,reason=info['reason'],diagnostics=info['diagnostics'])
            rows.append(row)
            if info['reason']!='success':entries.append(dict(source_kind=kind,source_seed=seed,
                failure_stage=info['diagnostics']['failure_stage'],states=initial))
            print(json.dumps(dict(kind=kind,seed=seed,reason=info['reason'],hard_bank_size=len(entries))),flush=True)
    if not entries:raise RuntimeError('No failures mined; focused sampling is unnecessary for this pool.')
    bank=dict(source=str(source),source_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),
        communication=communication,entries=entries,results=rows,
        note='Fresh mining seeds only; no old 12000000 or new final 23000000 seeds are used for training.')
    (out/'hard_bank.json').write_text(json.dumps(bank,indent=2))
    focused=json.loads((root/'configs/tasks/formation_relational_120s_training.json').read_text())
    focused['task_name']='formation_hard_focus_v1'
    focused['initialization']['near_formation']['probability']=.8
    focused['initialization']['hard_examples']=dict(probability=.3,position_jitter_m=.04,
        heading_jitter_rad=.35,minimum_pair_error_m=.12,entries=entries)
    (out/'focused_task.json').write_text(json.dumps(focused,indent=2))
    print('Mined hard starts:',len(entries),flush=True)

if __name__=='__main__':main()
