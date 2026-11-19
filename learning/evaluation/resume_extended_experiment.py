"""Recover an interrupted extended-horizon experiment without restarting weights."""
import json,subprocess,sys,shutil,statistics
from pathlib import Path
from collections import Counter
import torch

def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'experiments/runs/extended_horizon_001'
    source=out/'training'
    checkpoint=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    recovered_step=int(checkpoint['step']);remaining=200000-recovered_step
    if not 0<remaining<200000:raise ValueError('Unexpected recovery step.')
    run=out/'continuation';run.mkdir(exist_ok=False)
    # train.py owns creating its output directory.
    run.rmdir()
    protocol=json.loads((out/'protocol.json').read_text())
    protocol['recovery']=dict(recovered_step=recovered_step,remaining_steps=remaining,
        resume=str(source/'checkpoint.pt'),fresh_episode_seed=210,sampling_seed=1000124,
        note='Interrupted terminal process; model and optimizer state retained, episode/RNG state restarted.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def command(arguments,path):
        with path.open('w') as log:subprocess.run([sys.executable]+arguments,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Recovered {recovered_step} steps; training remaining {remaining}.',flush=True)
    command(['scripts/train_formation.py','--steps',str(remaining),'--rollout','512','--epochs','5',
        '--seed','210','--sampling-seed','1000124','--resume',str(source/'checkpoint.pt'),
        '--initial-std','.02','--final-std','.02','--std-decay-steps','1',
        '--entropy-coef','.001','--gae-lambda','.99','--snapshot-interval','40000',
        '--task','configs/tasks/formation_relational_120s_training.json',
        '--communication','configs/communication/nominal.json',
        '--reward','configs/training/formation_reward_mastery.json','--output',str(run)],out/'continuation.log')
    initial=root/'experiments/runs/relational_revision_001/checkpoint_0020480'
    def copyrun(src,dst,filename='checkpoint.pt'):
        dst.mkdir();shutil.copy2(src/'manifest.json',dst/'manifest.json');shutil.copy2(src/filename,dst/'checkpoint.pt')
    task='configs/tasks/formation_relational_120s_evaluation.json'
    def evaluate(folder,profile,seed,count,label):
        command(['scripts/evaluate_formation.py',str(folder),'--task',task,'--communication',
            f'configs/communication/{profile}.json','--seed',str(seed),'--episodes',str(count)],out/f'{label}.log')
        report=json.loads((folder/'evaluation.json').read_text());rows=report['results'];success=[x for x in rows if x['reason']=='success']
        result=dict(label=label,path=str(folder),successes=len(success),episodes=len(rows),success_rate=report['success_rate'],
            collisions=sum(x['reason']=='collision' for x in rows),mean_rmse_m=statistics.mean(x['mean_rmse_m'] for x in rows),
            median_success_time_s=statistics.median(x['diagnostics']['elapsed_time_s'] for x in success) if success else None,
            mean_abs_speed_mps=statistics.mean(x['diagnostics']['mean_abs_speed_mps'] for x in rows),
            failure_stages=dict(Counter(x['diagnostics']['failure_stage'] or x['reason'] for x in rows)),report=report)
        (out/f'{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
        return result
    baseline=out/'baseline';copyrun(initial,baseline)
    choices=[evaluate(baseline,'nominal',11000000,12,'baseline_selection')]
    checkpoints=[(source,p.name,'original_'+p.stem) for p in sorted(source.glob('checkpoint_*.pt'))]
    checkpoints.append((source,'checkpoint.pt',f'original_recovered_{recovered_step:07d}'))
    checkpoints += [(run,p.name,'continued_'+p.stem) for p in sorted(run.glob('checkpoint_*.pt'))]
    checkpoints.append((run,'checkpoint.pt','continued_final'))
    for src,filename,label in checkpoints:
        folder=out/label;copyrun(src,folder,filename)
        choices.append(evaluate(folder,'nominal',11000000,12,label+'_selection'))
    def score(row):return (row['success_rate'],-row['mean_rmse_m'],-(row['median_success_time_s'] or float('inf')))
    eligible=[x for x in choices if x['collisions']==0]
    selected=Path(max(eligible,key=score)['path'])
    trained=[x for x in eligible if x['path']!=str(baseline)]
    if not trained:raise RuntimeError('No trained checkpoint passed selection collision gate.')
    best=Path(max(trained,key=score)['path']);results=[]
    for label,folder in [('baseline',baseline),('trained',best)]:
        for profile in ('nominal','good'):results.append(evaluate(folder,profile,12000000,40,label+'_'+profile+'_heldout'))
    summary=dict(protocol=protocol,selected=str(selected),best_trained=str(best),choices=choices,results=results,
        total_new_training_steps=recovered_step+json.loads((run/'training_summary.json').read_text())['completed_steps'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for folder in {selected,best}:
        trajectory=folder/'trajectory_policy_formation_relational_120s_evaluation_v1_nominal_12000000.json'
        if trajectory.exists():shutil.copy2(trajectory,folder/'trajectory.json')
        command(['scripts/create_replay.py',str(folder)],out/(folder.name+'_replay.log'))
        command(['scripts/check_actor_export.py',str(folder)],out/(folder.name+'_export.log'))
    lines=['# Extended-horizon training results','',
        'Same 120-second limit, distance/stop tolerances and common final starts for both policies.',
        f'Additional training steps: {summary["total_new_training_steps"]}. Training recovered from an interrupted process at step {recovered_step}.',
        'Actor, critic and optimizer state were restored; episode and RNG state restarted as documented in protocol.json.',
        'Selection: 12 development starts. Final tests: 40 different common starts.',
        'Near-triangle initial positions with random headings. No broad-start or five-meter-mission claim.','',
        '| Policy/profile | Success | Collisions | Median successful duration (s) | Mean RMSE (m) |',
        '|---|---:|---:|---:|---:|']
    for row in results:lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['collisions']} | {row['median_success_time_s']} | {row['mean_rmse_m']:.4f} |")
    lines += ['',f'Selected before final tests: {selected}',f'Best trained checkpoint: {best}',
        'No automatic deployment or replacement of the registered research candidate.']
    (root/'docs/extended_horizon_results.md').write_text('\n'.join(lines)+'\n')
    print('Completed recovered extended experiment.',flush=True)

if __name__=='__main__':main()
