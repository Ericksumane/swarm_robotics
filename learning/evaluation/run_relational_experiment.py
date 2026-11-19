"""Train, select on development seeds, then evaluate on separate held-out seeds."""
import argparse,json,shutil,subprocess,sys,statistics
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    cli=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    output=Path(cli.output).resolve();output.mkdir(parents=True,exist_ok=False)
    def command(arguments,log):
        with log.open('w') as file:
            subprocess.run([sys.executable]+arguments,cwd=root,stdout=file,stderr=subprocess.STDOUT,check=True)
    protocol=dict(training_seed=67,selection_seed=4000000,selection_episodes=12,
        heldout_seed=5000000,heldout_episodes=40,training_steps=120000,
        selection_rule='highest nominal random-heading development success; fewer collisions; lower mean RMSE',
        scope='formation and stationary hold, not full five-meter mission')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train=output/'training'
    print('Training revised MAPPO for 120000 steps.',flush=True)
    command(['scripts/train_formation.py','--steps','120000','--rollout','512','--epochs','5',
        '--seed','67','--initial-std','.1','--final-std','.02','--std-decay-steps','60000',
        '--entropy-coef','.001','--gae-lambda','.99','--snapshot-interval','20000',
        '--task','configs/tasks/formation_relational_mastery.json',
        '--communication','configs/communication/nominal.json',
        '--reward','configs/training/formation_reward_mastery.json','--output',str(train)],output/'training.log')
    def evaluate(folder,task,profile,seed,episodes,label,mode='policy'):
        command(['scripts/evaluate_formation.py',str(folder),'--task',task,
            '--communication',f'configs/communication/{profile}.json','--seed',str(seed),
            '--episodes',str(episodes),'--mode',mode],output/f'{label}.log')
        report=json.loads((folder/'evaluation.json').read_text())
        rows=report['results']
        result=dict(label=label,successes=sum(x['reason']=='success' for x in rows),episodes=len(rows),
            success_rate=report['success_rate'],collisions=sum(x['reason']=='collision' for x in rows),
            mean_rmse_m=statistics.mean(x['mean_rmse_m'] for x in rows),report=report)
        (output/f'{label}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
        return result
    candidates=[]
    for checkpoint in sorted(train.glob('checkpoint_*.pt')):
        folder=output/checkpoint.stem;folder.mkdir()
        shutil.copy2(train/'manifest.json',folder/'manifest.json')
        shutil.copy2(checkpoint,folder/'checkpoint.pt')
        result=evaluate(folder,'configs/tasks/formation_relational_random_evaluation.json','nominal',4000000,12,checkpoint.stem+'_selection')
        candidates.append((folder,result))
    selected,selection=max(candidates,key=lambda pair:(pair[1]['success_rate'],-pair[1]['collisions'],-pair[1]['mean_rmse_m']))
    print('Selected:',selected.name,flush=True)
    results=[]
    results.append(evaluate(selected,'configs/tasks/formation_relational_random_evaluation.json','nominal',5000000,40,'revised_nominal_heldout'))
    results.append(evaluate(selected,'configs/tasks/formation_relational_random_evaluation.json','good',5000000,40,'revised_good_heldout'))
    results.append(evaluate(selected,'configs/tasks/formation_relational_easy_evaluation.json','nominal',5000000,20,'revised_easy_diagnostic'))
    # Copy the previous candidate so evaluation cannot overwrite earlier evidence.
    baseline_source=root/'experiments/runs/good_communication_audit_001/train_good'
    baseline=output/'previous_baseline';baseline.mkdir()
    for name in ('manifest.json','checkpoint.pt'):shutil.copy2(baseline_source/name,baseline/name)
    results.append(evaluate(baseline,'configs/tasks/formation_relative_critic.json','nominal',5000000,40,'previous_nominal_heldout'))
    results.append(evaluate(selected,'configs/tasks/formation_relational_random_evaluation.json','nominal',5000000,40,'stationary_nominal_heldout',mode='stationary'))
    summary=dict(protocol=protocol,selected=str(selected),selection=selection,
        candidates=[dict(path=str(folder),result=result) for folder,result in candidates],results=results,
        training=json.loads((train/'training_summary.json').read_text()))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    # Restore learned nominal trajectory after the stationary reference evaluation.
    trajectory=selected/'trajectory_policy_formation_relational_random_evaluation_v1_nominal_5000000.json'
    shutil.copy2(trajectory,selected/'trajectory.json')
    command(['scripts/create_replay.py',str(selected)],output/'replay.log')
    command(['scripts/check_actor_export.py',str(selected)],output/'export_check.log')
    lines=['# Revised formation training results','',
        'Selection used 12 development seeds; final tests used 40 different common starts.',
        'One training seed only. Initial geometry remains near formation; broad placement is not tested.',
        'No claim of five-meter translation or hardware readiness.','',
        '| Evaluation | Success | Collisions | Mean RMSE (m) |',
        '|---|---:|---:|---:|']
    for row in results:lines.append(f"| {row['label']} | {row['successes']}/{row['episodes']} | {row['collisions']} | {row['mean_rmse_m']:.4f} |")
    lines+=['',f'Selected checkpoint: {selected}',f'Training summary: {summary["training"]}']
    (root/'docs/relational_training_results.md').write_text('\n'.join(lines)+'\n')
    print('Experiment completed; report: docs/relational_training_results.md',flush=True)

if __name__=='__main__':main()
