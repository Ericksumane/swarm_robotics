"""Register a research candidate only after paired final evaluations complete."""
import json,re
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'experiments/runs/extended_horizon_001'
    summary=json.loads((out/'summary.json').read_text())
    rows={x['label']:x for x in summary['results']}
    base=[rows['baseline_'+p+'_heldout'] for p in ('nominal','good')]
    new=[rows['trained_'+p+'_heldout'] for p in ('nominal','good')]
    valid=(new[0]['successes']>base[0]['successes'] and new[1]['successes']>=base[1]['successes']
        and all(n['collisions']<=b['collisions'] for n,b in zip(new,base)))
    best=Path(summary['best_trained'])
    broad=json.loads((out/'final_broad_diagnostic/comparison.json').read_text()) if (out/'final_broad_diagnostic/comparison.json').exists() else None
    paired={}
    for profile,b,n in zip(('nominal','good'),base,new):
        previous={x['seed']:x['reason']=='success' for x in b['report']['results']}
        paired[profile]=dict(new_only=sum(x['reason']=='success' and not previous[x['seed']] for x in n['report']['results']),
            previous_only=sum(x['reason']!='success' and previous[x['seed']] for x in n['report']['results']))
    recordfile=root/'experiments/selected_candidate.json'
    if valid:
        history=root/'experiments/candidate_history';history.mkdir(exist_ok=True)
        backup=history/'before_extended_horizon.json'
        if recordfile.exists() and not backup.exists():backup.write_text(recordfile.read_text())
        relative=str(best.relative_to(root))
        record=dict(status='research_candidate_not_hardware_validated',contract='formation_relational_v4',
            run=relative,actor=relative+'/actor.pt',checkpoint=relative+'/checkpoint.pt',replay=relative+'/replay.html',
            episode_limit_s=120.,num_agents=3,pair_distances_m=[1.,1.,1.],
            additional_training_steps=summary['total_new_training_steps'],
            scope='near-triangle starts with random headings; formation and stationary hold only',
            evaluation=dict(seed=12000000,episodes=40,
                nominal_successes=new[0]['successes'],good_successes=new[1]['successes'],
                nominal_collisions=new[0]['collisions'],good_collisions=new[1]['collisions'],
                baseline_nominal_successes=base[0]['successes'],baseline_good_successes=base[1]['successes']),
            full_five_meter_mission_validated=False,hardware_validated=False,
            report='docs/extended_training_conclusion.md')
        if broad and best.name=='continued_final':record['broad_initialization_diagnostic']=broad
        recordfile.write_text(json.dumps(record,indent=2)+'\n')
    lines=['# Extended training conclusion','',
        'Both policies were tested on the same 40 initial seeds with identical 120-second limits.',
        'The new policy completed 200000 additional training steps; checkpoint selection used separate seeds.',
        'An interrupted process was recovered with actor/critic/optimizer state intact and fresh episode RNG.','',
        '| Communication | Previous success | Trained success | Previous collisions | Trained collisions |',
        '|---|---:|---:|---:|---:|']
    for profile,b,n in zip(('nominal','good'),base,new):lines.append(f"| {profile} | {b['successes']}/40 | {n['successes']}/40 | {b['collisions']} | {n['collisions']} |")
    lines+=['',f'Paired success changes: {paired}',f'Registered as updated research candidate: {valid}',
        f'Candidate: {best}',
        'One training seed and 40 evaluation starts do not establish a population success rate.',
        'These tests cover positions near the target triangle and random initial headings.']
    if broad:lines+=['',f'Separate broad-position diagnostic (10 cases): {broad}',
        'The broad-position diagnostic is not a generalization guarantee.']
    lines+=['','## Geometry and scope','',
        'The retained triangle is equilateral: all three center distances are 1m.',
        'Distances 1m,1m,2m define a collinear degenerate triangle and were not substituted.',
        'The geometric tolerance and velocity/hold thresholds were not relaxed.',
        'Five-meter translation and hardware execution remain unvalidated.',
        'The registered candidate is for research and offline replay; ROS actuator deployment is not approved by these results.',
        '', '## External references used','',
        'VMAS DiffDrive equations informed the independent unicycle integration check.',
        'BenchMARL was reviewed for standardized training/evaluation practices.',
        'ChoiRbot was reviewed for decentralized ROS 2 architecture; its README targets an older ROS release.',
        'See repository_review.md for source links and compatibility boundaries.']
    (root/'docs/extended_training_conclusion.md').write_text('\n'.join(lines)+'\n')
    if valid:
        readme=root/'README.md';text=readme.read_text()
        section=f"""## Latest measured training status

The extended MAPPO candidate completed 200000 additional steps with 120-second
formation-only episodes. On 40 common final starts, success was
{new[0]['successes']}/40 nominal and {new[1]['successes']}/40 good communication,
versus {base[0]['successes']}/40 and {base[1]['successes']}/40 for the frozen baseline.
The triangle retains 1m distance for every pair. Initial positions in the main
evaluation are near formation; see docs/extended_training_conclusion.md for
the separate broad-position diagnostic and evidence limits.

Research artifact paths are in experiments/selected_candidate.json.
The five-meter mission and hardware transfer are not validated. Gazebo replay
remains visualization of recorded Python trajectories.

"""
        if '## Latest measured training status' in text:text=re.sub(r'(?ms)^## Latest measured training status\n.*?(?=^## |\Z)',lambda _:section,text,count=1)
        else:text=text.rstrip()+'\n\n'+section
        readme.write_text(text.rstrip()+chr(10))
    print('\n'.join(lines[:14]));print('Updated research candidate:',valid)

if __name__=='__main__':main()
