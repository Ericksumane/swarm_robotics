"""Summarize measured timed-campaign results without inferring convergence."""
import argparse,json
from collections import Counter
from pathlib import Path
import statistics

def main():
 parser=argparse.ArgumentParser();parser.add_argument('run');cli=parser.parse_args()
 run=Path(cli.run);control=run.with_name(run.name+'_control')
 status=json.loads((control/'status.json').read_text())
 if status['phase']!='complete':raise RuntimeError('Campaign and evaluations are not complete.')
 summary=json.loads((run/'training_summary.json').read_text())
 manifest=json.loads((run/'manifest.json').read_text())
 lines=['# Thirty-minute formation training report','',
  '## Measured execution','',
  f"- Training elapsed: {summary['elapsed_s']:.2f} seconds.",
  f"- Environment steps: {summary['completed_steps']}.",
  f"- Completed training episodes: {summary['episodes']}.",
  f"- Stop reason: {summary['stop_reason']}.",
  f"- Training outcomes: {summary.get('outcomes', {})}.",
  '- Three agents share one actor; an environment step includes all three actions.',
  '- Offline training only; no physical robot was actuated.','',
  '## Evaluation','',
  '| Scenario | Success | Collision | Time limit | Mean formation RMSE (m) | Minimum clearance (m) |',
  '|---|---:|---:|---:|---:|---:|']
 for name in ('near_nominal','broad_nominal','broad_moderate','broad_severe'):
  report=json.loads((control/(name+'.json')).read_text());rows=report['results'];counts=Counter(x['reason'] for x in rows)
  lines.append(f"| {name} | {counts['success']}/{len(rows)} | {counts['collision']} | {counts['time_limit']} | {statistics.mean(x['mean_rmse_m'] for x in rows):.4f} | {min(x['minimum_clearance_m'] for x in rows):.4f} |")
 lines += ['', '## Interpretation and limits','',
  'Evaluation seeds start at 200100 and are now development evidence. Five',
  'episodes per scenario are insufficient to establish reliability. Near-start',
  'stabilization is not evidence of formation acquisition from dispersed starts.',
  'Formation-and-hold is not the five-meter translation mission.',
  'Synthetic communication and geometry do not establish physical transfer.',
  'A valid checkpoint and finite optimization losses do not establish task success.','',
  '## Intermediate checkpoints','',
  'Snapshots at approximately 5, 10 and 20 minutes are preserved under this run.',
  'They are development comparisons, not a formally selected best policy.','',
  '## Reproduction','',
  'Configuration, source hashes and upstream commit: manifest.json.',
  'Updates: updates.jsonl. Episode outcomes: episodes.jsonl.',
  'Final weights: checkpoint.pt. Actor-only artifact: actor.pt.',
  'Evaluation details and logs: adjacent _control directory.',
  'Visual replay: replay.html. Gazebo replay accepts this run directory.']
 report_path=Path('docs/training_30min_results.md')
 report_path.write_text('\n'.join(lines)+'\n')
 print(report_path.resolve())
 print('\n'.join(lines[:25]))

if __name__=='__main__':main()
