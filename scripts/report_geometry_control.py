"""Summarize the completed controlled experiment without changing registries."""
from pathlib import Path
import json
import numpy as np

def main():
 root=Path(__file__).resolve().parents[1];out=root/'experiments/runs/geometry_control_001'
 summary=json.loads((out/'summary.json').read_text())
 lines=['# Controlled cruise geometry continuation','',
 'Reserved evaluation: 40 matched seeds 61000000 through 61000039 per cell.',
 'Both trained branches use the same seed and settings; only barrier weight differs.',
 'The approach actor and HOLD hysteresis remain fixed. All tests start already',
 'formed and approximately aligned, from zero progress, with 5 m / 120 s tasks.','',
 '| Model | Communication | Endpoint success | Entire-path success | Collisions | Mean formation compliance |',
 '|---|---|---:|---:|---:|---:|']
 metrics={}
 for name in ['baseline','control','barrier']:
  for profile in ['nominal','good']:
   d=json.loads((out/(name+'_'+profile+'_final')/'evaluation.json').read_text())
   compliance=float(np.mean([r['formation_compliant_fraction'] for r in d['results']]))
   metrics[name+'_'+profile]=dict(successes=d['successes'],strict_successes=d['strict_successes'],collisions=d['collisions'],mean_formation_compliance=compliance)
   lines.append(f"| {name} | {profile} | {d['successes']}/40 | {d['strict_successes']}/40 | {d['collisions']} | {100*compliance:.1f}% |")
 lines+=['','Endpoint success requires every robot within 0.2 m of its own target,',
 'all pair errors <=0.1 m, linear speed <=0.03 m/s and yaw speed <=0.05 rad/s',
 'continuously for 2 s. Entire-path success additionally requires pair error',
 '<=0.1 m at every simulated step. Mean compliance is a time fraction, not',
 'a mission success rate. No tolerance or time-limit relaxation was used.','',
 'Each training branch completed 20000 additional environment steps. The selected',
 'checkpoint was chosen on 8 nominal development cases before reserved tests.',
 'See geometry_control_protocol.md for the cost equation and fixed settings.',
 'The barrier is a project-specific reward extension; PPO updates use upstream',
 'MAPPO unchanged. A single seed does not establish training-seed robustness.','',
 'Training logs, source hashes, optimizer settings, checkpoint selection and all',
 'reserved trajectories are under experiments/runs/geometry_control_001.',
 'No bad/severe communication, broad-position formation acquisition, real robot',
 'dynamics or hardware transfer is established by this experiment. Gazebo remains',
 'recorded Python-trajectory playback, not closed-loop Gazebo physics validation.','']
 (root/'docs/geometry_control_results.md').write_text(chr(10).join(lines))
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
 print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
