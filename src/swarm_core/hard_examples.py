"""Perturbed on-policy reset sampling around independently mined hard starts."""
import math
import numpy as np
from swarm_core.dynamics import RobotState

def sample_hard_example(config,rng,x_bounds,y_bounds,speed,yaw_rate):
    settings=config['initialization'];hard=settings['hard_examples']
    entries=hard['entries'];count=config['team']['num_agents']
    position_jitter=hard.get('position_jitter_m',.04)
    heading_jitter=hard.get('heading_jitter_rad',.35)
    if not np.isfinite([position_jitter,heading_jitter]).all() or min(position_jitter,heading_jitter)<0:
        raise ValueError('Hard reset perturbations must be finite and nonnegative.')
    if not entries:raise ValueError('Hard reset bank must be nonempty.')
    templates=[]
    for entry in entries:
        states=entry['states']
        if len(states)!=count:raise ValueError('Hard reset team size mismatch.')
        values=np.asarray([[s['x'],s['y'],s['heading']] for s in states],dtype=float)
        if not np.isfinite(values).all():raise ValueError('Hard reset states must be finite.')
        templates.append(values)
    desired=config['team']['desired_pair_distance_m']
    minimum_error=hard.get('minimum_pair_error_m',.12)
    if not math.isfinite(minimum_error) or minimum_error<0:raise ValueError('Invalid minimum hard reset error.')
    for _ in range(settings['max_sampling_attempts']):
        values=templates[int(rng.integers(len(templates)))]
        positions=values[:,:2]-values[:,:2].mean(axis=0)
        angle=rng.uniform(-math.pi,math.pi);c,s=math.cos(angle),math.sin(angle)
        rotation=np.array([[c,-s],[s,c]])
        center=np.array([rng.uniform(*x_bounds),rng.uniform(*y_bounds)])
        positions=positions@rotation.T+center+rng.uniform(-position_jitter,position_jitter,(count,2))
        if any(not(x_bounds[0]<=x<=x_bounds[1] and y_bounds[0]<=y<=y_bounds[1]) for x,y in positions):continue
        distances=[np.linalg.norm(positions[i]-positions[j]) for i in range(count) for j in range(i)]
        if min(distances)<settings['min_center_distance_m']:continue
        if max(abs(d-desired) for d in distances)<minimum_error:continue
        headings=(values[:,2]+angle+rng.uniform(-heading_jitter,heading_jitter,count)+math.pi)%(2*math.pi)-math.pi
        return [RobotState(float(x),float(y),float(h),float(speed),float(yaw_rate)) for (x,y),h in zip(positions,headings)]
    raise RuntimeError('Could not sample a valid perturbed hard reset.')
