"""Initial-state augmentation of captured simulator approaches, without actions."""
import math
import numpy as np
from swarm_core.dynamics import RobotState

def sample_approach_reset(bank,seed,count,limits,area):
 entries=bank.get('entries',[])
 if not entries:raise ValueError('Approach reset bank must not be empty.')
 jitter=np.array([bank.get('position_jitter_m',0.),bank.get('heading_jitter_rad',0.),bank.get('speed_jitter_mps',0.)])
 if not np.isfinite(jitter).all() or np.any(jitter<0):raise ValueError('Approach reset jitter must be finite and nonnegative.')
 rng=np.random.default_rng(np.random.SeedSequence([seed,5181]));entry=entries[int(rng.integers(len(entries)))]
 starts=np.asarray(entry['starts'],dtype=float);goals=np.asarray(entry['goals'],dtype=float)
 if starts.shape!=(count,2) or goals.shape!=(count,2) or not np.isfinite(starts).all() or not np.isfinite(goals).all():raise ValueError('Invalid approach starts or goals.')
 if len(entry['states'])!=count:raise ValueError('Approach state count differs from team size.')
 for _ in range(100):
  states=[]
  for row in entry['states']:
   heading=row['heading']+rng.uniform(-jitter[1],jitter[1]);heading=(heading+math.pi)%(2*math.pi)-math.pi
   speed=float(np.clip(row['speed']+rng.uniform(-jitter[2],jitter[2]),-limits['max_reverse_speed_mps'],limits['max_forward_speed_mps']))
   states.append(RobotState(x=row['x']+rng.uniform(-jitter[0],jitter[0]),y=row['y']+rng.uniform(-jitter[0],jitter[0]),heading=heading,speed=speed,yaw_rate=row['yaw_rate']))
  values=np.array([[s.x,s.y,s.heading,s.speed,s.yaw_rate] for s in states])
  if not np.isfinite(values).all() or np.any(np.abs(values[:,4])>limits['max_yaw_rate_radps']):raise ValueError('Invalid approach state values.')
  radius=limits['footprint_radius_m']
  inside=all(area['x_bounds_m'][0]+radius<=s.x<=area['x_bounds_m'][1]-radius and area['y_bounds_m'][0]+radius<=s.y<=area['y_bounds_m'][1]-radius for s in states)
  separated=all(np.linalg.norm(values[i,:2]-values[j,:2])>2*radius for i in range(count) for j in range(i))
  if inside and separated:return states,[p.copy() for p in starts],[p.copy() for p in goals]
 raise RuntimeError('Could not generate a feasible approach reset.')
