"""Initialization-only heading curriculum; never generates robot actions."""
from collections import deque
import math
import numpy as np

def sample_headings(positions, desired_distance, settings, rng):
    """Orient initial bodies near a distance-error descent direction.

    Used only by synthetic reset sampling. Full-width perturbations recover
    uniform random headings. No desired direction is exposed to the actor.
    """
    width = settings['half_width_rad']
    if not math.isfinite(width) or not 0 <= width <= math.pi:
        raise ValueError('Heading half-width must lie in [0, pi].')
    positions=np.asarray(positions,dtype=float)
    headings=[]
    for i, own in enumerate(positions):
        direction=np.zeros(2)
        for j, peer in enumerate(positions):
            if i == j: continue
            delta=peer-own
            distance=float(np.linalg.norm(delta))
            if distance > 1e-12:
                direction+=(distance-desired_distance)*delta/distance
        angle=math.atan2(direction[1],direction[0]) if np.linalg.norm(direction)>1e-12 else rng.uniform(-math.pi,math.pi)
        angle+=rng.uniform(-width,width)
        headings.append((angle+math.pi)%(2*math.pi)-math.pi)
    return headings

class HeadingCurriculum:
    """Promote reset difficulty only after sufficient recent episode success."""
    def __init__(self, settings, seed=0):
        self.widths=tuple(settings['half_widths_rad'])
        self.window=settings.get('window_episodes',20)
        self.threshold=settings.get('promotion_success_rate',.7)
        self.retention=settings.get('previous_stage_probability',0.)
        if not math.isfinite(self.retention) or not 0 <= self.retention < 1:
            raise ValueError('Invalid curriculum retention probability.')
        self.rng=np.random.default_rng(seed)
        self.sampled_stage=None
        if (not self.widths or any(not math.isfinite(x) or not 0<=x<=math.pi for x in self.widths)
                or any(b<a for a,b in zip(self.widths,self.widths[1:]))
                or type(self.window) is not int or self.window<1
                or not math.isfinite(self.threshold) or not 0<self.threshold<=1):
            raise ValueError('Invalid mastery curriculum.')
        self.stage=0
        self.history=deque(maxlen=self.window)

    def record(self, success):
        # Rehearsal success must not promote a harder, untested frontier.
        if self.sampled_stage is not None and self.sampled_stage != self.stage:
            return False
        self.history.append(bool(success))
        promoted=(len(self.history)==self.window and sum(self.history)/self.window>=self.threshold
                  and self.stage+1<len(self.widths))
        if promoted:
            self.stage+=1
            self.history.clear()
            self.sampled_stage=None
        return promoted

    def apply(self, simulation):
        self.sampled_stage=self.stage
        if self.stage>0 and self.retention>0 and self.rng.random()<self.retention:
            self.sampled_stage=int(self.rng.integers(self.stage))
        simulation.task['initialization']['heading_alignment']={'half_width_rad':self.widths[self.sampled_stage]}

    def status(self):
        return dict(stage=self.stage,half_width_rad=self.widths[self.stage],
            sampled_stage=self.sampled_stage,previous_stage_probability=self.retention,
            recent_episodes=len(self.history),recent_success_rate=sum(self.history)/max(1,len(self.history)))
