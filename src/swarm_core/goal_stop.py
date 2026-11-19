"""Explicit local terminal stop state, with bounded goal hysteresis.

This is an operational stop supervisor, not a newly learned controller.
It never steers toward a goal: it either retains the policy action or requests
zero velocity after the locally measured arrival/settling condition.
"""
import math
import numpy as np

class GoalStopLatch:
    def __init__(self,settings):
        self.settings=dict(settings)
        required=('entry_m','exit_m','settle_s','time_step_s','linear_tolerance_mps','yaw_tolerance_radps')
        if any(not math.isfinite(settings[k]) or settings[k]<=0 for k in required):raise ValueError('Invalid goal stop settings.')
        if settings['entry_m']>=settings['exit_m']:raise ValueError('Stop entry must be inside the release bound.')
        self.reset()
    def reset(self):self.parked=False;self.settled=0.;self.previous_time=None
    def apply(self,action,observation):
        cfg=self.settings;now=float(observation['control_time_s'])
        if not math.isfinite(now) or (self.previous_time is not None and now<=self.previous_time):raise ValueError('Stop supervisor requires increasing control time.')
        delta=cfg['time_step_s'] if self.previous_time is None else now-self.previous_time
        self.previous_time=now
        distance=observation['goal_distance_m'];local=observation['local_features']
        if distance is None or local[9]>.5:
            self.parked=False;self.settled=0.;return action,False
        if not math.isfinite(distance) or distance<0:raise ValueError('Invalid local goal distance.')
        heading_tolerance=cfg.get('heading_tolerance_rad')
        aligned=heading_tolerance is None or observation['mission_heading_error_rad']<=heading_tolerance
        if distance>cfg['exit_m'] or not aligned:self.parked=False;self.settled=0.
        if self.parked:return np.zeros(2,dtype=np.float32),True
        qualifying=aligned and distance<=cfg['entry_m'] and abs(local[0])<=cfg['linear_tolerance_mps'] and abs(local[1])<=cfg['yaw_tolerance_radps']
        if delta>1.5*cfg['time_step_s']:self.settled=0.
        self.settled=self.settled+min(delta,cfg['time_step_s']) if qualifying else 0.
        if self.settled>=cfg['settle_s']-1e-8:self.parked=True
        return (np.zeros(2,dtype=np.float32),True) if self.parked else (action,False)


def limit_forward_for_arrival(action, own, goal, limits, dt, braking):
    """Cap forward speed using remaining distance along the current heading."""
    if braking is None:
        return action
    if not math.isfinite(braking) or braking <= 0.:
        raise ValueError("Braking envelope requires positive finite deceleration.")
    if goal is None or action[0] <= 0.:
        return action
    dx, dy = goal - np.array([own.x, own.y])
    remaining = dx * math.cos(own.heading) + dy * math.sin(own.heading)
    available = max(0., remaining - max(0., own.speed) * dt)
    capped = np.asarray(action, dtype=np.float32).copy()
    capped[0] = min(capped[0], math.sqrt(2. * braking * available)
                    / limits["max_forward_speed_mps"])
    return capped
