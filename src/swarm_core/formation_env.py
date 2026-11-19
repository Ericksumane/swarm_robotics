"""Initial cooperative formation environment.

Custom multi-agent interface, not a Gymnasium Env subclass.
Training integration is provided by learning/training/train.py.
"""

from copy import deepcopy

import numpy as np

from swarm_core.rewards import formation_hold_reward
from swarm_core.geometry import formation_metrics
from swarm_core.simulation import Simulation
from swarm_core.termination import FormationTermination


class FormationEnv:
    def __init__(self, task, communication, reward_weights):
        self.task = deepcopy(task)
        self.reward_weights = deepcopy(reward_weights)
        for key in ('outside_area_penalty','progress_weight','success_bonus','proximity_weight','ready_reward_per_s'):
            value = reward_weights.get(key, 0.)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative.')
        if reward_weights.get('proximity_weight', 0.) > 0:
            margin = reward_weights.get('proximity_margin_m', 0.)
            if not np.isfinite(margin) or margin <= 0:
                raise ValueError('Proximity margin must be positive and finite.')
        if type(reward_weights.get('settling_only_within_formation',False)) is not bool:
            raise ValueError('Settling gate must be boolean.')
        self.progress_metric=reward_weights.get('progress_metric','formation_rmse_m')
        if self.progress_metric not in ('formation_rmse_m','max_pair_error_m'):
            raise ValueError('Unsupported formation progress metric.')
        if reward_weights.get('proximity_aggregation','mean') not in ('mean','max'):
            raise ValueError('Unsupported proximity aggregation.')
        for key in ('closing_risk_weight','closing_risk_horizon_s','closing_risk_margin_m'):
            value=reward_weights.get(key,0.)
            if not np.isfinite(value) or value<0: raise ValueError('Invalid closing-risk setting.')
        if reward_weights.get('closing_risk_weight',0.)>0 and min(reward_weights.get('closing_risk_horizon_s',0.),reward_weights.get('closing_risk_margin_m',0.))<=0:
            raise ValueError('Positive closing-risk horizon and margin required.')
        self.simulation = Simulation(task, communication)
        self.monitor = FormationTermination(task)
        self.done = False

    def reset(self, seed=None):
        """Reset physical state, communication queues and episode counters."""
        self.simulation.reset(seed=seed)
        self.monitor.reset()
        self.done = False
        return self.simulation.observe_all(), {
            "time_s": self.simulation.time_s
        }

    def step(self, actions):
        """Return observations, shared rewards, outcome flags and metrics."""
        if self.done:
            raise RuntimeError("Episode ended; call reset before stepping.")

        previous_error = formation_metrics(
            self.simulation.states,
            self.task["team"]["desired_pair_distance_m"],
            self.task["robot_defaults"]["footprint_radius_m"],
        )[self.progress_metric]
        heading_weight = self.reward_weights.get("pose_heading_weight", 0.)
        direction = self.task['mission']['direction_world']
        target_heading = np.arctan2(direction[1], direction[0])
        previous_heading_error = (
            np.mean([1. - np.cos(state.heading - target_heading) for state in self.simulation.states])
            if heading_weight else 0.
        )
        self.simulation.advance(actions)

        reward, terms = formation_hold_reward(
            self.simulation.states,
            self.task["robot_defaults"],
            self.task["team"]["desired_pair_distance_m"],
            self.simulation.dt,
            self.reward_weights,
        )
        outcome = self.monitor.update(self.simulation.states)
        if (self.reward_weights.get('settling_only_within_formation',False)
                and not outcome['diagnostics']['geometry_ready']):
            # Acquisition should not pay a settling cost before reaching geometry.
            reward -= terms['motion']
            terms['motion'] = 0.


        outside_penalty = self.reward_weights["outside_area_penalty"]
        if not np.isfinite(outside_penalty) or outside_penalty < 0:
            raise ValueError("Outside-area penalty must be finite and nonnegative.")

        terms["outside_area"] = (
            -outside_penalty
            if outcome["reason"] == "outside_area"
            else 0.0
        )
        reward += terms["outside_area"]
        if self.reward_weights.get("progress_weight", 0):
            current_error = formation_metrics(
                self.simulation.states,
                self.task["team"]["desired_pair_distance_m"],
                self.task["robot_defaults"]["footprint_radius_m"],
            )[self.progress_metric]
            # Task-specific dense shaping; not policy-invariant potential shaping.
            terms["progress"] = self.reward_weights["progress_weight"] * (previous_error - current_error)
            reward += terms["progress"]

        if heading_weight:
            heading_error = np.mean([1. - np.cos(state.heading - target_heading) for state in self.simulation.states])
            terms["pose_heading"] = float(heading_weight * (
                previous_heading_error - heading_error - self.simulation.dt * heading_error
            ))
            reward += terms["pose_heading"]
        terms["success"] = self.reward_weights.get("success_bonus", 0) if outcome["reason"] == "success" else 0.0
        reward += terms["success"]
        # Dense credit for qualifying hold steps, with unchanged success rules.
        terms['ready'] = (self.simulation.dt*self.reward_weights.get('ready_reward_per_s',0.)
            if outcome['hold_time_s'] > 0 else 0.)
        reward += terms['ready']

        if self.reward_weights.get("proximity_weight", 0):
            geometry = formation_metrics(self.simulation.states,
                self.task["team"]["desired_pair_distance_m"],
                self.task["robot_defaults"]["footprint_radius_m"])
            margin = self.reward_weights["proximity_margin_m"]
            if not np.isfinite(margin) or margin <= 0:
                raise ValueError("Proximity margin must be positive and finite.")
            clearances = geometry["pair_distances_m"] - 2 * self.task["robot_defaults"]["footprint_radius_m"]
            aggregate=np.max if self.reward_weights.get('proximity_aggregation','mean')=='max' else np.mean
            terms["proximity"] = -self.simulation.dt * self.reward_weights["proximity_weight"] * float(aggregate(np.maximum(0., (margin-clearances)/margin)**2))
            reward += terms["proximity"]

        if self.reward_weights.get('closing_risk_weight',0.)>0:
            from swarm_core.collision_risk import closing_risk
            risk=closing_risk(self.simulation.states,self.task['robot_defaults']['footprint_radius_m'],
                self.reward_weights['closing_risk_horizon_s'],self.reward_weights['closing_risk_margin_m'])
            terms['closing_risk']=-self.simulation.dt*self.reward_weights['closing_risk_weight']*risk
            reward+=terms['closing_risk']

        self.done = outcome["terminated"] or outcome["truncated"]

        # Every agent receives the same cooperative reward.
        rewards = np.full(
            len(self.simulation.states), reward, dtype=np.float32
        )
        info = {
            "time_s": self.simulation.time_s,
            "reward_terms": terms,
            **outcome,
        }

        return (
            self.simulation.observe_all(),
            rewards,
            outcome["terminated"],
            outcome["truncated"],
            info,
        )