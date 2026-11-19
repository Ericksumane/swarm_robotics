"""Initial simulation kernel: motion, transmission and message reception.

States are currently ideal local estimates as well as physical states.
Measurement noise and drift will be added before robustness training.
This is not yet a complete reinforcement-learning environment.
"""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np

from swarm_core.observations import append_formation_geometry
from swarm_core.communication.message import StateMessage
from swarm_core.communication.network import Network
from swarm_core.dynamics import step
from swarm_core.initialization import sample_initial_states
from swarm_core.observations import formation_pose_features


class Simulation:
    def __init__(self, task, communication):
        self.task = deepcopy(task)
        self.communication = deepcopy(communication)
        self.dt = task["simulation"]["time_step_s"]
        frequency = communication["transmission_frequency_hz"]

        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Time step must be positive and finite.")
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("Transmission frequency must be positive.")

        interval = 1.0 / (frequency * self.dt)
        if interval < 1 or not math.isclose(
            interval, round(interval), abs_tol=1e-9
        ):
            raise ValueError("Transmission period must match control steps.")

        self.transmission_steps = round(interval)
        self.reset()

    def reset(self, seed=None):
        seed = self.task["seed"] if seed is None else seed
        self.states = sample_initial_states(self.task, seed)
        self.rosbot_curvature = 0.
        self.measurement_rng = np.random.default_rng(np.random.SeedSequence([seed, 771]))
        self.position_bias = np.zeros((len(self.states), 2))
        self.heading_bias = np.zeros(len(self.states))
        self._measure()
        self.step_index = 0
        self.sequence = 0
        self.formation_ready = [False]*len(self.states)
        self.start_proposals = [None]*len(self.states)
        self.start_committed = [False]*len(self.states)
        self.reset_network(seed)

    def reset_network(self, seed):
        """Rebuild episode-start queues after a curriculum state replacement."""
        if self.step_index != 0:
            raise RuntimeError("Network reset is permitted only at episode start.")
        self.sequence = 0
        self.network = Network(
            num_agents=len(self.states),
            delay_s=self.communication["delay_s"],
            loss_probability=self.communication["loss_probability"],
            max_age_s=self.communication["max_information_age_s"],
            seed=seed,
            channel_options=self.communication.get("channel_options", {}),
        )
        self._transmit()
        self.network.deliver(self.time_s)

    @property
    def time_s(self):
        return self.step_index * self.dt

    def _transmit(self):
        for robot_id, state in enumerate(self.estimates):
            message = StateMessage(
                sender_id=robot_id,
                sequence=self.sequence,
                generated_at_s=self.time_s,
                x=state.x,
                y=state.y,
                heading=state.heading,
                speed=state.speed,
                yaw_rate=state.yaw_rate,
                payload_size_bytes=self.communication["payload_size_bytes"]+(8 if self.task.get("history_startup") else 0),
                start_at_s=self.start_proposals[robot_id],
                start_committed=self.start_committed[robot_id],
                formation_ready=getattr(self, "formation_ready", [False]*len(self.states))[robot_id],
            )
            self.network.broadcast(message, self.time_s)
        self.sequence += 1

    def advance(self, actions):
        """Apply all actions simultaneously, then transmit and deliver."""
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (len(self.states), 2):
            raise ValueError("Expected one two-component action per robot.")
        if not np.isfinite(actions).all():
            raise ValueError("Actions must be finite.")

        if self.task.get("mixed_command_limits"):
            from swarm_core.mixed_commands import constrain_mixed_commands

            limits = self.task["robot_defaults"]
            actions = np.clip(actions, -1., 1.)
            forward = limits["max_forward_speed_mps"]
            reverse = limits["max_reverse_speed_mps"]
            yaw = limits["max_yaw_rate_radps"]
            scales = np.column_stack((
                np.where(actions[:, 0] >= 0, forward, reverse),
                np.full(len(actions), yaw),
            ))
            commands = np.asarray(constrain_mixed_commands(actions * scales))
            actions = commands / scales

        steering = self.task.get("rosbot_steering_time_constant_s")
        if steering is not None:
            if not self.task.get("mixed_command_limits"):
                raise ValueError("RosBot steering response requires mixed command limits.")
            from swarm_core.mixed_commands import update_rosbot_curvature

            self.rosbot_curvature = update_rosbot_curvature(
                self.rosbot_curvature, *commands[1], self.dt, steering
            )
        self.states = [
            step(state, action, self.task["robot_defaults"], self.dt,
                 steering_curvature=self.rosbot_curvature if steering is not None and i == 1 else None)
            for i, (state, action) in enumerate(zip(self.states, actions))
        ]
        self.step_index += 1
        self._measure()

        if self.step_index % self.transmission_steps == 0:
            self._transmit()

        self.network.deliver(self.time_s)

    def observe(self, robot_id):
        """Build raw local inputs using only this robot's receiver cache."""
        from swarm_core.observations import (
            build_local_features,
            build_neighbor_observation,
        )

        if type(robot_id) is not int or not 0 <= robot_id < len(self.states):
            raise ValueError("Unknown robot ID.")

        own_state = self.estimates[robot_id]
        neighbors = [
            i for i in range(len(self.states)) if i != robot_id
        ]

        observation = build_neighbor_observation(
            own_state=own_state,
            receiver=self.network.receivers[robot_id],
            neighbor_ids=neighbors,
            now_s=self.time_s,
        )
        if self.task.get("observation_contract") in ("formation_history_v5","formation_history_risk_v6"):
            from swarm_core.communication.history import build_history_observation
            observation = build_history_observation(own_state, self.network.receivers[robot_id],
                neighbors, self.time_s, self.task["team"]["desired_pair_distance_m"],
                self.task["peer_history"])
        observation["local_features"] = build_local_features(
            own_state,
            self.task["robot_defaults"],
            self.task["team"]["desired_pair_distance_m"],
        )
        if self.task.get("formation_pose_goal"):
            observation["local_features"] = np.concatenate((
                observation["local_features"],
                formation_pose_features(own_state.heading, self.task['mission']['direction_world']),
            )).astype(np.float32)
        if self.task.get("formation_capabilities"):
            from swarm_core.mixed_commands import FORMATION_CAPABILITIES
            observation["local_features"] = np.concatenate((
                observation["local_features"], FORMATION_CAPABILITIES[robot_id],
            )).astype(np.float32)
        if self.task.get('observation_contract') in ('formation_geometry_v2','formation_geometry_critic_v3','formation_relational_v4'):
            observation["neighbor_features"] = append_formation_geometry(
                observation["neighbor_features"], observation["neighbor_mask"],
                self.task["team"]["desired_pair_distance_m"],
            )
        return observation

    def observe_all(self):
        """Return separate observations for every agent."""
        return [self.observe(i) for i in range(len(self.states))]
    def _measure(self):
        """Sample once per step; reuse for own observation and outgoing state."""
        settings = self.task.get('measurement', {})
        position_std = settings.get('position_noise_std_m', 0.)
        heading_std = settings.get('heading_noise_std_rad', 0.)
        position_drift = settings.get('position_drift_std_m_per_sqrt_s', 0.)
        heading_drift = settings.get('heading_drift_std_rad_per_sqrt_s', 0.)
        if not np.isfinite([position_std,heading_std,position_drift,heading_drift]).all() or min(position_std,heading_std,position_drift,heading_drift)<0:
            raise ValueError('Measurement scales must be finite and nonnegative.')
        self.position_bias += self.measurement_rng.normal(0,position_drift*np.sqrt(self.dt),self.position_bias.shape)
        self.heading_bias += self.measurement_rng.normal(0,heading_drift*np.sqrt(self.dt),self.heading_bias.shape)
        noise=self.measurement_rng.normal(0,position_std,self.position_bias.shape)
        angles=self.measurement_rng.normal(0,heading_std,self.heading_bias.shape)
        self.estimates=[replace(state,x=state.x+self.position_bias[i,0]+noise[i,0],
            y=state.y+self.position_bias[i,1]+noise[i,1],
            heading=(state.heading+self.heading_bias[i]+angles[i]+np.pi)%(2*np.pi)-np.pi)
            for i,state in enumerate(self.states)]
