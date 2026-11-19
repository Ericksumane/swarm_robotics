"""Simulate the three odometry links to the cooperative coordinator."""
import math

import numpy as np

from check_rosbot_formation_path import advance
from swarm_core.communication.prediction import predict_received_pose

from swarm_core.communication.channel import Channel
from swarm_core.communication.message import StateMessage
from swarm_core.communication.receiver import Receiver


class CooperativeTelemetry:
    def __init__(self, profile, seed, robot_count):
        self.receiver = Receiver(robot_count, profile["max_information_age_s"])
        self.payload_size = profile["payload_size_bytes"]
        self.period_s = 1 / profile["transmission_frequency_hz"]
        self.sequence = 0
        self.links = [
            Channel(index, robot_count, profile["delay_s"], profile["loss_probability"],
                    seed=link_seed, **profile["channel_options"])
            for index, link_seed in enumerate(np.random.SeedSequence(seed).spawn(robot_count))
        ]
        self.unavailable_ticks = 0
        self.max_used_age_s = 0.0
        self.prediction_age_s = 1.5 if profile["profile_name"] == "severe" else None
        self.prediction_ticks = 0
        self.clearance_stops = 0

    def update(self, states, now_s):
        if now_s + 1e-8 >= self.sequence * self.period_s:
            for link, state in zip(self.links, states):
                link.send(StateMessage(
                    link.sender_id, self.sequence, now_s, state["x"], state["y"],
                    state["heading"], state["speed"], state["yaw_rate"], self.payload_size,
                ), now_s)
            self.sequence += 1

        for link in self.links:
            for _, message in link.receive(now_s):
                self.receiver.accept(message, now_s)

        samples = [self.receiver.read(link.sender_id, now_s) for link in self.links]
        ages = [age for _, age, _ in samples]
        if self.prediction_age_s is None:
            available = all(valid for _, _, valid in samples)
        else:
            available = all(age is not None and age <= self.prediction_age_s + 1e-9
                            for age in ages)
        if not available:
            self.unavailable_ticks += 1
            return None, ages
        self.max_used_age_s = max(self.max_used_age_s, *ages)
        if self.prediction_age_s is not None:
            self.prediction_ticks += int(any(not valid for _, _, valid in samples))
        estimates = []
        for message, age, _ in samples:
            x, y, heading = message.x, message.y, message.heading
            if self.prediction_age_s is not None:
                x, y, heading = predict_received_pose(message, min(age, .5))
            estimates.append(dict(x=x, y=y, heading=heading,
                                  speed=message.speed, yaw_rate=message.yaw_rate))
        return estimates, ages

    def constrain_commands(self, states, ages, commands):
        if self.prediction_age_s is None or states is None:
            return commands
        poses = [(state["x"], state["y"], state["heading"]) for state in states]
        for i, first in enumerate(poses):
            for j in range(i + 1, len(poses)):
                second = poses[j]
                initial = math.dist(first[:2], second[:2])
                # Reserve more clearance as each estimate ages beyond the fresh window.
                margin = .65 + .05 * (max(0., ages[i] - .5) + max(0., ages[j] - .5))
                for horizon in (.2, .4, .6, .8, 1.2, 2.):
                    projected = [
                        self.braking_pose(poses[index], states[index], commands[index], horizon)
                        for index in (i, j)
                    ]
                    distance = math.dist(projected[0][:2], projected[1][:2])
                    if distance < margin and distance < initial - 1e-6:
                        self.clearance_stops += 1
                        return [[0., 0.] for _ in states]
        return commands

    @staticmethod
    def braking_pose(pose, state, command, horizon):
        measured = state["speed"]
        v, w = command
        if abs(measured) > abs(v) or measured * v < 0.:
            v, w = measured, state["yaw_rate"]
        if not v:
            return pose
        deceleration = .30
        reaction = min(horizon, .1)
        braking_time = min(horizon - reaction, abs(v) / deceleration)
        travel = abs(v) * (reaction + braking_time) - .5 * deceleration * braking_time**2
        return advance(pose, math.copysign(travel, v), w / v)

    @classmethod
    def local_clearance_risk(cls, states, commands, minimum_center_distance=.70):
        """Detect a closing pair using measured motion and commanded braking paths."""
        poses = [(state["x"], state["y"], state["heading"]) for state in states]
        pairs = [(i, j, math.dist(poses[i][:2], poses[j][:2]))
                 for i in range(len(poses)) for j in range(i + 1, len(poses))]
        for horizon in (.1, .2, .4, .6, .9, 1.2):
            projected = [cls.braking_pose(pose, state, command, horizon)
                         for pose, state, command in zip(poses, states, commands)]
            if any(math.dist(projected[i][:2], projected[j][:2])
                   < min(minimum_center_distance, initial) - 1e-6
                   for i, j, initial in pairs):
                return True
        return False

    @classmethod
    def filter_local_clearance(cls, states, commands):
        if not cls.local_clearance_risk(states, commands):
            return commands
        # Prefer allowing the staged anchors to clear the WheelTec's path.
        for stopped_ids in ((1,), (2,), (0,), (1, 2), (0, 1), (0, 2)):
            candidate = [([0., 0.] if i in stopped_ids else command)
                         for i, command in enumerate(commands)]
            if candidate != commands and any(v or w for v, w in candidate):
                if not cls.local_clearance_risk(states, candidate):
                    return candidate
        return [[0., 0.] for _ in commands]

    def metrics(self):
        return dict(
            sent=sum(link.sent for link in self.links),
            delivered=sum(link.delivered for link in self.links),
            dropped=sum(link.dropped for link in self.links),
            unavailable_ticks=self.unavailable_ticks,
            max_used_age_s=self.max_used_age_s,
            prediction_ticks=self.prediction_ticks,
            clearance_stops=self.clearance_stops,
        )
