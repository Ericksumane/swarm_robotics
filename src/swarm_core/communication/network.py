"""Fully connected directed network for the Python training simulator."""

import math

import numpy as np

from .channel import Channel
from .receiver import Receiver


class Network:
    """Manage independent links and one receiver cache per robot.

    This simulator component is not a central deployment server.
    Transmission frequency is controlled by the calling environment.
    """

    def __init__(self, num_agents, delay_s=0.0,
                 loss_probability=0.0, max_age_s=0.5, seed=42, channel_options=None):
        if type(num_agents) is not int or num_agents < 2:
            raise ValueError("At least two agents are required.")

        self.num_agents = num_agents
        self._time = 0.0
        self.receivers = {
            i: Receiver(i, max_age_s) for i in range(num_agents)
        }
        pairs = [
            (i, j)
            for i in range(num_agents)
            for j in range(num_agents)
            if i != j
        ]
        seeds = np.random.SeedSequence(seed).spawn(len(pairs))
        self.links = {
            pair: Channel(
                pair[0], pair[1],
                delay_s=delay_s,
                loss_probability=loss_probability,
                seed=link_seed,
                **(channel_options or {}),
            )
            for pair, link_seed in zip(pairs, seeds)
        }

    def _advance(self, now_s):
        if not math.isfinite(now_s) or now_s < self._time:
            raise ValueError("Time must be finite and nondecreasing.")
        self._time = now_s

    def broadcast(self, message, now_s):
        """Send one state message independently to every other robot."""
        if message.sender_id not in self.receivers:
            raise ValueError("Unknown sender.")
        if message.generated_at_s > now_s:
            raise ValueError("Message generation time is in the future.")
        self._advance(now_s)

        return {
            j: self.links[message.sender_id, j].send(message, now_s)
            for j in range(self.num_agents)
            if j != message.sender_id
        }

    def deliver(self, now_s):
        """Deliver all due messages and return the accepted-message count."""
        self._advance(now_s)
        accepted = 0
        for (_, receiver_id), link in self.links.items():
            for _, message in link.receive(now_s):
                accepted += int(
                    self.receivers[receiver_id].accept(message, now_s)
                )
        return accepted

    def read(self, receiver_id, sender_id, now_s):
        """Read only information actually received by the selected robot."""
        if receiver_id not in self.receivers:
            raise ValueError("Unknown receiver.")
        if sender_id not in self.receivers or sender_id == receiver_id:
            raise ValueError("Sender must be a known neighbor.")
        self._advance(now_s)
        return self.receivers[receiver_id].read(sender_id, now_s)