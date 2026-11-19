"""Receiver cache with sequence ordering and information-age checks."""

import math

from .message import StateMessage


class Receiver:
    """Store the latest accepted message per sender.

    Times are in seconds and must share a common clock basis.
    Sequence numbers must increase per sender within an episode.
    Create a new receiver when resetting an episode.
    """

    def __init__(self, robot_id, max_age_s):
        if type(robot_id) is not int or robot_id < 0:
            raise ValueError("Robot ID must be a nonnegative integer.")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Maximum age must be positive and finite.")

        self.robot_id = robot_id
        self.max_age_s = max_age_s
        self._latest = {}
        self._time = 0.0

    def _advance(self, now_s):
        if not math.isfinite(now_s) or now_s < self._time:
            raise ValueError("Time must be finite and nondecreasing.")
        self._time = now_s

    def accept(self, message: StateMessage, now_s):
        """Reject self messages, duplicates and out-of-order sequences."""
        if message.generated_at_s > now_s:
            raise ValueError("Message generation time is in the future.")
        self._advance(now_s)

        if message.sender_id == self.robot_id:
            return False

        previous = self._latest.get(message.sender_id)
        if previous is not None:
            if message.sequence <= previous.sequence:
                return False
            if message.generated_at_s < previous.generated_at_s:
                return False

        self._latest[message.sender_id] = message
        return True

    def read(self, sender_id, now_s):
        """Return (message, age, valid); never substitute hidden state.

        An expired message remains available for explicit history handling,
        but its validity flag is False. Missing information returns
        (None, None, False).
        """
        self._advance(now_s)
        message = self._latest.get(sender_id)

        if message is None:
            return None, None, False

        age = now_s - message.generated_at_s
        return message, age, age <= self.max_age_s or math.isclose(age,self.max_age_s,rel_tol=0.,abs_tol=1e-9)