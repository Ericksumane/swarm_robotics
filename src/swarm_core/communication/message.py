"""Immutable state messages exchanged between robots."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StateMessage:
    """Locally estimated robot state expressed in a shared planar frame.

    Coordinates are in meters, angles in radians, and time in seconds.
    Sequence numbers increase per sender during an episode.

    payload_size_bytes is an explicit transmission-budget estimate.
    It is not the Python object's memory size or measured DDS wire size.
    """

    sender_id: int
    sequence: int
    generated_at_s: float
    x: float
    y: float
    heading: float
    speed: float
    yaw_rate: float
    payload_size_bytes: int
    formation_ready: bool = False
    start_at_s: float | None = None
    start_committed: bool = False

    def __post_init__(self):
        if type(self.start_committed) is not bool or (self.start_committed and self.start_at_s is None):
            raise ValueError("Committed start requires a proposal and boolean flag.")
        if self.start_at_s is not None and (not math.isfinite(self.start_at_s) or self.start_at_s<0):
            raise ValueError("Start proposal must be finite and nonnegative.")
        for name in ("sender_id", "sequence"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")

        values = (
            self.generated_at_s,
            self.x,
            self.y,
            self.heading,
            self.speed,
            self.yaw_rate,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Message measurements and time must be finite.")
        if self.generated_at_s < 0:
            raise ValueError("Generation time must be nonnegative.")
        if (
            type(self.payload_size_bytes) is not int
            or self.payload_size_bytes <= 0
        ):
            raise ValueError("Payload size must be a positive integer.")