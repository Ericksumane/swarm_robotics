"""Immutable translation target for a leaderless formation mission."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TranslationMission:
    """Common-frame reference and destination, expressed in meters."""

    start: tuple[float, float]
    destination: tuple[float, float]

    @classmethod
    def create(cls, start, direction, distance):
        """Normalize the direction and freeze the destination."""
        start = np.asarray(start, dtype=float)
        direction = np.asarray(direction, dtype=float)

        if start.shape != (2,) or direction.shape != (2,):
            raise ValueError("Start and direction must be 2D vectors.")
        if not np.isfinite(start).all() or not np.isfinite(direction).all():
            raise ValueError("Vectors must be finite.")
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Distance must be positive and finite.")

        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Direction must be nonzero.")

        destination = start + distance * direction / norm
        return cls(tuple(start), tuple(destination))

    def destination_error(self, reference):
        """Return distance from the supplied reference to the destination."""
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (2,) or not np.isfinite(reference).all():
            raise ValueError("Reference must be a finite 2D vector.")
        return float(np.linalg.norm(reference - self.destination))
