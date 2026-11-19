"""Verify formation geometry using known configurations."""

import math

from swarm_core.dynamics import RobotState
from swarm_core.geometry import formation_metrics


def main():
    triangle = [
        RobotState(x=0, y=0),
        RobotState(x=1, y=0),
        RobotState(x=0.5, y=math.sqrt(3) / 2),
    ]

    result = formation_metrics(
        triangle,
        desired_distance=1.0,
        footprint_radius=0.25,
    )

    assert result["formation_rmse_m"] < 1e-12
    assert math.isclose(result["minimum_clearance_m"], 0.5)
    assert not result["collision"]

    overlap = [
        RobotState(x=0),
        RobotState(x=0.4),
    ]
    assert formation_metrics(overlap, 1.0, 0.25)["collision"]

    print("Equilateral triangle formation error: PASSED")
    print("Circular footprint collision detection: PASSED")


if __name__ == "__main__":
    main()