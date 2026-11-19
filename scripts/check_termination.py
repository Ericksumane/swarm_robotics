"""Check success, hold reset, failures and time-limit truncation."""

from copy import deepcopy
from dataclasses import replace
import json
import math

from swarm_core.dynamics import RobotState
from swarm_core.termination import FormationTermination


def main():
    with open("configs/tasks/formation_v1.json") as file:
        task = json.load(file)

    triangle = [
        RobotState(x=0, y=0),
        RobotState(x=1, y=0),
        RobotState(x=0.5, y=math.sqrt(3) / 2),
    ]
    monitor = FormationTermination(task)

    for _ in range(monitor.required_steps - 1):
        assert not monitor.update(triangle)["terminated"]

    moving = [replace(s, speed=0.1) for s in triangle]
    assert monitor.update(moving)["hold_time_s"] == 0.0

    for _ in range(monitor.required_steps):
        result = monitor.update(triangle)
    assert result["reason"] == "success"
    assert result["terminated"] and not result["truncated"]

    monitor.reset()
    collision = [triangle[0], replace(triangle[1], x=0.1), triangle[2]]
    assert monitor.update(collision)["reason"] == "collision"

    monitor.reset()
    outside = [replace(s, x=s.x + 20) for s in triangle]
    assert monitor.update(outside)["reason"] == "outside_area"

    short = deepcopy(task)
    short["simulation"]["episode_time_limit_s"] = 0.2
    monitor = FormationTermination(short)
    monitor.update(moving)
    result = monitor.update(moving)
    assert result["truncated"] and not result["terminated"]

    try:
        monitor.update(moving)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Post-episode update was accepted.")

    print("Consecutive stationary formation hold: PASSED")
    print("Collision and operating-area failures: PASSED")
    print("Time-limit truncation and post-end guard: PASSED")


if __name__ == "__main__":
    main()