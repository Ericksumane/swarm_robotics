"""Verify formation reward against analytically known configurations."""

from dataclasses import replace
import json
import math

from swarm_core.dynamics import RobotState
from swarm_core.rewards import formation_hold_reward


def main():
    with open("configs/tasks/formation_v1.json") as file:
        task = json.load(file)
    with open("configs/training/formation_reward_v1.json") as file:
        weights = json.load(file)

    limits = task["robot_defaults"]
    dt = task["simulation"]["time_step_s"]
    distance = task["team"]["desired_pair_distance_m"]

    triangle = [
        RobotState(x=0.0, y=0.0),
        RobotState(x=distance, y=0.0),
        RobotState(x=distance / 2, y=math.sqrt(3) * distance / 2),
    ]

    def evaluate(states):
        return formation_hold_reward(
            states, limits, distance, dt, weights
        )

    perfect_reward, _ = evaluate(triangle)
    assert math.isclose(perfect_reward, 0.0, abs_tol=1e-12)

    # Doubling every distance produces normalized squared error of one.
    expanded = [
        replace(state, x=2 * state.x, y=2 * state.y)
        for state in triangle
    ]
    expanded_reward, expanded_terms = evaluate(expanded)
    assert math.isclose(
        expanded_terms["formation"],
        -dt * weights["formation_weight"],
        abs_tol=1e-12,
    )
    assert expanded_reward < perfect_reward

    moving = [
        replace(state, speed=limits["max_forward_speed_mps"])
        for state in triangle
    ]
    moving_reward, moving_terms = evaluate(moving)
    assert math.isclose(
        moving_terms["motion"],
        -dt * weights["motion_weight"],
        abs_tol=1e-12,
    )
    assert moving_reward < perfect_reward

    overlapping = [
        triangle[0],
        replace(triangle[1], x=0.1),
        triangle[2],
    ]
    collision_reward, collision_terms = evaluate(overlapping)
    assert collision_terms["collision"] == -weights["collision_penalty"]
    assert math.isclose(
        collision_reward, sum(collision_terms.values()), abs_tol=1e-12
    )

    print("Perfect stationary formation:", perfect_reward)
    print("Expanded formation:", expanded_reward)
    print("Moving formation:", moving_reward)
    print("Collision:", collision_reward)
    print("Formation reward checks: PASSED")


if __name__ == "__main__":
    main()