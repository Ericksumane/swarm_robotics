"""Check fresh formation input parity with a rotated mission direction."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np

from swarm_core.deployment.local_runtime import LocalMissionRuntime
from swarm_core.external_simulation import ExternalSimulation
from swarm_core.learning.mappo_adapter import pack
from swarm_core.learning.staged_policy import BatchedMissionPolicy
from swarm_core.mission_env import MissionEnv
from swarm_core.simulation import Simulation


def main():
    with open("configs/tasks/history_full_mission.json") as stream:
        mission_task = json.load(stream)
    with open("configs/communication/nominal.json") as stream:
        communication = json.load(stream)

    mission_task["team"]["desired_pair_distance_m"] = 1.2
    mission_task["mission"]["direction_world"] = [0., 1.]
    mission_task["observation_contract"] = "mission_local_goal_v3"
    for local_dim in (9, 13, 15):
        task = deepcopy(mission_task)
        task.update(formation_capabilities=local_dim >= 13,
                    formation_pose_goal=local_dim == 15)
        training_task = deepcopy(task)
        training_task["observation_contract"] = "formation_relational_v4"
        training_task.pop("formation_peer_history", None)
        training = Simulation(training_task, communication)
        mission = MissionEnv(task, communication, {})
        training.reset(42)
        mission.reset(42)
        states = [vars(state) for state in training.states]
        mission.simulation = ExternalSimulation(task, communication)
        mission.simulation.initialize_physics(states, 0., 42)
        headings = (.4, -1.2, 2.4)
        for env in (training, mission.simulation):
            env.estimates = [replace(state, heading=heading)
                             for state, heading in zip(env.estimates, headings)]

        policy = BatchedMissionPolicy.__new__(BatchedMissionPolicy)
        policy.fresh_local_dim = local_dim
        expected = pack(training.observe_all())
        observations = mission.observe_all()
        raw = pack(observations)
        assert raw.shape == (3, 36)
        reconstructed = policy._fresh_inputs(raw, observations)
        np.testing.assert_array_equal(reconstructed, expected)

        if local_dim == 15:
            relative = np.asarray(headings) - np.pi / 2
            np.testing.assert_allclose(reconstructed[:, 9:11],
                                       np.stack([np.cos(relative), -np.sin(relative)], axis=1))

        for index, own in enumerate(mission.simulation.estimates):
            runtime = LocalMissionRuntime(task, index, "mission", "session", "map", None)
            runtime.receiver = mission.simulation.network.receivers[index]
            observation = runtime.observation(own, mission.simulation.time_s)
            actual = policy._fresh_inputs(pack([observation]), [observation])
            np.testing.assert_array_equal(actual[0], expected[index])

    print("Formation pose and capabilities match in training and mission inference.")


if __name__ == "__main__":
    main()
