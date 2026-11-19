"""Verify translation distance, direction and fixed destination."""

import math

from swarm_core.mission import TranslationMission


def main():
    mission = TranslationMission.create(
        start=[1.0, 2.0],
        direction=[2.0, 0.0],
        distance=5.0,
    )

    assert mission.start == (1.0, 2.0)
    assert mission.destination == (6.0, 2.0)
    assert math.isclose(mission.destination_error([1, 2]), 5.0)
    assert math.isclose(mission.destination_error([4, 2]), 2.0)
    assert mission.destination_error([6, 2]) == 0.0

    # Checking progress must not move the destination.
    assert mission.destination == (6.0, 2.0)

    try:
        TranslationMission.create([0, 0], [0, 0], 5.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero direction was accepted.")

    print("Translation distance and fixed destination: PASSED")
    print("Invalid direction rejection: PASSED")


if __name__ == "__main__":
    main()