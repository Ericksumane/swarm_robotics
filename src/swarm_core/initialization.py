"""Seeded initial-state sampling for synthetic circular robot footprints."""

import numpy as np

from swarm_core.dynamics import RobotState


def sample_initial_states(config, seed=None):
    """Sample separated robot centers inside the operating area.

    The initial version uses the same synthetic footprint for all agents.
    Raises ValueError for invalid settings and RuntimeError if sampling fails.
    """
    rng = np.random.default_rng(config["seed"] if seed is None else seed)
    settings = config["initialization"]
    area = config["operating_area"]
    count = config["team"]["num_agents"]
    radius = config["robot_defaults"]["footprint_radius_m"]
    separation = settings["min_center_distance_m"]
    attempts = settings["max_sampling_attempts"]

    if not isinstance(count, int) or count < 1:
        raise ValueError("Agent count must be a positive integer.")
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("Sampling attempts must be a positive integer.")
    if not np.isfinite([radius, separation]).all() or radius <= 0:
        raise ValueError("Footprint and separation must be finite and valid.")
    if separation <= 2 * radius:
        raise ValueError("Initial separation must exceed footprint diameter.")

    def interval(values):
        bounds = np.asarray(values, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all():
            raise ValueError("Ranges must contain two finite bounds.")
        if bounds[0] >= bounds[1]:
            raise ValueError("Range lower bound must be below upper bound.")
        return bounds

    x_area = interval(area["x_bounds_m"])
    y_area = interval(area["y_bounds_m"])
    x_range = interval(settings["position_x_range_m"])
    y_range = interval(settings["position_y_range_m"])
    headings = interval(settings["heading_range_rad"])

    x_low = max(x_range[0], x_area[0] + radius)
    x_high = min(x_range[1], x_area[1] - radius)
    y_low = max(y_range[0], y_area[0] + radius)
    y_high = min(y_range[1], y_area[1] - radius)

    if x_low >= x_high or y_low >= y_high:
        raise ValueError("No valid initialization region inside the area.")

    speed = settings["initial_linear_speed_mps"]
    yaw_rate = settings["initial_angular_speed_radps"]
    limits = config["robot_defaults"]
    if not np.isfinite([speed, yaw_rate]).all():
        raise ValueError("Initial velocities must be finite.")
    if not -limits["max_reverse_speed_mps"] <= speed <= limits["max_forward_speed_mps"]:
        raise ValueError("Initial speed exceeds limits.")
    if abs(yaw_rate) > limits["max_yaw_rate_radps"]:
        raise ValueError("Initial yaw rate exceeds limits.")

    hard=settings.get('hard_examples')
    if hard is not None:
        probability=hard.get('probability',0.)
        if not np.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Hard reset probability must lie in [0,1].')
        if probability>0 and not hard.get('entries'):
            raise ValueError('Hard reset bank must be nonempty.')
        if probability>0 and rng.random()<probability:
            from swarm_core.hard_examples import sample_hard_example
            return sample_hard_example(config,rng,(x_low,x_high),(y_low,y_high),speed,yaw_rate)

    curriculum = settings.get("near_formation", {})
    probability = curriculum.get("probability", 0.0)
    jitter = curriculum.get("position_jitter_m", 0.0)
    if not np.isfinite([probability,jitter]).all() or not 0 <= probability <= 1 or jitter < 0:
        raise ValueError("Invalid near-formation curriculum settings.")
    if 'heading_alignment' in settings and probability != 1.:
        raise ValueError('Heading alignment requires near-formation probability 1.')
    if probability > 0 and rng.random() < probability:
        if count != 3:
            raise ValueError("Near-triangle curriculum requires three agents.")
        distance = config["team"]["desired_pair_distance_m"]
        triangle = np.array([[0.,0.],[distance,0.],[distance/2,np.sqrt(3)*distance/2]])
        triangle -= triangle.mean(axis=0)
        for _ in range(attempts):
            angle=rng.uniform(-np.pi,np.pi)
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            center=np.array([rng.uniform(x_low,x_high),rng.uniform(y_low,y_high)])
            positions=triangle@rotation.T+center+rng.uniform(-jitter,jitter,(3,2))
            if any(not (x_low<=x<=x_high and y_low<=y<=y_high) for x,y in positions):
                continue
            if any(np.linalg.norm(positions[i]-positions[j])<separation for i in range(3) for j in range(i)):
                continue
            minimum_error = curriculum.get("minimum_pair_error_m", 0.)
            if not np.isfinite(minimum_error) or minimum_error < 0:
                raise ValueError("Minimum formation error must be nonnegative and finite.")
            maximum_error = max(abs(np.linalg.norm(positions[i]-positions[j])-distance)
                for i in range(3) for j in range(i))
            if maximum_error < minimum_error:
                continue
            if 'heading_alignment' in settings:
                from swarm_core.curriculum import sample_headings
                angles=sample_headings(positions,distance,settings['heading_alignment'],rng)
            else:
                angles=[rng.uniform(*headings) for _ in positions]
            return [RobotState(float(x),float(y),float(angle),float(speed),float(yaw_rate))
                for (x,y),angle in zip(positions,angles)]
        raise RuntimeError("Could not sample near-formation initial states.")

    if 'heading_alignment' in settings:
        raise ValueError('Heading alignment requires near-formation probability 1.')

    dispersed = settings.get("hard_dispersed", {})
    hard_probability = dispersed.get("probability", 0.)
    if not np.isfinite(hard_probability) or not 0 <= hard_probability <= 1:
        raise ValueError("Hard dispersed probability must be within [0, 1].")
    require_hard_start = hard_probability > 0 and rng.random() < hard_probability
    if require_hard_start:
        hard_separation = dispersed["min_center_distance_m"]
        if count != 3 or not np.isfinite(hard_separation) or hard_separation < separation:
            raise ValueError("Invalid mixed-platform hard-start separation.")
        separation = hard_separation

    states = []
    for _ in range(attempts):
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))

        if any(np.hypot(x - s.x, y - s.y) < separation for s in states):
            continue

        states.append(RobotState(
            x=x,
            y=y,
            heading=float(rng.uniform(*headings)),
            speed=float(speed),
            yaw_rate=float(yaw_rate),
        ))
        if len(states) == count:
            if require_hard_start:
                rosbot = states[1]
                center_dx = (states[0].x + states[2].x) / 2 - rosbot.x
                center_dy = (states[0].y + states[2].y) / 2 - rosbot.y
                facing_toward_team = (
                    np.cos(rosbot.heading) * center_dx
                    + np.sin(rosbot.heading) * center_dy > 0
                )
                if facing_toward_team:
                    states.clear()
                    continue
            return states

    raise RuntimeError("Could not sample all agents within the attempt limit.")
