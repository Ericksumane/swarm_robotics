"""Odometry-based reference for an aligned triangle with an Ackermann robot."""

import math

from check_rosbot_formation_path import advance, find_path
from swarm_core.mixed_commands import GO2_YAW_DEADBAND, ROSBOT_TURN_RADIUS


ROSBOT_CENTER_MARGIN_M = .65


def pose(state):
    return state["x"], state["y"], state["heading"]


def stopped(state):
    return abs(state["speed"]) <= .03 and abs(state["yaw_rate"]) <= .05


def hold_pose_ready(states, destinations, *, pair_tolerance_m=.10):
    """Check the nominal terminal position, heading and formation tolerances."""
    return (all(math.dist(pose(s)[:2], goal[:2]) <= .20
                and abs(math.remainder(s["heading"] - goal[2], math.tau)) <= math.radians(5)
                for s, goal in zip(states, destinations))
            and all(abs(math.dist(pose(a)[:2], pose(b)[:2]) - 1.2) <= pair_tolerance_m
                    for i, a in enumerate(states) for b in states[i+1:]))


def hold_braking_risk(states, command, deceleration, *, recovery_slack_m=0.):
    """Project one control tick and braking using measured WheelTec curvature."""
    own = states[1]
    current = pose(own)
    speed = own["speed"]
    curvature = own["yaw_rate"] / speed if abs(speed) > .005 else 0.
    curvature = max(-1. / ROSBOT_TURN_RADIUS, min(1. / ROSBOT_TURN_RADIUS, curvature))
    peak = max(abs(speed), abs(command[0]))
    distance = math.copysign(peak * .1 + peak**2 / (2. * deceleration),
                             speed if abs(speed) > .005 else command[0])
    for i in (0, 2):
        anchor = pose(states[i])[:2]
        initial = abs(math.dist(current[:2], anchor) - 1.2)
        predicted = max(abs(math.dist(advance(current, distance * f, curvature)[:2],
                                      anchor) - 1.2) for f in (.25, .5, .75, 1.))
        if predicted > .095 and predicted > initial + recovery_slack_m:
            return True
    return False


class CooperativePoseReference:
    def __init__(self, goals, area, maneuver_speed_scale=1., precision=False, *,
                 wheeltec_heading_tolerance_rad=math.radians(5),
                 wheeltec_reversal_deceleration_mps2=None, wheeltec_path_pair_tolerance_m=None,
                 wheeltec_goal_pair_tolerance_m=None):
        self.wheeltec_heading_tolerance_rad = wheeltec_heading_tolerance_rad
        self.wheeltec_reversal_deceleration_mps2 = wheeltec_reversal_deceleration_mps2
        self.wheeltec_path_pair_tolerance_m = wheeltec_path_pair_tolerance_m
        self.wheeltec_goal_pair_tolerance_m = wheeltec_goal_pair_tolerance_m
        self.anchor_pose_actors = {}
        self.demonstration_observer = None
        self.wheeltec_actor = None
        self.wheeltec_actor_uses_vertex = False
        self.wheeltec_replan_margin_m = ROSBOT_CENTER_MARGIN_M
        self.wheeltec_clearance_recovery = False
        self.wheeltec_features = None
        self.reversal_pending = False
        self.reversal_stops = 0
        self.arc_rejections = 0
        if not 0. < wheeltec_heading_tolerance_rad <= math.pi:
            raise ValueError("WheelTec heading tolerance must be in (0, pi].")
        if wheeltec_reversal_deceleration_mps2 is not None and (
                not math.isfinite(wheeltec_reversal_deceleration_mps2)
                or wheeltec_reversal_deceleration_mps2 <= 0.):
            raise ValueError("WheelTec reversal deceleration must be positive and finite.")
        self.goals = goals
        self.area = area
        self.maneuver_speed_scale = maneuver_speed_scale
        self.precision = precision
        self.path = []
        self.path_uses_measured_geometry = False
        self.segment_start = None
        self.replans = 0
        self.aligning = [False] * len(goals)
        self.plan_cooldown = 0
        self.failed_plan_anchors = None
        self.failed_plan_in_position = False

    def point_command(self, index, state, goal, *, local_feedback=False):
        x, y, heading = pose(state)
        dx, dy = goal[0] - x, goal[1] - y
        distance = math.hypot(dx, dy)
        enter_distance = .04
        exit_distance = .065 if index == 0 else .05
        if self.precision:
            enter_distance, exit_distance = .022, .032
        if local_feedback:
            # Turning the Go2 can shift its body by a few centimeters.
            exit_distance = .05 if self.precision else .09
        limit = exit_distance if self.aligning[index] else enter_distance
        self.aligning[index] = distance <= limit
        if self.aligning[index]:
            error = math.remainder(goal[2] - heading, math.tau)
            return self.anchor_command(index, dx, dy, heading, goal, [0., self.turn_command(index, error)])

        error = math.remainder(math.atan2(dy, dx) - heading, math.tau)
        direction = 1.
        if abs(error) > math.pi / 2:
            direction = -1.
            error = math.remainder(error + math.pi, math.tau)
        speed_limit = (.25 if direction > 0 else .15) * self.maneuver_speed_scale
        speed = min(speed_limit, max(.035, distance * .8))
        forward = direction * speed if abs(error) < .25 else 0.
        return self.anchor_command(index, dx, dy, heading, goal,
                                   [forward, self.turn_command(index, error)])

    def anchor_command(self, index, dx, dy, heading, goal, reference):
        actor = self.anchor_pose_actors.get(index)
        if (actor is None and self.demonstration_observer is None) or not any(reference):
            return reference
        c, s = math.cos(heading), math.sin(heading)
        error = goal[2] - heading
        features = [c*dx+s*dy, -s*dx+c*dy, math.sin(error),
                    math.cos(error), float(self.aligning[index])]
        if self.demonstration_observer is not None:
            self.demonstration_observer(index, features, [abs(v) for v in reference])
        if actor is None:
            return reference
        speed, yaw = actor(features)
        v, w = reference
        return [
            math.copysign(max(.035, min(speed, (.25 if v > 0 else .15) * self.maneuver_speed_scale)), v) if v else 0.,
            math.copysign(max(.09 if index == 0 else .04, min(yaw, .35 if index == 0 else .50)), w) if w else 0.,
        ]

    @staticmethod
    def turn_command(index, error):
        if abs(error) <= .035:
            return 0.
        minimum = GO2_YAW_DEADBAND + .01 if index == 0 else .04
        maximum = .35 if index == 0 else .5
        return math.copysign(min(maximum, max(minimum, 1.5 * abs(error))), error)

    def command(self, states, *, go2_local_state=None, wheeltec_local_state=None, jackal_local_state=None):
        self.wheeltec_features = None
        commands = [[0., 0.] for _ in states]
        anchor_ids = (0, 2)
        anchors_ready = True
        anchors_in_position = True
        for index in anchor_ids:
            state, goal = states[index], self.goals[index]
            if index == 0 and go2_local_state is not None:
                state = go2_local_state
            if index == 2 and jackal_local_state is not None:
                state = jackal_local_state
            commands[index] = self.point_command(
                index, state, goal, local_feedback=index == 0 and go2_local_state is not None)
            anchors_ready &= commands[index] == [0., 0.] and stopped(state)
            anchors_in_position &= (
                self.aligning[index] and abs(state["speed"]) <= .03
            )

        self.plan_cooldown = max(0, self.plan_cooldown - 1)
        own = states[1] if wheeltec_local_state is None else wheeltec_local_state
        current = pose(own)
        goal = self.goals[1]
        position_tolerance = .040 if self.precision else .065
        if self.precision and wheeltec_local_state is not None:
            position_tolerance = .055
        heading_tolerance = self.wheeltec_heading_tolerance_rad
        anchor_positions = [pose(states[index])[:2] for index in anchor_ids]
        goal_pair_tolerance = self.wheeltec_goal_pair_tolerance_m
        acceptance_pair_tolerance = (.065 if self.precision else .10) if goal_pair_tolerance is None else goal_pair_tolerance
        planning_pair_tolerance = (.055 if self.precision else .085) if goal_pair_tolerance is None else goal_pair_tolerance
        pairs_ok = all(abs(math.dist(current[:2], anchor) - 1.2) <= acceptance_pair_tolerance
                       for anchor in anchor_positions)
        if (not self.precision and anchors_ready and stopped(own)
                and math.dist(current[:2], goal[:2]) <= position_tolerance
                and abs(current[2]) <= heading_tolerance and not pairs_ok):
            self.path.clear()
            return commands, "geometry_mismatch"
        if ((pairs_ok or not anchors_in_position)
                and math.dist(current[:2], goal[:2]) <= position_tolerance
                and abs(current[2]) <= heading_tolerance):
            self.path.clear()
            self.wheeltec_clearance_recovery = False
            return commands, "settled" if anchors_ready and stopped(own) else None
        deceleration = self.wheeltec_reversal_deceleration_mps2
        if self.reversal_pending:
            if not stopped(own):
                return commands, None
            self.reversal_pending = False
            self.segment_start = current
        if self.path:
            travel, curvature = self.path[0]
            expected = advance(self.segment_start, travel, curvature)
            turn = travel * curvature
            if curvature:
                delta = math.remainder(current[2] - self.segment_start[2], math.tau)
                progress = math.copysign(1., travel) * delta / curvature
                tolerance = .004 / abs(curvature)
            else:
                dx, dy = (a - b for a, b in zip(current[:2], self.segment_start[:2]))
                heading = self.segment_start[2]
                progress = math.copysign(1., travel) * (dx*math.cos(heading)+dy*math.sin(heading))
                tolerance = .003
            remaining = abs(travel) - progress
            next_reverses = len(self.path) > 1 and travel * self.path[1][0] < 0.
            if deceleration is not None:
                traveled = math.copysign(max(0., min(abs(travel), progress)), travel)
                expected_now = advance(self.segment_start, traveled, curvature)
                if progress < -.04 or math.dist(current[:2], expected_now[:2]) > .06:
                    self.path.clear()
                    self.arc_rejections += 1
                    return commands, None
            if remaining <= tolerance:
                self.path.pop(0)
                self.segment_start = current
                if math.dist(current[:2], expected[:2]) > .06:
                    self.path.clear()
                if deceleration is not None and next_reverses:
                    self.reversal_pending = True
                    self.reversal_stops += 1
                    return commands, None
            elif deceleration is not None and next_reverses:
                stopping_distance = own["speed"] ** 2 / (2. * deceleration)
                stopping_distance += .1 * abs(own["speed"])
                if not stopped(own) and remaining <= stopping_distance:
                    return commands, None
        if self.path and anchors_in_position and not self.path_uses_measured_geometry:
            self.path.clear()
        if not self.path:
            if not stopped(own) or self.plan_cooldown:
                return commands, None
            if (not anchors_ready and self.failed_plan_anchors is not None
                    and self.failed_plan_in_position == anchors_in_position):
                moved = max(math.dist(old, new) for old, new in
                            zip(self.failed_plan_anchors, anchor_positions))
                if moved < .12:
                    return commands, None
            # Keep the route clear of both current and eventual anchor positions.
            anchors = anchor_positions
            planning_margin = self.wheeltec_replan_margin_m
            if self.wheeltec_clearance_recovery:
                # Permit retreat from a pose just inside the predictive margin.
                # The live guard still rejects closing motion.
                planning_margin = min(planning_margin, *(math.dist(current[:2], p) for p in anchors))
            else:
                anchors = anchors + [self.goals[index][:2] for index in anchor_ids]
            precision_approach = anchors_in_position and math.dist(current[:2], goal[:2]) < .45
            # An angular grid coarser than twice the tolerance can straddle
            # the target heading. Refine the recovery grid before searching.
            fine_stride = min(.045 if self.wheeltec_path_pair_tolerance_m is not None else .09,
                              1.8 * ROSBOT_TURN_RADIUS * heading_tolerance)
            coarse_stride = .18
            heading_step = coarse_stride / ROSBOT_TURN_RADIUS
            coarse_heading_error = abs(math.remainder(current[2], heading_step))
            stride = fine_stride if precision_approach or coarse_heading_error > heading_tolerance else coarse_stride
            result = find_path(
                current, goal[:2], anchors, self.area,
                position_tolerance=position_tolerance,
                heading_tolerance=heading_tolerance,
                minimum_center_distance=planning_margin,
                stride=stride,
                goal_anchors=anchor_positions if anchors_in_position else (),
                path_pair_tolerance=self.wheeltec_path_pair_tolerance_m,
                goal_pair_tolerance=planning_pair_tolerance,
                max_expansions=80000 if anchors_ready else 3000,
            )
            if result is None and anchors_in_position and stride != fine_stride:
                result = find_path(
                    current, goal[:2], anchors, self.area, stride=fine_stride,
                    position_tolerance=position_tolerance,
                    heading_tolerance=heading_tolerance,
                    minimum_center_distance=planning_margin,
                    goal_anchors=anchor_positions,
                    path_pair_tolerance=self.wheeltec_path_pair_tolerance_m,
                    goal_pair_tolerance=planning_pair_tolerance,
                    max_expansions=80000 if anchors_ready else 3000,
                )
            if (result is None and anchors_in_position
                    and (self.wheeltec_path_pair_tolerance_m is not None
                         or (self.precision and wheeltec_local_state is not None))):
                result = find_path(
                    current, goal[:2], anchors, self.area, stride=.03,
                    position_tolerance=position_tolerance,
                    heading_tolerance=heading_tolerance,
                    minimum_center_distance=planning_margin,
                    goal_anchors=anchor_positions,
                    path_pair_tolerance=self.wheeltec_path_pair_tolerance_m,
                    goal_pair_tolerance=planning_pair_tolerance,
                    max_expansions=12000,
                )
            if result is None or not result[0]:
                self.plan_cooldown = 10
                self.failed_plan_anchors = anchor_positions
                self.failed_plan_in_position = anchors_in_position
                return commands, "unresolved" if anchors_ready else None
            self.failed_plan_anchors = None
            self.path_uses_measured_geometry = anchors_in_position
            self.path = list(result[0])
            self.segment_start = current
            self.replans += 1

        travel, curvature = self.path[0]
        speed_limit = .06 if self.wheeltec_path_pair_tolerance_m is not None else .15
        speed = math.copysign(speed_limit * self.maneuver_speed_scale, travel)
        endpoint = advance(self.segment_start, travel, curvature)
        dx, dy = endpoint[0] - current[0], endpoint[1] - current[1]
        c, s = math.cos(current[2]), math.sin(current[2])
        heading_error = math.remainder(endpoint[2] - current[2], math.tau)
        self.wheeltec_features = [c*dx+s*dy, -s*dx+c*dy, heading_error,
                                  math.copysign(1., travel), curvature,
                                  own["speed"], own["yaw_rate"]]
        if self.wheeltec_actor_uses_vertex:
            dx, dy = goal[0] - current[0], goal[1] - current[1]
            vertex_heading_error = math.remainder(goal[2] - current[2], math.tau)
            self.wheeltec_features.extend([c * dx + s * dy, -s * dx + c * dy,
                                           vertex_heading_error])
        if self.demonstration_observer is not None:
            self.demonstration_observer(1, self.wheeltec_features, [speed, curvature])
        if self.wheeltec_actor is not None:
            learned_speed, learned_curvature = self.wheeltec_actor(self.wheeltec_features)
            speed = math.copysign(min(abs(speed), max(.10, abs(learned_speed))), travel)
            curvature = min(curvature + .03, max(curvature - .03, learned_curvature))
            curvature = min(1. / ROSBOT_TURN_RADIUS, max(-1. / ROSBOT_TURN_RADIUS, curvature))
        if self.wheeltec_actor_uses_vertex and len(self.path) == 1:
            # Brake at the final arc endpoint instead of crossing it at cruise speed.
            dx, dy = endpoint[0] - current[0], endpoint[1] - current[1]
            if math.hypot(dx, dy) < .08:
                speed = math.copysign(min(abs(speed), max(.035, math.hypot(dx, dy))), speed)
        # A short planned arc may be clear at a lower speed over the same horizon.
        for candidate in (speed, .5 * speed, .25 * speed):
            projected_speed = own["speed"]
            if candidate * projected_speed <= 0. or abs(candidate) > abs(projected_speed):
                projected_speed = candidate
            if self.wheeltec_arc_clear(current, projected_speed, curvature, states, anchor_ids):
                commands[1] = [candidate, candidate * curvature]
                return commands, None

        self.path.clear()
        self.failed_plan_anchors = anchor_positions
        self.failed_plan_in_position = anchors_in_position
        self.plan_cooldown = 10
        return commands, None

    @staticmethod
    def wheeltec_arc_clear(current, speed, curvature, states, anchor_ids):
        for horizon in (.2, .4, .6, .8):
            predicted = advance(current, speed * horizon, curvature)
            for index in anchor_ids:
                anchor = states[index]
                measured = pose(anchor)
                projected = advance(measured, anchor["speed"] * horizon, 0.)
                if min(math.dist(predicted[:2], point[:2])
                       for point in (measured, projected)) < ROSBOT_CENTER_MARGIN_M:
                    return False
        return True


class CooperativeTranslationReference:
    """Translate measured formation vertices along world +X and hold."""

    distance_m = 5.0
    command_limit_mps = 1.60
    acceleration_mps2 = .16
    braking_mps2 = .22

    def __init__(self, states, dt=.1, group_speed_mps=.34, adaptive_speed=False, smooth_entry=False,
                 terminal_recovery=False):
        self.starts = [(state["x"], state["y"]) for state in states]
        self.goals = self.translation_goals(self.distance_m)
        self.dt = dt
        self.group_speed_mps = group_speed_mps
        self.adaptive_speed = adaptive_speed
        self.smooth_entry = smooth_entry
        self.terminal_recovery = terminal_recovery
        self.reference_progress = 0.
        self.reference_speed = 0.
        self.filtered_speeds = [0.] * len(states)
        self.speed_correction = [0.] * len(states)
        self.speed_actor = None

    def translation_goals(self, distance_m):
        """Fit an equilateral triangle to the measured vertices, preserving centroid."""
        center_x = sum(x for x, _ in self.starts) / len(self.starts)
        center_y = sum(y for _, y in self.starts) / len(self.starts)
        offsets = [(x - center_x, y - center_y) for x, y in self.starts]
        indices = sorted(range(3), key=lambda index: math.atan2(
            offsets[index][1], offsets[index][0],
        ))
        radius = 1.20 / math.sqrt(3.)
        template = [(radius * math.cos(math.tau * i / 3.),
                     radius * math.sin(math.tau * i / 3.)) for i in range(3)]
        dot = cross = 0.
        for index, (x, y) in zip(indices, template):
            measured_x, measured_y = offsets[index]
            dot += x * measured_x + y * measured_y
            cross += x * measured_y - y * measured_x
        angle = math.atan2(cross, dot)
        cosine, sine = math.cos(angle), math.sin(angle)
        goals = [None] * 3
        for index, (x, y) in zip(indices, template):
            goals[index] = (center_x + distance_m + cosine*x - sine*y,
                            center_y + sine*x + cosine*y)
        return goals

    def command(self, states):
        reference_stop = self.distance_m - .025
        remaining_reference = reference_stop - self.reference_progress
        braking_speed = math.sqrt(2. * self.braking_mps2 * remaining_reference)
        shape_distance = 2.5 if self.smooth_entry else reference_stop
        fraction = min(1., self.reference_progress / shape_distance)
        group_speed = self.group_speed_mps
        if self.adaptive_speed:
            lags = [
                self.reference_progress + fraction * (goal[0] - start[0] - self.distance_m)
                - (state["x"] - start[0])
                for state, start, goal in zip(states, self.starts, self.goals)
            ]
            # A lagging robot slows the moving reference before the gap grows.
            lag = max(lags)
            spread = lag - min(lags)
            speed_reduction = 2.5 * max(0., lag - .015, spread - .025)
            group_speed = max(min(.22, group_speed), group_speed - speed_reduction)
        target_speed = min(group_speed, braking_speed)
        previous_speed = self.reference_speed
        acceleration = self.acceleration_mps2
        if self.smooth_entry and self.reference_progress < .8:
            acceleration = .10
        self.reference_speed = max(
            previous_speed - self.braking_mps2 * self.dt,
            min(target_speed, previous_speed + acceleration * self.dt),
        )
        self.reference_progress = min(
            reference_stop,
            self.reference_progress + (previous_speed + self.reference_speed) * self.dt / 2.,
        )
        fraction = min(1., self.reference_progress / shape_distance)
        commands = []
        for index, (state, (target_x, target_y)) in enumerate(zip(states, self.goals)):
            remaining = target_x - state["x"]
            if remaining <= .04 and self.reference_progress >= self.distance_m - .08:
                yaw = 0. if index == 1 else CooperativePoseReference.turn_command(
                    index, -state["heading"],
                )
                self.speed_correction[index] = 0.
                commands.append([0., yaw])
                continue

            start_x, start_y = self.starts[index]
            desired_x = start_x + self.reference_progress + fraction * (
                target_x - start_x - self.distance_m
            )
            desired = self.reference_speed + 2.8 * (desired_x - state["x"])
            desired = max(0., min(desired, self.group_speed_mps + .14, math.sqrt(.44 * max(0., remaining - .025))))
            measured = state["speed"] * math.cos(state["heading"])
            self.filtered_speeds[index] += .35 * (measured - self.filtered_speeds[index])
            error = desired - self.filtered_speeds[index]
            if desired > 0.:
                integral = self.speed_correction[index] + .75 * self.dt * error
                self.speed_correction[index] = max(-.12, min(.40, integral))
                speed = max(.025, min(self.command_limit_mps,
                            desired + .40 * error + self.speed_correction[index]))
            else:
                self.speed_correction[index] *= .8
                speed = 0.
            if self.speed_actor is not None and desired > 0.:
                features = [desired, error, self.speed_correction[index]]
                speed = max(.025, min(self.command_limit_mps, self.speed_actor(features, speed)))
            if self.terminal_recovery and index == 2 and remaining < .85:
                # Keep the Jackal's integral correction inside the braking envelope.
                speed = min(speed, math.sqrt(.44 * max(0., remaining - .025)))
                self.speed_correction[index] = min(0., self.speed_correction[index])
            # Cross-track feedback approaches the lane while restoring +X heading.
            desired_y = start_y + fraction * (target_y - start_y)
            heading_target = max(-.25, min(.25, 2. * (desired_y - state["y"])))
            error = math.remainder(heading_target - state["heading"], math.tau)
            yaw = max(-.45, min(.45, 2.5 * error))
            commands.append([speed, yaw])
        return commands

    def metrics(self, states):
        position_errors = [math.hypot(state["x"] - x, state["y"] - y)
                           for state, (x, y) in zip(states, self.goals)]
        heading_error = max(abs(state["heading"]) for state in states)
        pair_error = max(
            abs(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) - 1.2)
            for index, a in enumerate(states) for b in states[index + 1:]
        )
        return {
            "translation_m": sum(state["x"] - start[0]
                                 for state, start in zip(states, self.starts)) / len(states),
            "max_destination_error_m": max(position_errors),
            "max_heading_error_deg": math.degrees(heading_error),
            "max_pair_error_m": pair_error,
            "settled": max(position_errors) <= .20 and heading_error <= math.radians(5)
            and pair_error <= .10 and all(stopped(state) for state in states),
        }
