"""Episode outcomes for the initial formation-and-hold task."""

from copy import deepcopy
import math

from swarm_core.geometry import formation_metrics


class FormationTermination:
    """Evaluate true simulator states without choosing robot actions.

    Call once after each simulation step.
    Success requires consecutive qualifying steps.
    Failure takes precedence over success; success precedes the time limit.
    """

    def __init__(self, task):
        self.task = deepcopy(task)
        self.dt = task["simulation"]["time_step_s"]
        duration = task["mission"]["success_hold_duration_s"]

        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Time step must be positive and finite.")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Hold duration must be positive and finite.")

        self.required_steps = max(1, math.ceil(duration / self.dt - 1e-12))
        self.reset()

    def reset(self):
        self.hold_steps = 0
        self.elapsed_steps = 0
        self.done = False
        self.absolute_speed_sum = 0.
        self.absolute_yaw_sum = 0.
        self.geometry_steps = 0
        self.stopped_steps = 0
        self.qualifying_steps = 0
        self.max_hold_steps = 0
        self.best_pair_error = None
        self.first_geometry_time = None

    def update(self, states):
        if self.done:
            raise RuntimeError("Episode ended; reset before updating again.")
        if len(states) != self.task["team"]["num_agents"]:
            raise ValueError("Unexpected number of robot states.")

        self.elapsed_steps += 1
        limits = self.task["robot_defaults"]
        mission = self.task["mission"]
        area = self.task["operating_area"]
        radius = limits["footprint_radius_m"]

        geometry = formation_metrics(
            states,
            self.task["team"]["desired_pair_distance_m"],
            radius,
        )

        outside = any(
            not (
                area["x_bounds_m"][0] + radius <= s.x
                <= area["x_bounds_m"][1] - radius
                and area["y_bounds_m"][0] + radius <= s.y
                <= area["y_bounds_m"][1] - radius
            )
            for s in states
        )
        velocities = [value for s in states for value in (s.speed, s.yaw_rate)]
        if not all(math.isfinite(value) for value in velocities):
            raise ValueError("Velocities must be finite.")

        self.absolute_speed_sum += sum(abs(s.speed) for s in states)/len(states)
        self.absolute_yaw_sum += sum(abs(s.yaw_rate) for s in states)/len(states)
        geometry_ready = geometry["max_pair_error_m"] <= mission["formation_tolerance_m"]
        stopped = all(
                abs(s.speed) <= mission["stop_linear_speed_tolerance_mps"]
                and abs(s.yaw_rate) <= mission["stop_angular_speed_tolerance_radps"]
                for s in states
            )
        heading_ready = True
        if self.task.get("formation_pose_goal"):
            direction = mission['direction_world']
            target_heading = math.atan2(direction[1], direction[0])
            heading_ready = all(
                abs(math.remainder(state.heading - target_heading, 2 * math.pi))
                <= mission["final_heading_tolerance_rad"]
                for state in states
            )
        ready = geometry_ready and stopped and heading_ready
        self.geometry_steps += int(geometry_ready)
        self.stopped_steps += int(stopped)
        self.qualifying_steps += int(ready)
        error = geometry['max_pair_error_m']
        self.best_pair_error = error if self.best_pair_error is None else min(self.best_pair_error,error)
        if geometry_ready and self.first_geometry_time is None:
            self.first_geometry_time = self.elapsed_steps*self.dt
        self.hold_steps = self.hold_steps + 1 if ready else 0

        self.max_hold_steps = max(self.max_hold_steps,self.hold_steps)

        reason = None
        terminated = False
        truncated = False

        if geometry["collision"]:
            reason, terminated = "collision", True
        elif outside:
            reason, terminated = "outside_area", True
        elif self.hold_steps >= self.required_steps:
            reason, terminated = "success", True
        elif (
            self.elapsed_steps * self.dt
            >= self.task["simulation"]["episode_time_limit_s"] - 1e-12
        ):
            reason, truncated = "time_limit", True

        self.done = terminated or truncated
        failure_stage = None
        if reason == 'time_limit':
            failure_stage = ('never_formed' if self.geometry_steps == 0 else
                'formed_never_stopped' if self.qualifying_steps == 0 else 'hold_not_sustained')
        elif terminated and reason != 'success':
            failure_stage = reason
        return {
            "terminated": terminated,
            "truncated": truncated,
            "reason": reason,
            "hold_time_s": self.hold_steps * self.dt,
            "diagnostics": {
                "elapsed_time_s": self.elapsed_steps*self.dt,
                "mean_abs_speed_mps": self.absolute_speed_sum/self.elapsed_steps,
                "mean_abs_yaw_rate_radps": self.absolute_yaw_sum/self.elapsed_steps,
                "geometry_ready": bool(geometry_ready), "stopped": bool(stopped),
                "geometry_ready_time_s": self.geometry_steps*self.dt,
                "stopped_time_s": self.stopped_steps*self.dt,
                "qualifying_time_s": self.qualifying_steps*self.dt,
                "max_hold_time_s": self.max_hold_steps*self.dt,
                "first_geometry_time_s": self.first_geometry_time,
                "best_max_pair_error_m": self.best_pair_error,
                "final_max_pair_error_m": geometry['max_pair_error_m'],
                "max_abs_speed_mps": max(abs(s.speed) for s in states),
                "max_abs_yaw_rate_radps": max(abs(s.yaw_rate) for s in states),
                "failure_stage": failure_stage,
            },
        }