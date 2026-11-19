# Initial Planar Motion Model

Implementation: src/swarm_core/dynamics.py

## Scope

Each robot is represented by a planar unicycle model.
This is a synthetic high-level motion approximation.
It does not simulate wheel slip, contacts, joint motion, or Go2 walking.

## State and action

State:
s = (x, y, theta, v, omega)

Units:
- x, y: meters.
- theta: radians.
- v: meters per second.
- omega: radians per second.

The two action components are clipped to [-1, 1].

For the normalized linear action a_v:

v_target = a_v * v_forward_max, when a_v >= 0.
v_target = a_v * v_reverse_max, when a_v < 0.

For the normalized angular action a_w:

omega_target = a_w * omega_max.

## Acceleration limits

Let clip(z, lower, upper) bound z to the given interval.

v_next = v + clip(v_target - v, -a_max * dt, a_max * dt)

omega_next = omega
    + clip(omega_target - omega, -alpha_max * dt, alpha_max * dt)

The updated velocities are used throughout the integration interval.
This is a discrete approximation, not continuous acceleration integration.

## Position integration

Let:

delta = omega_next * dt
theta_mid = theta + delta / 2

Using sinc(z) = sin(pi*z)/(pi*z), with sinc(0) = 1:

distance = v_next * dt * sinc(delta / (2*pi))

x_next = x + distance * cos(theta_mid)
y_next = y + distance * sin(theta_mid)
theta_next = wrap_to_pi(theta + delta)

This integrates constant-twist motion over the interval and remains
numerically stable as angular velocity approaches zero.

## Initial verification

Starting from rest with action [1, 0]:
- Time step: 0.1 s.
- Steps: 100.
- Maximum forward speed: 0.3 m/s.
- Maximum linear acceleration: 0.3 m/s^2.
- Expected final x: 2.865 m.
- Expected final y and heading: zero.

Straight-line position and linear acceleration checks passed.
Turning, reverse motion, and multi-agent behavior require further checks.

## Deployment boundary

The model parameters are engineering assumptions.
Physical robot limits and response must be measured before deployment.
