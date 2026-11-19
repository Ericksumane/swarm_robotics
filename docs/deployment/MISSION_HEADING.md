# Mission heading

The cooperative Gazebo reference aligns every robot at yaw zero in the world
frame. It translates the team centroid by 5 m along world +X and checks the
final yaw against the same direction.

For a laboratory deployment, define a shared mission frame whose +X axis
points from a surveyed start mark toward a second mark along the test lane.
Express each robot's base pose in that frame before calculating pair
distances, assigning vertices, or issuing a start. Align each robot's
odometry frame using a measured initial position and yaw or localization in
a shared map. Individual odometry origins can differ.

The mission configuration should carry the shared frame ID, direction, and
distance. Each robot computes its yaw error as the wrapped difference
between the mission yaw and its own base yaw in the shared frame. Verify the
pose transform and its freshness before arming. Wheel or leg odometry and
inertial sensing can propagate the initial pose. Shared-map localization can
correct accumulated drift.

A surveyed geographic heading is an outdoor option. In a ROS ENU frame,
east is yaw zero and north is yaw +pi/2. Verify absolute heading sources and
their calibration on each platform before using them.

- [ROS REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)
- [ROS REP 105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst)

## First laboratory setup

Mark two points A and B on the lane. Set A as the mission origin and point
the mission +X axis from A towards B. The marks define a direction, so their
separation can be chosen independently of the 5 m mission distance. Each
robot's initial heading can differ from the mission heading.

With the robots stationary, measure each base centre and its heading in the
mission frame. Record the corresponding initial odometry pose. Compute the
fixed transform that maps that odometry pose to the measured pose. Publish
poses through this transform into the mission frame, keeping twist in the
body frame. Repeat calibration whenever an odometry source resets its origin.
Verify the transforms against known stationary positions and headings before
relying on them. Initial calibration does not correct later odometry drift.

## Robot identity

Use the same code on each onboard computer with separate configuration.
Assign identity at installation and label the chassis with the matching ID.

| Robot ID | Platform | Namespace |
| --- | --- | --- |
| 0 | Unitree Go2 Edu | /robot_0 |
| 1 | WheelTec RosBot Pro | /robot_1 |
| 2 | Clearpath Jackal J100 | /robot_2 |

The ID selects the platform profile, policy component and namespace. The
mission frame and session identifier are shared. The robot ID stays associated
with its chassis even if the mission assigns a different triangle vertex.
Verify that each ID has exactly one odometry source before running the
mission. While the Jackal is unavailable, reserve ID 2 for its simulator.

The mixed-fleet shadow package uses these same three IDs on a central
computer. The per-robot mixed-fleet motor adapters and the laboratory
odometry transforms still require implementation and physical verification.
