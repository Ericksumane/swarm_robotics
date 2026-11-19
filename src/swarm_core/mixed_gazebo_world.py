"""Build the three-platform Gazebo integration scene."""
from pathlib import Path
import math
import subprocess
import xml.etree.ElementTree as ET

from swarm_core.gazebo_world import build_live_world, inertial

def plugin(model, system, library, fields):
    node = ET.SubElement(model, "plugin", {
        "name": f"gz::sim::systems::{system}",
        "filename": f"gz-sim-{library}-system",
    })
    for name, value in fields:
        ET.SubElement(node, name).text = str(value)

def convert(urdf):
    output = subprocess.check_output(["gz", "sdf", "-p", str(urdf)], text=True)
    return ET.fromstring(output).find("model")

def wheeltec():
    """Pro-sized approximation. Wheelbase and track are provisional."""
    model = ET.fromstring(
        '<model name="wheeltec_pro_approx"><link name="chassis">'
        + inertial(19.54, .61, 1.02, 1.51)
        + '<collision name="body"><pose>0 0 .16 0 0 0</pose>'
        '<geometry><box><size>.774 .50 .134</size></box></geometry></collision>'
        '<visual name="body"><pose>0 0 .16 0 0 0</pose>'
        '<geometry><box><size>.774 .50 .134</size></box></geometry>'
        '<material><ambient>.8 .32 .04 1</ambient><diffuse>.8 .32 .04 1</diffuse>'
        '</material></visual></link></model>'
    )
    for axle, x in (("front", .25), ("rear", -.25)):
        for side, y in (("left", .255), ("right", -.255)):
            name = f"{axle}_{side}"
            parent = "chassis"
            if axle == "front":
                parent = name + "_knuckle"
                model.append(ET.fromstring(
                    f'<link name="{parent}"><pose>{x} {y} .075 0 0 0</pose>'
                    + inertial(.08, .0001, .0001, .0001) + '</link>'
                ))
                model.append(ET.fromstring(
                    f'<joint name="{name}_steer" type="revolute"><parent>chassis</parent>'
                    f'<child>{parent}</child><axis><xyz>0 0 1</xyz>'
                    '<limit><lower>-.7</lower><upper>.7</upper></limit></axis></joint>'
                ))
            geometry = '<geometry><cylinder><radius>.075</radius><length>.06</length></cylinder></geometry>'
            model.append(ET.fromstring(
                f'<link name="{name}_wheel"><pose>{x} {y} .075 0 0 0</pose>'
                + inertial(.45, .001, .002, .001)
                + f'<collision name="tire"><pose>0 0 0 1.5707963 0 0</pose>{geometry}</collision>'
                + f'<visual name="tire"><pose>0 0 0 1.5707963 0 0</pose>{geometry}'
                '<material><ambient>.05 .05 .05 1</ambient><diffuse>.05 .05 .05 1</diffuse>'
                '</material></visual></link>'
            ))
            model.append(ET.fromstring(
                f'<joint name="{name}_wheel_joint" type="revolute"><parent>{parent}</parent>'
                f'<child>{name}_wheel</child><axis><xyz>0 1 0</xyz>'
                '<limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>'
            ))
    plugin(model, "AckermannSteering", "ackermann-steering", [
        ("left_joint", "rear_left_wheel_joint"), ("right_joint", "rear_right_wheel_joint"),
        ("left_steering_joint", "front_left_steer"), ("right_steering_joint", "front_right_steer"),
        ("wheel_separation", .51), ("kingpin_width", .51), ("wheel_base", .5),
        ("wheel_radius", .075), ("steering_limit", math.asin(.5 / 1.02)),
        ("topic", "/model/robot_1/cmd_vel"), ("max_linear_acceleration", .3),
        ("min_linear_acceleration", -.3),
    ])
    return model

def build_mixed_world(out, states, limits):
    root = ET.fromstring(build_live_world(states, step_s=.001, real_time_factor=0, limits=limits))
    world = root.find("world")
    for model in world.findall("model"):
        if model.get("name").startswith("robot_"):
            world.remove(model)
    share = Path(subprocess.check_output(["ros2", "pkg", "prefix", "--share", "go2_description"], text=True).strip())
    go_urdf = out / "go2.urdf"
    robot = ET.fromstring(subprocess.check_output(
        ["xacro", str(share / "xacro/robot.xacro"), "robot_name:=swarm_go2"], text=True
    ))
    # The first integration trial uses simulator odometry and the gait controller.
    for extension in robot.findall("gazebo"):
        for sensor in extension.findall("sensor"):
            extension.remove(sensor)
    for joint in robot.findall("./ros2_control/joint"):
        name = joint.get("name")
        angle = 0.0 if "hip" in name else (1.0 if "upper_leg" in name else -1.9)
        position = joint.find("state_interface[@name='position']")
        ET.SubElement(position, "param", {"name": "initial_value"}).text = str(angle)
    ET.ElementTree(robot).write(go_urdf, encoding="utf-8", xml_declaration=True)
    go2 = convert(go_urdf)
    for gravity in go2.findall("./link/gravity"):
        gravity.text = "true"
    jackal_xacro = out / "jackal.urdf.xacro"
    jackal_xacro.write_text(
        '<robot name="jackal_j100" xmlns:xacro="http://www.ros.org/wiki/xacro">'
        '<xacro:include filename="$(find clearpath_platform_description)/urdf/j100/j100.urdf.xacro"/>'
        '<xacro:j100/></robot>'
    )
    jackal_urdf = out / "jackal.urdf"
    jackal_urdf.write_text(subprocess.check_output(
        ["xacro", str(jackal_xacro), "use_platform_controllers:=false"], text=True
    ).replace("package://clearpath_platform_description/", "file:///opt/ros/jazzy/share/clearpath_platform_description/"))
    jackal = convert(jackal_urdf)
    wheels = [(f"{side}_joint", f"{axle}_{side}_wheel_joint")
              for side in ("left", "right") for axle in ("front", "rear")]
    names = {joint.get("name") for joint in jackal.findall("joint")}
    if not all(name in names for _, name in wheels):
        raise RuntimeError(f"Unexpected Jackal wheel joints: {sorted(names)}")
    plugin(jackal, "DiffDrive", "diff-drive", wheels + [
        ("wheel_separation", .37559), ("wheel_radius", .098),
        ("topic", "/model/robot_2/cmd_vel"), ("max_linear_acceleration", .3),
        ("min_linear_acceleration", -.3),
    ])
    for i, (model, name, z) in enumerate(zip(
        (go2, wheeltec(), jackal), ("go2_edu", "wheeltec_pro_approx", "jackal_j100"),
        (.29, .005, .069)
    )):
        model.set("name", name)
        pose = model.find("pose")
        if pose is None:
            pose = ET.SubElement(model, "pose")
        state = states[i]
        pose.text = f"{state['x']} {state['y']} {z} 0 0 {state['heading']}"
        plugin(model, "OdometryPublisher", "odometry-publisher", [
            ("odom_frame", "world"), ("odom_topic", f"/model/robot_{i}/odometry"),
            ("odom_publish_frequency", 200), ("dimensions", 3),
        ])
        world.append(model)
    return ET.tostring(root, encoding="unicode")
