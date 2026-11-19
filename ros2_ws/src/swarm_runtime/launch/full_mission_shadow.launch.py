"""Launch three independent shadow mission nodes on one host."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('project_root'),DeclareLaunchArgument('mission_id'),
        DeclareLaunchArgument('deployment_manifest',default_value=''),
        DeclareLaunchArgument('completion_profile',default_value='nominal'),
        *[Node(package='swarm_runtime',executable='full_mission_node',namespace=f'robot_{i}',parameters=[dict(project_root=LaunchConfiguration('project_root'),mission_id=LaunchConfiguration('mission_id'),deployment_manifest=LaunchConfiguration('deployment_manifest'),completion_profile=LaunchConfiguration('completion_profile'),robot_id=i,common_frame='mission',body_frame=f'robot_{i}/base_link')]) for i in range(3)]])
