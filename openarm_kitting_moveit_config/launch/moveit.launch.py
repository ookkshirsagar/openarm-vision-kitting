"""MoveIt 2 + ros2_control for the bimanual OpenArm 2.0 in the kitting cell.

  ros2 launch openarm_kitting_moveit_config moveit.launch.py                  # mock hardware
  ros2 launch openarm_kitting_moveit_config moveit.launch.py hardware:=isaac  # Isaac Sim (isaac/run_scene.sh)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

CONTROLLERS = ["left_arm_controller", "right_arm_controller", "left_gripper_controller", "right_gripper_controller"]


def moveit_config(hardware):
    return (
        MoveItConfigsBuilder("openarm_kitting", package_name="openarm_kitting_moveit_config")
        .robot_description(file_path="config/openarm_kitting.urdf.xacro", mappings={"hardware": hardware})
        .robot_description_semantic(file_path="config/openarm_kitting.srdf")
        .joint_limits(file_path="config/joint_limits.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )


def launch_setup(context):
    hardware = context.launch_configurations["hardware"]
    use_sim_time = hardware == "isaac"
    sim = {"use_sim_time": use_sim_time}
    pkg = get_package_share_directory("openarm_kitting_moveit_config")
    config = moveit_config(hardware)

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[config.robot_description, sim], output="screen"),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[os.path.join(pkg, "config", "ros2_controllers.yaml"), sim],
             remappings=[("~/robot_description", "/robot_description")], output="screen"),
        Node(package="controller_manager", executable="spawner",
             # Isaac Sim publishes /joint_states itself
             arguments=[*([] if hardware == "isaac" else ["joint_state_broadcaster"]), *CONTROLLERS,
                        "--controller-manager-timeout", "60"],
             parameters=[sim], output="screen"),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[config.to_dict(), sim,
                         {"capabilities": "move_group/ExecuteTaskSolutionCapability",
                          "publish_robot_description_semantic": True}]),
    ]
    if context.launch_configurations["rviz"] == "true":
        nodes.append(Node(package="rviz2", executable="rviz2", output="log",
                          arguments=["-d", os.path.join(pkg, "rviz", "moveit.rviz")],
                          parameters=[config.robot_description, config.robot_description_semantic,
                                      config.robot_description_kinematics, config.planning_pipelines,
                                      config.joint_limits, sim]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="mock", choices=["mock", "isaac"]),
        DeclareLaunchArgument("rviz", default_value="true", choices=["true", "false"]),
        OpaqueFunction(function=launch_setup),
    ])
