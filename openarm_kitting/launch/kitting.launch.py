"""Kitting task, plus the part detector when the parts come from a camera.

  ros2 launch openarm_kitting kitting.launch.py hardware:=isaac        # with isaac/run_scene.sh and moveit.launch.py
  ros2 launch openarm_kitting kitting.launch.py hardware:=mock seed:=3     # simulated detections, no simulator
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context):
    lc = context.launch_configurations
    isaac = lc["hardware"] == "isaac"
    config = os.path.join(get_package_share_directory("openarm_kitting"), "config", "kitting.yaml")
    moveit = (
        MoveItConfigsBuilder("openarm_kitting", package_name="openarm_kitting_moveit_config")
        .robot_description(file_path="config/openarm_kitting.urdf.xacro", mappings={"hardware": lc["hardware"]})
        .robot_description_semantic(file_path="config/openarm_kitting.srdf")
        .joint_limits(file_path="config/joint_limits.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )
    nodes = [Node(package="openarm_kitting", executable="kitting_task", output="screen",
                  parameters=[moveit.to_dict(), {
                      "use_sim_time": isaac, "config": config,
                      "detections": "camera" if isaac else "simulated",
                      "seed": int(lc["seed"]), "parts": int(lc["parts"]), "cycles": int(lc["cycles"]),
                  }])]
    if isaac:
        nodes.append(Node(package="openarm_kitting_perception", executable="part_detector", output="screen",
                          parameters=[config, {"use_sim_time": True, "segmentation": lc["segmentation"]}]))
        # mirrors MoveIt's attach and detach into the simulator
        nodes.append(Node(package="openarm_kitting", executable="isaac_attach_bridge", output="screen",
                          parameters=[{"use_sim_time": True}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="mock", choices=["mock", "isaac"]),
        DeclareLaunchArgument("segmentation", default_value="depth", choices=["depth", "ground_truth"]),
        DeclareLaunchArgument("seed", default_value="1", description="part layout (hardware:=mock)"),
        DeclareLaunchArgument("parts", default_value="4", description="number of parts (hardware:=mock)"),
        DeclareLaunchArgument("cycles", default_value="0", description="pick and place cycles, 0 = until done"),
        OpaqueFunction(function=launch_setup),
    ])
