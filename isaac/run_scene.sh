#!/usr/bin/env bash
# Start the Isaac Sim side of the demo (hardware:=isaac).
#
#   source install/setup.bash                       # the colcon workspace
#   ./src/openarm-vision-kitting/isaac/run_scene.sh [--parts 6] [--seed 3] [--headless] ...
#
# ISAAC_SIM (default ~/isaacsim) points at the Isaac Sim 6.x install. Every argument is passed on to
# kitting_scene.py; the URDF is expanded here from the workspace xacro with hardware:=isaac.
set -euo pipefail

ISAAC_SIM=${ISAAC_SIM:-$HOME/isaacsim}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${AMENT_PREFIX_PATH:?source /opt/ros/jazzy/setup.bash and the workspace install/setup.bash first}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
URDF=$TMP/openarm_kitting.urdf                       # the file name also names the USD cache folder
xacro "$(ros2 pkg prefix --share openarm_kitting_moveit_config)/config/openarm_kitting.urdf.xacro" \
  hardware:=isaac cell:=false > "$URDF"

# Isaac ships its own ROS 2 Jazzy libraries; keep the system Python packages out of its interpreter.
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  AMENT_PREFIX_PATH="$AMENT_PREFIX_PATH" \
  ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  LD_LIBRARY_PATH="$ISAAC_SIM/exts/isaacsim.ros2.core/jazzy/lib" \
  "$ISAAC_SIM/python.sh" "$HERE/kitting_scene.py" --urdf "$URDF" --usd-dir "$TMP/usd" "$@"
