"""End-to-end check on mock hardware, headless: the parts of a seeded layout (seed 3: one connector, one block,
one pinion) are each kitted into their slot."""
import re
import unittest

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_testing
import launch_testing.actions
import pytest

TIMEOUT = 240.0          # seconds: up to four plans and executions at the configured velocity scaling
DONE = re.compile(r"done: (\d+) parts placed in [0-9.]+ s: (.*)")


def include(package, launch_file, **args):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare(package), "launch", launch_file])),
        launch_arguments=args.items())


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription([
        include("openarm_kitting_moveit_config", "moveit.launch.py", rviz="false", hardware="mock"),
        # start planning once move_group and the controllers are up
        TimerAction(period=10.0, actions=[include("openarm_kitting", "kitting.launch.py", hardware="mock",
                                                  seed="3", parts="4")]),
        launch_testing.actions.ReadyToTest(),
    ])


class TestKitting(unittest.TestCase):

    def test_all_parts_kitted(self, proc_output):
        proc_output.assertWaitFor(DONE, timeout=TIMEOUT)
        m = next(DONE.search(o.text.decode(errors="replace")) for o in proc_output
                 if DONE.search(o.text.decode(errors="replace")))
        placed = sorted(p.strip() for p in m.group(2).split(","))
        self.assertEqual(int(m.group(1)), 3, m.group(0))
        self.assertEqual(placed, ["block → kit", "connector → kit", "pinion → kit"])
