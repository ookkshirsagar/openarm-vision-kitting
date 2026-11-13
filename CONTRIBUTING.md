# Contributing

Issues and pull requests are welcome.

## Setup

- Ubuntu 24.04, ROS 2 Jazzy, MoveIt 2 with MoveIt Task Constructor
- Isaac Sim 6.x, only for `hardware:=isaac`
- Build the workspace as described in the [README](README.md#quick-start)

## Reporting a bug

Open an issue and include:

- the launch commands and their arguments (`hardware`, `seed`, `parts`, `segmentation`)
- what you expected, and what happened instead
- the relevant console output (the `kitting_task`, `part_detector` and `move_group` lines are usually enough)

## Pull requests

1. Fork the repository and create a branch: `git checkout -b feat/short-description`
2. Keep each change focused.
3. Run the tests, and check that the demo still runs on mock hardware:
   ```bash
   colcon build --symlink-install
   colcon test && colcon test-result --verbose
   ```
4. Use [Conventional Commits](https://www.conventionalcommits.org/) messages, for example
   `feat: add a side grasp for tall parts` or `fix: reject detections cut by the image border`.
5. Open the pull request against `main` and describe what changed and how you tested it.

## Code style

- Python: PEP 8, lines up to 120 characters (`setup.cfg`)
- Comments explain why, not what

By contributing, you agree that your contributions are licensed under the [Apache 2.0 License](LICENSE).
