from setuptools import find_packages, setup

package_name = "openarm_kitting_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Omkar Chandrakant Kshirsagar",
    maintainer_email="105544526+ookkshirsagar@users.noreply.github.com",
    description="Part detection, classification and pose estimation from an RGB-D camera.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "part_detector = openarm_kitting_perception.part_detector_node:main",
        ],
    },
)
