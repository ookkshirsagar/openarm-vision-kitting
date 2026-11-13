cmake_minimum_required(VERSION 3.8)
project(openarm_kitting)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/kitting_task scripts/isaac_attach_bridge DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_kitting_mock.py TIMEOUT 300)
endif()

ament_package()
