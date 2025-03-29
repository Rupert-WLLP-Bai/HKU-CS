cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Set the project name
project(tinyml)

# Ensure all components are included (remove minimal build)
# idf_build_set_property(MINIMAL_BUILD ON)  <-- Remove this line
