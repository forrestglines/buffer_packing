cmake_minimum_required(VERSION 3.10)
project(view_of_views_testing LANGUAGES CXX CUDA)

# Options
#option(DISABLE_MPI "MPI is enabled by default if found, set this to True to disable MPI" OFF)
#option(DISABLE_OPENMP "OpenMP is enabled by default if found, set this to True to disable OpenMP" OFF)
#option(DISABLE_HDF5 "HDF5 is enabled by default if found, set this to True to disable HDF5" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR
      "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
      "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# If the user doesn't specify a build type, prefer Release
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(Kokkos_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos)
add_subdirectory(${Kokkos_ROOT} Kokkos)


#if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/CMakeLists.txt)
#  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon parthenon)
#else()
#  find_package(parthenon REQUIRED)
#endif()

add_subdirectory(tests)

