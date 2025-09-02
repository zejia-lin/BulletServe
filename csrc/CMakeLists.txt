cmake_minimum_required(VERSION 3.17)
project(SMCtrl LANGUAGES C CXX)

# CXX Flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fno-rtti -Wno-narrowing -Wno-error=narrowing")
message(STATUS "Build type ${CMAKE_BUILD_TYPE}")
message(STATUS "Using ${CMAKE_CXX_COMPILER}")

# CUDA librarires, see https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit REQUIRED)
if(NOT ${CUDAToolkit_FOUND})
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
message(STATUS "Found CUDA Toolkit(${CUDAToolkit_VERSION}) at ${CUDAToolkit_BIN_DIR}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES not defined")
endif()
message(STATUS "Using CUDA Arch ${CMAKE_CUDA_ARCHITECTURES}")

add_library(smctrl SHARED src/libsmctrl_core.c src/libsmctrl_validator.cu)
target_include_directories(smctrl PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(smctrl PRIVATE cuda)
target_link_options(smctrl PRIVATE LINKER:-rpath,${CUDAToolkit_LIBRARY_DIR})
set_target_properties(smctrl PROPERTIES PUBLIC_HEADER src/libsmctrl.h)

add_subdirectory(test)
