#===============================================================================
#  CMakeLists.txt for CUDA Sample Applications
#
#  Copyright 2023 Shogo OKADA (shogo.okada@kek.jp), KEK-CRC, Japan
#
#  ***************************
#  !! DONOT EDIT THIS FILE  !!
#  ***************************
#===============================================================================
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
#===============================================================================
set(PPCC_ENV True CACHE BOOL "Run on PPCC VM environment")

if (PPCC_ENV)
  message("-- Build sample codes on the PPCC VM system")
  set(CMAKE_CXX_COMPILER /opt/anaconda3/envs/jupyter/bin/gcc)
  set(CMAKE_CUDA_HOST_COMPILER /opt/anaconda3/envs/jupyter/bin/gcc)
else()
  message("-- Build sample codes on your system")
endif()

project(PPCC CXX CUDA)

# applications
add_subdirectory(cpp/saxpy)
add_subdirectory(cpp/pi)
add_subdirectory(cuda/saxpy)
add_subdirectory(cuda/pi)
