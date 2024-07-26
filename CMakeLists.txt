#===============================================================================
#  CMakeLists.txt for CUDA Sample Applications
#
#  Copyright 2023-2024 Shogo OKADA (shogo.okada@kek.jp), KEK-CRC, Japan
#
#  ***************************
#  !! DONOT EDIT THIS FILE  !!
#  ***************************
#===============================================================================
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
#===============================================================================
project(PPCC CXX CUDA)

# applications
add_subdirectory(cpp/saxpy)
add_subdirectory(cpp/pi)
add_subdirectory(cuda/saxpy)
add_subdirectory(cuda/pi)
