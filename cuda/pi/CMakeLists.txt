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
set(APP_NAME cuda-pi)
project(${APP_NAME} CUDA CXX)

#-------------------------------------------------------------------------------
set(CMAKE_BUILD_TYPE "Release")

#-------------------------------------------------------------------------------
# Supported CUDA architecture
# 80 : A100
# 86 : RTX3090 / A6000
# 89 : RTX4090 / RTX6000 Ada Generation
#set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
set(CMAKE_CUDA_ARCHITECTURES native)

#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)

#-------------------------------------------------------------------------------
add_executable(${APP_NAME})

target_sources(${APP_NAME} PRIVATE
  main.cu
  ../../common/cuda_helper.cu
  ../../common/kernel_timer.cu
)

target_include_directories(${APP_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/../../common
)

#-------------------------------------------------------------------------------
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/../../bin)
