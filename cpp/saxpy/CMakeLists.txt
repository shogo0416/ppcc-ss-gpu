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
set(APP_NAME cpp-saxpy)
project(${APP_NAME} CXX)

#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

#-------------------------------------------------------------------------------
add_executable(${APP_NAME})

target_sources(${APP_NAME} PRIVATE main.cc)

target_include_directories(${APP_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/../../common
)

if (PPCC_ENV)
  link_directories(/opt/anaconda3/envs/jupyter/lib)
  target_link_libraries(${APP_NAME} PRIVATE stdc++)
endif()

#-------------------------------------------------------------------------------
install(TARGETS ${APP_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/../../bin)
