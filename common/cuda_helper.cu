/*============================================================================
  Copyright (c) 2023-2024 Shogo OKADA (shogo.okada@kek.jp)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
============================================================================*/
#include "cuda_helper.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

namespace {

//------------------------------------------------------------------------------
void err_helper(cudaError_t err)
{
  if (err != cudaSuccess) {
    std::string s(cudaGetErrorString(err));
    throw std::runtime_error("print_device_info::" + s);
  }
}

//------------------------------------------------------------------------------
int get_cuda_cores(int major, int minor, int mp)
{
  auto warning_message = []() {
    std::cout << "[WARNING] Unknown device type..." << std::endl;
  };

  auto num_cores{0};
  switch (major) {
    case 2: // Fermi
      if (minor == 1) {
        num_cores = mp * 48;
      } else {
        num_cores = mp * 32;
      }
      break;
    case 3: // Kepler
      num_cores = mp * 192;
      break;
    case 5: // Maxwell
      num_cores = mp * 128;
      break;
    case 6: // Pascal
      if (minor == 0) {
        num_cores = mp * 64;
      } else if (minor == 1 || minor == 2) {
        num_cores = mp * 128;
      } else {
        warning_message();
      }
      break;
    case 7: // Volta and Turing
      if (minor == 0 || minor == 5) {
        num_cores = mp * 64;
      } else {
        warning_message();
      }
      break;
    case 8: // Ampere and Ada Lovelace
      if (minor == 0) {
        num_cores = mp * 64;
      } else if (minor == 6 || minor == 9) {
        num_cores = mp * 128;
      } else {
        warning_message();
      }
      break;
    default:
      warning_message();
      break;
  }

  return num_cores;
}

} // end of anonymous namespace

//==============================================================================

namespace ppcc {

//------------------------------------------------------------------------------
void print_device_info()
{

  // get device id
  int device_id;
  ::err_helper(cudaGetDevice(&device_id));

  // get device properties
  cudaDeviceProp prop;
  ::err_helper(cudaGetDeviceProperties(&prop, device_id));

  // get global memory info
  size_t mfree, mtotal;
  ::err_helper(cudaMemGetInfo(&mfree, &mtotal));

  // print
  std::stringstream msg;

  msg << "---------------------------------------------------------------------"
      << std::endl;

  msg << "[MESSAGE] GPU Properties" << std::endl;

  msg << "- Device ID: " << device_id << std::endl;

  msg << "- Device Name: " << prop.name << std::endl;

  msg << "- Compute Capability: "
      << prop.major << "." << prop.minor << std::endl;

  msg << "- Multi-Processor Number: " << prop.multiProcessorCount << std::endl;

  msg << "- CUDA Core Number: "
      << ::get_cuda_cores(prop.major, prop.minor, prop.multiProcessorCount)
      << std::endl;

  msg << "- Clock Rate: " << prop.clockRate / 1000.0 << " MHz" << std::endl;

  constexpr double bpgb = 1024.0 * 1024.0 * 1024.0;
  msg << "- Total Global Memory: " << mtotal / bpgb << " GiB ";
  msg << "(Free: " << mfree / bpgb << " GiB)" << std::endl;

  auto ECC = prop.ECCEnabled == 1 ? "True" : "False";
  msg << "- ECC Enabled: " << ECC << std::endl;

  msg << "---------------------------------------------------------------------"
      << std::endl;

  std::cout << msg.str() << std::endl;
}

} // end of namespace ppcc
