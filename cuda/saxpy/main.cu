/*==============================================================================
  Copyright (c) 2023 Shogo OKADA (shogo.okada@kek.jp)
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
==============================================================================*/
#include <getopt.h>
#include <iostream>
#include <random>
#include <numeric>
#include "lexical_cast.h"
#include "cuda_helper.h"
#include "kernel_timer.h"

namespace {

//------------------------------------------------------------------------------
void print_usage()
{
  const char* usage = R"(
    [Usage] cuda-saxpy <options>
      -h, --help                Print this information

      -g, --gpuid <val>         Set GPU device id
                                [default: 0]

      -s, --seed <val>          Set seed for a random number generator
                                [default: 123456789]

      -n, --num_element <val>   Set the number of elements for vectors
                                [default: 10000000]

      -t, --num_thread <val>    Set the number of threads
                                [default: 128]
  )";

  std::cout << usage << std::endl;
}

//------------------------------------------------------------------------------
__global__
void saxpy(int n, float a, float* x, float* y)
{
  int id = ppcc::get_thread_id();
  if (id < n) { y[id] += a * x[id]; }
}

} // end of anonymous namespace

//==============================================================================
// main function
int main(int argc, char** argv)
{

  struct option opts[] = {
    {"help",         no_argument,       nullptr, 'h'},
    {"gpuid",        required_argument, nullptr, 'g'},
    {"seed",         required_argument, nullptr, 's'},
    {"num_element",  required_argument, nullptr, 'n'},
    {"num_thread",   required_argument, nullptr, 't'},
    {nullptr,        0,                 nullptr, 0}
  };

  int gpuid{0};              // GPU device identifier
  size_t seed{123456789};    // seed for random number generator
  int num_element{10000000}; // the number of elements for vectors
  int num_thread{128};       // the number of threads

  const char* optstr = "hg:s:n:t:";

  int opt, index;
  while ((opt = getopt_long(argc, argv, optstr, opts, &index)) != -1) {
    switch (opt) {
      case 'h':
        ::print_usage();
        std::exit(EXIT_SUCCESS);
      case 's':
        seed = ppcc::lexical_cast<size_t>(optarg);
        break;
      case 'n':
        num_element = ppcc::lexical_cast<int>(optarg);
        break;
      case 't':
        num_thread = ppcc::lexical_cast<int>(optarg);
        break;
      default:
        std::exit(EXIT_FAILURE);
    }
  }

  // set GPU device
  cudaError_t err = cudaSetDevice(gpuid);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] Failed to set GPU device (ID: " << gpuid << ")"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // check thread number
  if (num_thread > ppcc::kUppLimNumThread) {
    std::cerr << "[ERROR] The thread number should be lower than "
              << ppcc::kUppLimNumThread << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  ppcc::print_device_info();

  // setup random generator
  std::mt19937 mtgen(seed);
  std::uniform_real_distribution<float> udist(0.0f, 1.0f);

  // setup vectors
  float* x = new float[num_element];
  float* y = new float[num_element];
  for (int i = 0; i < num_element; i++) {
    x[i] = udist(mtgen);
    y[i] = udist(mtgen);
  }

  // data size (unit: Byte)
  size_t size = sizeof(float) * num_element;

  // memory allocation on GPU
  float *dx, *dy;
  cudaMalloc(&dx, size);
  cudaMalloc(&dy, size);

  // data copy from CPU to GPU
  cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, size, cudaMemcpyHostToDevice);

  // setup timer
  auto& timer = ppcc::KernelTimer::GetInstance();
  timer.Set();

  // setup thread configuration
  int num_block  = num_element / num_thread;
  if (num_element % num_thread > 0) { num_block += 1; }

  timer.Start(); // start timer

  // call kernel function
  ::saxpy<<<num_block, num_thread>>>(num_element, 3.0f, dx, dy);
  cudaDeviceSynchronize();

  timer.Stop();  // stop timer

  // data copy from GPU to CPU
  cudaMemcpy(y, dy, size, cudaMemcpyDeviceToHost);

  float elaptime = timer.GetElapsedTime();

  std::stringstream msg;
  msg << "[MESSAGE] Summary of CUDA SAXPY" << std::endl;
  msg << "- Vector Size: " << num_element << std::endl;
  msg << "- Thread Configuration: (" << num_block << " x " << num_thread << ")"
      << std::endl;
  msg << "- Elapsed Time: " << elaptime << " (msec)" << std::endl;
  msg << "- Sum of Z elements: " << std::accumulate(y, y + num_element, 0.0)
      << std::endl;

  std::cout << msg.str();

  if (x)  { delete x; }
  if (y)  { delete y; }
  if (dx) { cudaFree(dx); }
  if (dy) { cudaFree(dy); }

  std::exit(EXIT_SUCCESS);
}
