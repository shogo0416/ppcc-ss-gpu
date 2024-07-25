/*==============================================================================
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
==============================================================================*/
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include "cuda_helper.h"
#include "kernel_timer.h"
#include "lexical_cast.h"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

namespace {

//------------------------------------------------------------------------------
void print_usage()
{
  const char* usage = R"(
    [Usage] cuda-saxpy <options>
      -h, --help              Print this information

      -g, --gpuid <val>       Set GPU device id
                              [default: 0]

      -s, --seed <val>        Set seed for a random number generator
                              [default: 123456789]

      -n, --num_point <val>   Set the number of points
                              [default: 10000000]

      -b, --num_block <val>   Set the number of blocks
                              [default: 4000]

      -t, --num_thread <val>  Set the number of threads
                              [default: 128]
  )";

  std::cout << usage << std::endl;
}

//------------------------------------------------------------------------------
// initialize random number engine for each CUDA thread
__global__
void init_generator(size_t seed, curandState_t* rng_state)
{
  auto id = ppcc::get_thread_id();
  curand_init(seed, id, 0, &rng_state[id]);
}

//------------------------------------------------------------------------------
__global__
void gen_rand_num(size_t num_point, curandState_t* rng_state, size_t* counter)
{
  auto tot_thread = ppcc::get_tot_thread();
  auto num_iteration = num_point / size_t(tot_thread);

  auto id = ppcc::get_thread_id();
  if (id < num_point % tot_thread) { num_iteration += 1; }

  auto loc_state = rng_state[id];
  size_t loc_counter{0};
  for (size_t i = 0; i < num_iteration; i++) {
    auto x = curand_uniform(&loc_state);
    auto y = curand_uniform(&loc_state);
    if (x * x + y * y < 1.0f) { loc_counter++; }
  }

  counter[id] = loc_counter;
  rng_state[id] = loc_state;
}

} // end of anonymous namespace

//==============================================================================
// main function
int main(int argc, char** argv)
{

  struct option opts[] = {
    {"help",       no_argument,       nullptr, 'h'},
    {"gpuid",      required_argument, nullptr, 'g'},
    {"seed",       required_argument, nullptr, 's'},
    {"num_point",  required_argument, nullptr, 'n'},
    {"num_block",  required_argument, nullptr, 'b'},
    {"num_thread", required_argument, nullptr, 't'},
    {nullptr,      0,                 nullptr, 0}
  };

  int gpuid{0};               // GPU device identifier
  size_t seed{123456789};     // seed for random number generator
  size_t num_point{10000000}; // the number of points
  int num_block{4000};        // the number of blocks
  int num_thread{128};        // the number of threads

  const char* optstr = "hg:s:n:b:t:";

  int opt, index;
  while ((opt = getopt_long(argc, argv, optstr, opts, &index)) != -1) {
    switch (opt) {
      case 'h':
        ::print_usage();
        std::exit(EXIT_SUCCESS);
      case 'g':
        gpuid = ppcc::lexical_cast<int>(optarg);
        break;
      case 's':
        seed = ppcc::lexical_cast<size_t>(optarg);
        break;
      case 'n':
        num_point = ppcc::lexical_cast<size_t>(optarg);
        break;
      case 'b':
        num_block = ppcc::lexical_cast<int>(optarg);
        break;
      case 't':
        num_thread = ppcc::lexical_cast<int>(optarg);
        break;
      default:
        std::exit(EXIT_FAILURE);
    }
  }

  // set GPU device
  cudaError_t cuda_err = cudaSetDevice(gpuid);
  if (cuda_err != cudaSuccess) {
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

  const auto tot_thread = num_block * num_thread;

  thrust::device_vector<curandState_t> dv_rng_state(tot_thread);
  auto rng_state = thrust::raw_pointer_cast(dv_rng_state.data());

  thrust::device_vector<size_t> dv_counter(tot_thread, 0);
  auto counter = thrust::raw_pointer_cast(dv_counter.data());

  auto& timer = ppcc::KernelTimer::GetInstance();
  timer.Set();   // setup timer
  timer.Start(); // start timer

  // initialize cuRAND generator
  ::init_generator<<<num_block, num_thread>>>(seed, rng_state);

  // generate random number pairs
  ::gen_rand_num<<<num_block, num_thread>>>(num_point, rng_state, counter);

  cudaDeviceSynchronize();

  auto sum = thrust::reduce(dv_counter.begin(), dv_counter.end());

  timer.Stop(); // stop timer

  // reference value from G4PhysicalConstants.hh
  constexpr double ref_pi= 3.14159265358979323846264338328;

  auto pi  = 4.0 * static_cast<double>(sum) / static_cast<double>(num_point);
  auto err = (pi - ref_pi) / ref_pi * 1000000.0; // in ppm

  auto elap_time = timer.GetElapsedTime(); // in msec
  auto num_pairs_per_sec = num_point / elap_time * 1000.0;

  std::stringstream msg;
  msg << "[MESSAGE] Summary" << std::endl;
  msg << "- Point Number : " << num_point << std::endl;
  msg << "- CUDA Thread Configuration : (" << num_block
      << " blocks x " << num_thread << " threads/block)" << std::endl;
  msg << "- Elapsed Time : " << elap_time << " (msec)" << std::endl;
  msg << "- Random Number Pairs per sec : " << num_pairs_per_sec << std::endl;
  msg << "- pi  : " << std::setprecision(15) << pi << std::endl;
  msg << "- err : " << std::defaultfloat << err << " (ppm)" << std::endl;

  std::cout << msg.str();

  std::exit(EXIT_SUCCESS);
}
