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
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <thread>
#include <random>
#include <vector>

#include "lexical_cast.h"

using namespace std::chrono;

namespace {

//------------------------------------------------------------------------------
void print_usage()
{
  const char* usage = R"(
    [Usage] cpp-pi <options>
      -h, --help              Print this information

      -s, --seed <val>        Set seed for a random number generator
                              [default: 123456789]

      -n, --num_point <val>   Set the number of points
                              [default: 10000000]

      -t, --num_thread <val>  Set the number of worker threads
                              [default: 2]
  )";

  std::cout << usage << std::endl;
}

//------------------------------------------------------------------------------
void gen_rand_num(int thread_id, size_t seed, size_t num_iteration,
                  std::vector<size_t>& counter)
{
  // setup for random number generator
  std::mt19937 mtgen(seed);
  std::uniform_real_distribution<float> udist(0.0f, 1.0f);

  size_t loc_counter{0};
  for (size_t i = 0; i < num_iteration; i++) {
    auto x = udist(mtgen);
    auto y = udist(mtgen);
    if (x * x + y * y < 1.0f) { loc_counter++; }
  }

  counter[thread_id] = loc_counter;
}

} // end of anonymous namespace

//==============================================================================
// main function
int main(int argc, char** argv)
{
  struct option opts[] = {
    {"help",        no_argument,       nullptr, 'h'},
    {"seed",        required_argument, nullptr, 's'},
    {"num_element", required_argument, nullptr, 'n'},
    {nullptr,       0,                 nullptr, 0}
  };

  size_t seed{123456789};     // seed for random number generator
  size_t num_point{10000000}; // the number of points
  int num_thread{2};          // the number of worker threads

  const char* optstr = "hs:n:t:";

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
        num_point = ppcc::lexical_cast<size_t>(optarg);
        break;
      case 't':
        num_thread = ppcc::lexical_cast<int>(optarg);
        break;
      default:
        std::exit(EXIT_FAILURE);
    }
  }

  std::vector<size_t> counter(num_thread, 0);

  // setup for worker threads
  std::vector<std::thread> worker_threads(num_thread);

  // setup for timer
  system_clock::time_point start, stop;

  start = high_resolution_clock::now(); // start timer

  for (int id = 0; id < num_thread; id++) {
    auto num_iteration = num_point / static_cast<size_t>(num_thread);
    if (id < num_point % static_cast<size_t>(num_thread)) {
      num_iteration += 1;
    }
    worker_threads[id] = std::thread(::gen_rand_num, id, seed + id,
                                     num_iteration, std::ref(counter));
  }

  // synchronize
  for (auto& x : worker_threads) { x.join(); }

  auto sum = std::reduce(counter.begin(), counter.end());

  stop = high_resolution_clock::now(); // stop timer

  // reference value from G4PhysicalConstants.hh
  constexpr double ref_pi= 3.14159265358979323846264338328;

  auto pi  = 4.0 * static_cast<double>(sum) / static_cast<double>(num_point);
  auto err = (pi - ref_pi) / ref_pi * 1000000.0; // in ppm

  auto elap_time = static_cast<double>(
    duration_cast<nanoseconds>(stop - start).count()) / 1.0E+06f; // in msec

  auto num_pairs_per_sec = num_point / elap_time * 1000.0;

  std::stringstream msg;
  msg << "[MESSAGE] Summary" << std::endl;
  msg << "- Point Number : " << num_point << std::endl;
  msg << "- Worker Threads : " << num_thread << std::endl;
  msg << "- Elapsed Time : " << elap_time << " (msec)" << std::endl;
  msg << "- Random Number Pairs per sec : " << num_pairs_per_sec << std::endl;
  msg << "- pi  : " << std::setprecision(15) << pi << std::endl;
  msg << "- err : " << std::defaultfloat << err << " (ppm)" << std::endl;

  std::cout << msg.str();

  std::exit(EXIT_FAILURE);
}
