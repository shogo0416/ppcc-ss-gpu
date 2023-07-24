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
#include <chrono>
#include <numeric>
#include "lexical_cast.h"

using namespace std::chrono;

namespace {

//------------------------------------------------------------------------------
void print_usage()
{
  const char* usage = R"(
    [Usage] cpp-saxpy <options>
      -h, --help                Print this information

      -s, --seed <val>          Set seed for a random number generator
                                [default: 123456789]

      -n, --num_element <val>   Set the number of elements for vectors
                                [default: 10000000]
  )";

  std::cout << usage << std::endl;
}

//------------------------------------------------------------------------------
void saxpy(int n, float a, float* x, float* y)
{
  for (int i = 0; i < n; i++) { y[i] += a * x[i]; }
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

  size_t seed{123456789};    // seed for random number generator
  int num_element{10000000}; // the number of elements for vectors

  const char* optstr = "hs:n:";

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
      default:
        std::exit(EXIT_FAILURE);
    }
  }

  // setup for random generator
  std::mt19937 mtgen(seed);
  std::uniform_real_distribution<float> udist(0.0f, 1.0f);

  // setup for vectors
  float* x = new float[num_element];
  float* y = new float[num_element];
  for (int i = 0; i < num_element; i++) {
    x[i] = udist(mtgen);
    y[i] = udist(mtgen);
  }

  // setup for timer
  system_clock::time_point start, stop;

  start = high_resolution_clock::now(); // start timer

  saxpy(num_element, 3.0f, x, y);

  stop = high_resolution_clock::now(); // stop timer

  auto elap_time = static_cast<double>(
    duration_cast<nanoseconds>(stop - start).count()) / 1.0E+06f; // in msec

  std::stringstream msg;
  msg << "[MESSAGE] Summary of C++ SAXPY" << std::endl;
  msg << "- Vector Size:  " << num_element << std::endl;
  msg << "- Elapsed Time: " << elap_time << " (msec)" << std::endl;
  msg << "- Sum of Z elements: "
      << std::accumulate(y, y + num_element, 0.0) << std::endl;

  std::cout << msg.str();

  if (x) { delete x; }
  if (y) { delete y; }

  std::exit(EXIT_SUCCESS);
}
