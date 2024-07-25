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
#ifndef CUDA_HELPER_H_
#define CUDA_HELPER_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ppcc {

constexpr int kUppLimNumThread = 1024;

void print_device_info();

__device__ int get_thread_id();

__device__ int get_tot_thread();

//==============================================================================
inline __device__ int get_thread_id()
{
  // return thread id
  return threadIdx.x + blockIdx.x * blockDim.x;
}

//------------------------------------------------------------------------------
inline __device__ int get_tot_thread()
{
  // return total number of threads for this kernel call
  return gridDim.x * blockDim.x;
}

} // end of namespace ppcc

#endif // CUDA_HELPER_H_
