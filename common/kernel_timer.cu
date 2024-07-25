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
#include "kernel_timer.h"

namespace ppcc {

//------------------------------------------------------------------------------
KernelTimer& KernelTimer::GetInstance()
{
  static KernelTimer timer;
  return timer;
}

//------------------------------------------------------------------------------
void KernelTimer::Set()
{
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
}

//------------------------------------------------------------------------------
void KernelTimer::Reset()
{
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
  Set();
}

//------------------------------------------------------------------------------
void KernelTimer::Start()
{
  cudaEventRecord(start_);
}

//------------------------------------------------------------------------------
void KernelTimer::Stop()
{
  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);
}

//------------------------------------------------------------------------------
float KernelTimer::GetElapsedTime() const
{
  auto elaptime{0.0f};
  cudaEventElapsedTime(&elaptime, start_, stop_);
  return elaptime; // in msec
}

} // end of namespace ppcc
