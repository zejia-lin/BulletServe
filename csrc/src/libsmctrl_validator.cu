#include <cstdio>

#include <iostream>

#include <cuda_runtime.h>

#include "check_cuda.h"
#include "libsmctrl.h"

__global__ void echo_sm(int *used_sm, bool echo) {
  if (threadIdx.x != 1)
    return;
  int smIdx;
  asm("mov.u32 %0, %%smid;"
      : "=r"(smIdx));
  if (echo) {
    printf("%d, ", smIdx);
  }
  used_sm[smIdx] = 1;
}

int libsmctrl_validate_stream_mask(void *stream_ptr, int low, int high, bool echo) {
  if (echo) {
    std::cout << "validating stream '" << stream_ptr << "' with mask ranged (" << low << ", " << high << ")\n";
  }
  int *used_sm;
  int num_sms;
  int ret_code = 0;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  checkCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  checkCuda(cudaMallocManaged(&used_sm, sizeof(int) * num_sms));
  checkCuda(cudaMemset(used_sm, 0, sizeof(int) * num_sms));
  echo_sm<<<256, 12, 0, stream>>>(used_sm, echo);
  checkCuda(cudaStreamSynchronize(stream));
  for (int i = 0; i < num_sms; ++i) {
    if (used_sm[i] == 1) {
      if ((i / 2 < low) or (i / 2 >= high)) {
        std::cout << "SM " << i << " shouldn't be used\n";
        ret_code = -1;
      }
    } else if ((low <= i / 2) && (i / 2 < high)) {
      std::cout << "SM " << i << " should be used\n";
      ret_code = -1;
    }
  }
  checkCuda(cudaFree(used_sm));
  return ret_code;
}