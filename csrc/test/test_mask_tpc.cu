#include <cstdio>

#include <iostream>

#include "libsmctrl.h"
#include "libsmctrl_test_mask_shared.h"
#include "check_cuda.h"

__global__ void echo_sm(int *used_sm) {
  if (threadIdx.x != 1)
    return;
  int smIdx;
  asm("mov.u32 %0, %%smid;"
      : "=r"(smIdx));
  printf("%d, ", smIdx);
  used_sm[smIdx] = 1;
}

int main(int argc, char **argv) {
  int *used_sm;
  int num_sms;
  uint128_t mask;
  cudaStream_t stream;
  checkCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  checkCuda(cudaMallocManaged(&used_sm, sizeof(int) * num_sms));
  checkCuda(cudaStreamCreate(&stream));
  checkCuda(cudaMemset(used_sm, 0, sizeof(int) * num_sms));

  int low = atoi(argv[1]);
  int high = atoi(argv[2]);
  libsmctrl_make_mask_ext(&mask, low, high);
  libsmctrl_set_stream_mask_ext(stream, mask);
  echo_sm<<<256, 12, 0, stream>>>(used_sm);
  checkCuda(cudaDeviceSynchronize());
  for(int i = 0; i < num_sms; ++i){
    if(used_sm[i] == 1){
      if((i / 2 < low) or (i / 2 >= high)){
        std::cout << "SM " << i << " shouldn't be used\n";
      }
    } else if((low <= i / 2) && (i / 2 < high)){
      std::cout << "SM " << i << " should be used\n";
    }
  }
}