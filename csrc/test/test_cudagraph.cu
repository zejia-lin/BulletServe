#include <cstdio>

#include <iostream>

#include "check_cuda.h"
#include "libsmctrl.h"
#include "libsmctrl_test_mask_shared.h"

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
  uint64_t mask;

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStream_t stream, s2;
  checkCuda(cudaStreamCreate(&stream));
  checkCuda(cudaStreamCreate(&s2));

  checkCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  checkCuda(cudaMallocManaged(&used_sm, sizeof(int) * num_sms));

  int low = atoi(argv[1]);
  int high = atoi(argv[2]);
  libsmctrl_make_mask(&mask, low, high);
  libsmctrl_set_stream_mask(s2, mask);
  // libsmctrl_set_global_mask(mask);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < 5; ++i) {
    echo_sm<<<256, 12, 0, stream>>>(used_sm);
  }
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphLaunch(instance, s2);
  checkCuda(cudaDeviceSynchronize());
  
  for (int i = 0; i < num_sms; ++i) {
    if (used_sm[i] == 1) {
      if ((i / 2 < low) or (i / 2 >= high)) {
        std::cout << "SM " << i << " shouldn't be used\n";
      }
    } else if ((low <= i / 2) && (i / 2 < high)) {
      std::cout << "SM " << i << " should be used\n";
    }
  }
}