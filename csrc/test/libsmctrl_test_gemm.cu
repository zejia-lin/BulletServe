#include <errno.h>
#include <error.h>
#include <stdbool.h>
#include <stdio.h>

#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "check_cuda.h"
#include "libsmctrl.h"

#define SAFE(x) x

__global__ void helloworld() {
  printf("hello world");
  printf("(%d,%d) ", threadIdx.x, blockIdx.x);
  // if (threadIdx.x != 1)
  //   return;
  // int smIdx;
  // asm("mov.u32 %0, %%smid;"
  //     : "=r"(smIdx));
  // printf("%d, ", smIdx);
}

int main() {
  const int NUM_BLOCKS = 400;
  int res;
  uint32_t num_tpcs;
  int num_sms, sms_per_tpc;

  // Determine number of SMs per TPC
  checkCuda(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "libsmctrl_test_global: Unable to get TPC configuration for test");
  sms_per_tpc = num_sms / num_tpcs;

  printf("Find %d SM, %d SM/TPC\n", num_sms, sms_per_tpc);

  // Test baseline (native) behavior without partitioning
  // printf("Before partition\n");
  // helloworld<<<1, 1>>>();
  // checkCuda(cudaDeviceSynchronize());
  // helloworld<<<NUM_BLOCKS, 128>>>();
  // checkCuda(cudaDeviceSynchronize());

  // printf("\nAfter partition\n");
  // libsmctrl_set_global_mask(~0x3ull);
  // helloworld<<<NUM_BLOCKS, 128>>>();
  // checkCuda(cudaDeviceSynchronize());
  // exit(0);

  const size_t MAX_PROMPTS = 5120;
  const size_t MAX_DMODEL = 5120;
  const size_t MLP_HIDDEN = 22016;
  const int NUM_RUNS = 5;
  size_t *X, *W, *Y;
  const float ONE = 1;
  cudaEvent_t st, ed;
  float duration;
  cudaMalloc(&X, sizeof(__half) * MLP_HIDDEN * MLP_HIDDEN);
  cudaMalloc(&W, sizeof(__half) * MLP_HIDDEN * MLP_HIDDEN);
  cudaMalloc(&Y, sizeof(float) * MLP_HIDDEN * MLP_HIDDEN);
  checkCuda(cudaEventCreate(&st));
  checkCuda(cudaEventCreate(&ed));

  std::cout << "num_prompts,tpcs,time\n";
  for (size_t num_promts = 4096 - 8; num_promts <= 4096 + 8; num_promts += 1) {
    size_t d_model = 4096;
    size_t M = num_promts;
    size_t K = d_model;
    size_t N = 22016;

    cublasHandle_t blasHandle;
    cublasOperation_t trans = CUBLAS_OP_N;
    checkCuda(cublasCreate(&blasHandle));
    for (int i = 0; i < NUM_RUNS; ++i) {
      checkCuda(cublasGemmEx(blasHandle, trans, trans,
                             M, N, K,
                             &ONE, X, CUDA_R_16F, M,
                             W, CUDA_R_16F, K,
                             &ONE, Y, CUDA_R_32F, M,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    checkCuda(cudaDeviceSynchronize());

    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    checkCuda(cublasSetStream(blasHandle, stream));

    unsigned long long tpcs = 0ULL;
    for (int _ = 0; _ < 54; ++_) {
      tpcs = (tpcs << 1) | 1ULL;
      if(_ < 53){
        // continue;
      }
      libsmctrl_set_stream_mask(stream, ~tpcs);
      // libsmctrl_set_global_mask(~tpcs);
      // checkCuda(cublasSetSmCountTarget(blasHandle, 32));
      checkCuda(cudaEventRecord(st));
      for (int i = 0; i < NUM_RUNS; ++i) {
        checkCuda(cublasGemmEx(blasHandle, trans, trans,
                               M, N, K,
                               &ONE, X, CUDA_R_16F, M,
                               W, CUDA_R_16F, K,
                               &ONE, Y, CUDA_R_32F, M,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
      checkCuda(cudaDeviceSynchronize());
      checkCuda(cudaEventRecord(ed));
      checkCuda(cudaDeviceSynchronize());
      checkCuda(cudaEventElapsedTime(&duration, st, ed));
      std::cout << num_promts << "," << __builtin_popcountll(tpcs) << "," << duration / NUM_RUNS << "\n";
    }
  }
}
