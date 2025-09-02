// Copyright 2023 Joshua Bakita
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"
#include "testbench.h"
#include "libsmctrl_test_mask_shared.h"

__global__ void read_and_store_smid(uint8_t* smid_arr) {
  if (threadIdx.x != 1)
    return;
  int smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  smid_arr[blockIdx.x] = smid;
}

// Assuming SMs continue to support a maximum of 2048 resident threads, six
// blocks of 1024 threads should span at least three SMs without partitioning
#define NUM_BLOCKS 142 //6

static int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

// Warning: Mutates input array via qsort
static int count_unique(uint8_t* arr, int len) {
  qsort(arr, len, 1, sort_asc);
  int num_uniq = 1;
  for (int i = 0; i < len - 1; i++)
    num_uniq += (arr[i] != arr[i + 1]);
  return num_uniq;
}

// Test that adding an SM mask:
// 1. Constrains the number of SMs accessible
// 2. Constrains an application to the correct subset of SMs
int test_constrained_size_and_location(enum partitioning_type part_type) {
  int res;
  uint8_t *smids_native_d, *smids_native_h;
  uint8_t *smids_partitioned_d, *smids_partitioned_h;
  int uniq_native, uniq_partitioned;
  uint32_t num_tpcs;
  int num_sms, sms_per_tpc;
  cudaStream_t stream;

  SAFE(cudaStreamCreate(&stream));

  // Determine number of SMs per TPC
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "Unable to get TPC configuration for test");
  sms_per_tpc = num_sms/num_tpcs;

  // Test baseline (native) behavior without partitioning
  SAFE(cudaMalloc(&smids_native_d, NUM_BLOCKS));
  if (!(smids_native_h = (uint8_t*)malloc(NUM_BLOCKS)))
    error(1, errno, "Unable to allocate memory for test");
  read_and_store_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(smids_native_d);
  SAFE(cudaMemcpy(smids_native_h, smids_native_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

  uniq_native = count_unique(smids_native_h, NUM_BLOCKS);
  if (uniq_native < sms_per_tpc) {
    printf("%s: ***Test failure.***\n"
           "%s: Reason: In baseline test, %d blocks of 1024 "
           "threads were launched on the GPU, but only %d SMs were utilized, "
           "when it was expected that at least %d would be used.\n", program_invocation_name, program_invocation_name, NUM_BLOCKS,
           uniq_native, sms_per_tpc);
    return 1;
  }

  // Test at 32-TPC boundaries to verify that the mask is applied in the
  // correct order to each of the QMD/stream struct fields.
  char* reason[4] = {0};
  for (int enabled_tpc = 0; enabled_tpc < num_tpcs && enabled_tpc < 128; enabled_tpc += 32) {
    uint128_t mask = 1;
    mask <<= enabled_tpc;
    mask = ~mask;

    // Apply partitioning to enable only the first TPC of each 32-bit block
    switch (part_type) {
      case PARTITION_GLOBAL:
        libsmctrl_set_global_mask(mask);
        break;
      case PARTITION_STREAM:
        libsmctrl_set_stream_mask_ext(stream, mask);
        break;
      case PARTITION_STREAM_OVERRIDE:
        libsmctrl_set_global_mask(~mask);
        libsmctrl_set_stream_mask_ext(stream, mask);
        break;
      case PARTITION_NEXT:
        libsmctrl_set_next_mask(mask);
        break;
      case PARTITION_NEXT_OVERRIDE:
        libsmctrl_set_global_mask(~mask);
        libsmctrl_set_stream_mask_ext(stream, ~mask);
        libsmctrl_set_next_mask(mask);
        break;
      default:
        error(1, 0, "Shared test core called with unrecognized partitioning type.");
    }

    // Verify that partitioning changes the SMID distribution
    SAFE(cudaMalloc(&smids_partitioned_d, NUM_BLOCKS));
    if (!(smids_partitioned_h = (uint8_t*)malloc(NUM_BLOCKS)))
      error(1, errno, "Unable to allocate memory for test");
    read_and_store_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(smids_partitioned_d);
    SAFE(cudaMemcpy(smids_partitioned_h, smids_partitioned_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

    // Make sure it only ran on the number of TPCs provided
    // May run on up to two SMs, as up to two per TPC
    uniq_partitioned = count_unique(smids_partitioned_h, NUM_BLOCKS); // Sorts too
    if (uniq_partitioned > sms_per_tpc) {
      printf("%s: ***Test failure.***\n"
             "%s: Reason: With TPC mask set to "
             "constrain all kernels to a single TPC, a kernel of %d blocks of "
             "1024 threads was launched and found to run on %d SMs (at most %d---"
             "one TPC---expected).\n", program_invocation_name, program_invocation_name, NUM_BLOCKS, uniq_partitioned, sms_per_tpc);
      return 1;
    }

    // Make sure it ran on the right TPC
    if (smids_partitioned_h[NUM_BLOCKS - 1] > (enabled_tpc * sms_per_tpc) + sms_per_tpc - 1 ||
        smids_partitioned_h[NUM_BLOCKS - 1] < (enabled_tpc * sms_per_tpc)) {
      printf("%s: ***Test failure.***\n"
             "%s: Reason: With TPC mask set to "
             "constrain all kernels to TPC %d, a kernel was run and found "
             "to run on an SM IDs: as high as %d and as low as %d (range of %d to %d expected).\n",
             program_invocation_name, program_invocation_name, enabled_tpc, smids_partitioned_h[NUM_BLOCKS - 1], smids_partitioned_h[0], enabled_tpc * sms_per_tpc + sms_per_tpc - 1, enabled_tpc * sms_per_tpc);
      return 1;
    }

    // Div by 32 via a shift
    asprintf(&reason[enabled_tpc >> 5],
         "With a partition enabled which "
         "contained only TPC ID %d, the test kernel was found to use only %d "
         "SMs (%d without), and all SMs in-use had IDs between %d and %d (were contained"
         " in TPC %d).", enabled_tpc, uniq_partitioned, uniq_native, smids_partitioned_h[0], smids_partitioned_h[NUM_BLOCKS - 1], enabled_tpc);
  }

  printf("%s: Test passed!\n", program_invocation_name);
  for (int i = 0; i < 4 && reason[i]; i++)
    printf("%s: Reason %d: %s\n", program_invocation_name, i + 1, reason[i]);
  return 0;
}

