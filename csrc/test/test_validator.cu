#include <iostream>

#include "libsmctrl.h"
#include "check_cuda.h"

using namespace std;

int main(int argc, char **argv){
  int low = atoi(argv[1]);
  int hi = atoi(argv[2]);
  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));
  uint128_t mask;
  libsmctrl_make_mask_ext(&mask, low, hi);
  libsmctrl_set_stream_mask_ext(stream, mask);
  int ret_code = libsmctrl_validate_stream_mask(stream, low, hi, true);
  if(ret_code == 0){
    cout << "test passed\n";
  } else {
    cout << "test failed\n";
  }
}