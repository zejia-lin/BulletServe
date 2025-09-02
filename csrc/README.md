# libsmctrl: Quick & Easy Hardware Compute Partitioning on NVIDIA GPUs

Library to enable intra-context SM/TPC partitioning on NVIDIA GPUs. Forked from http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/

## Building

Run the following command and `libsmctrl.so` will be created in the `build` directory. This current fork only runs with CUDA <= 12.2 with driver version <= 535.xxx.xx.
The [original repository](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/) have updated to support CUDA 12.8.


```bash
make config
make build
```

## Usage

See Table 1 in [libsmctrl's paper](https://www.cs.unc.edu/~jbakita/rtas23.pdf).

## Testing

After building, just run the executables in the `build/test` directory, e.g.

```bash
./libsmctrl_test_global_mask
./libsmctrl_test_stream_mask
./libsmctrl_test_next_mask
```


# Origin Readme
This library was developed as part of the following paper:

_J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs", Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium, pp. 54-66, May 2023._

Please cite this paper in any work which leverages our library. Here's the BibTeX entry:
```
@inproceedings{bakita2023hardware,
  title={Hardware Compute Partitioning on {NVIDIA} {GPUs}},
  author={Bakita, Joshua and Anderson, James H},
  booktitle={Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium},
  year={2023},
  month={May},
  pages={54--66},
  _series={RTAS}
}
```

Please see [the paper](https://www.cs.unc.edu/~jbakita/rtas23.pdf) and libsmctrl.h for details and examples of how to use this library.
We strongly encourage consulting those resources first; the below comments serve merely as an appendum.

## Run-time Dependencies
`libcuda.so`, which is automatically installed by the NVIDIA GPU driver.

## Building
To build, ensure that you have `gcc` installed and access to the CUDA SDK including `nvcc`. Then run:
```
make libsmctrl.a
```

If you see an error that the command `nvcc` was not found, `nvcc` is not available on your `PATH`.
Correct this error by explictly specifying the location of `nvcc` to `make`, e.g.:
```
make NVCC=/playpen/jbakita/CUDA/cuda-archive/cuda-10.2/bin/nvcc libsmctrl.a
```

For binary backwards-compatibility to old versions of the NVIDIA GPU driver, we recommend building with an old version of the CUDA SDK.
For example, by building against CUDA 10.2, the binary will be compatible with any version of the NVIDIA GPU driver newer than 440.36 (Nov 2019), but by building against CUDA 8.0, the binary will be compatible with any version of the NVIDIA GPU driver newer that 375.26 (Dec 2016).

Older versions of `nvcc` may require you to use an older version of `g++`.
This can be explictly specified via the `CXX` variable, e.g.:
```
make NVCC=/playpen/jbakita/CUDA/cuda-archive/cuda-8.0/bin/nvcc CXX=g++-5 libsmctrl.a
```

`libsmctrl` supports being built as a shared library.
This will require you to distribute `libsmctrl.so` with your compiled program.
If you do not know what a shared library is, or why you would need to specify the path to `libsmctrl.so` in `LD_LIBRARY_PATH`, do not do this.
To build as a shared library, replace `libsmctrl.a` with `libsmctrl.so` in the above commands.

## Linking in Your Application
If you have cloned and built `libsmctrl` in the folder `/playpen/libsmctrl` (replace this with the location you use):

1. Add `-I/playpen/libsmctrl` to your compiler command (this allows `#include <libsmctrl.h>` in your C/C++ files).
2. Add `-lsmctrl` to your linker command (this allows the linker to resolve the `libsmctrl` functions you use to the implementations in `libsmctrl.a` or `libsmctrl.so`).
3. Add `-L/playpen/libsmctrl` to your linker command (this allows the linker to find `libsmctrl.a` or `libsmctrl.so`).
4. (If not already included) add `-lcuda` to your linker command (this links against the CUDA driver library).

Note that if you have compiled both `libsmctrl.a` (the static library) and `libsmctrl.so` (the shared library), most compilers will prefer the shared library.
To statically link against `libsmctrl.a`, delete `libsmctrl.so`.

For example, if you have a CUDA program written in `benchmark.cu` and have built `libsmctrl`, you can compile and link against `libsmctrl` via the following command:
```
nvcc benchmark.cu -o benchmark -I/playpen/libsmctl -lsmctrl -lcuda -L/playpen/libsmctrl
```
The resultant `benchmark` binary should be portable to any system with an equivalent or newer version of the NVIDIA GPU driver installed.

## Run Tests
To test partitioning:
```
make tests
./libsmctrl_test_global_mask
./libsmctrl_test_stream_mask
./libsmctrl_test_next_mask
```

To test that high-granularity masks override low-granularity ones:
```
make tests
./libsmctrl_test_stream_mask_override
./libsmctrl_test_next_mask_override
```

And if `nvdebug` has been installed:
```
make tests
./libsmctrl_test_gpc_info
```

## Supported GPUs

#### Known Working

- NVIDIA GPUs from compute capability 3.5 through 8.9, including embedded "Jetson" GPUs
- CUDA 8.1 through 12.2
- `x86_64` and Jetson `aarch64` platforms

#### Known Issues

- `next_mask` will not override `stream_mask` on CUDA 11.0+
    - _As of Feb 2024, a fix for this is coming soon..._
- `global_mask` and `next_mask` cannot disable TPCs with IDs above 128
    - Only relevant on GPUs with over 128 TPCs, such as the RTX 6000 Ada
- Untested on H100 (compute capability 9.0)
- Untested on non-Jetson `aarch64` platforms

## Important Limitations

1. Only supports partitioning _within_ a single GPU context.
   At time of writing, it is challenging to impossible to share a GPU context across multiple CPU address spaces.
   The implication is that your applications must first be combined together into a single CPU process.
2. No aspect of this system prevents implicit synchronization on the GPU.
   See prior work, particularly that of Amert et al. (perhaps the CUPiD^RT paper), for ways to avoid this.

## Porting to New Architectures

Build the tests with `make tests`. And then run the following:
```
for (( i=0; $?!=0; i+=8 )); do MASK_OFF=$i ./libsmctrl_test_stream_mask; done
```

How this works:

1. If `MASK_OFF` is set, `libsmctrl` applies this as a byte offset to a base address for the location
   of the SM mask fields in CUDA's stream data structure.
  - That base address is the one for CUDA 12.2 at time of writing
2. The stream masking test is run.
3. If the test succeeded (returned zero) the loop aborts, otherwise it increments the offset to attempt and repeats.

Once this loop aborts, take the found offset and add it into the switch statement for the appropriate CUDA version and CPU architecture.
