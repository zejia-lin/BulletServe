base_dir=$(shell pwd)
build_dir=${base_dir}/build
nvcc=${shell which nvcc}
C=${shell which gcc}
CXX=${shell which g++}

# Automatically detect CUDA architecture from available GPU
CUDA_ARCH=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | sed 's/\.//')

.PHONY: build config clean rm rebuild

build:
	cmake --build ${build_dir} --target all -j $(shell grep -c ^processor /proc/cpuinfo)

config: CMakeLists.txt
	cmake -B ${build_dir} -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${C} -DCMAKE_CXX_COMPILER=${CXX}

clean:
	cmake --build ${build_dir} --target clean

rebuild: clean build