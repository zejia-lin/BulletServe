/* Copyright 2021-2023 Joshua Bakita
 * Header for miscellaneous experimental helper functions.
 */

// cudaError_t and CUResult can both safely be cast to an unsigned int
static __thread unsigned int __SAFE_err;

// The very strange cast in these macros is to satisfy two goals at tension:
// 1. This file should be able to be included in non-CUDA-using files, and thus
//    should use no CUDA types outside of this macro.
// 2. We want to typecheck uses of these macros. The driver and runtime APIs
//    do not have identical error numbers and/or meanings, so runtime library
//    calls should use SAFE, and driver library calls should use SAFE_D.
// These macros allow typechecking, but keep a generic global error variable.
#define SAFE(x) \
	if ((*(cudaError_t*)(&__SAFE_err) = (x)) != 0) { \
		printf("(%s:%d) CUDA error %d: %s i.e. \"%s\" returned by %s. Aborting...\n", \
		       __FILE__, __LINE__, __SAFE_err, cudaGetErrorName((cudaError_t)__SAFE_err), cudaGetErrorString((cudaError_t)__SAFE_err), #x); \
		exit(1); \
	}

#define SAFE_D(x) \
	if ((*(CUresult*)&(__SAFE_err) = (x)) != 0) { \
		const char* name; \
		const char* desc; \
		cuGetErrorName((CUresult)__SAFE_err, &name); \
		cuGetErrorString((CUresult)__SAFE_err, &desc); \
		printf("(%s:%d) CUDA error %d: %s i.e. \"%s\" returned by %s. Aborting...\n", \
		       __FILE__, __LINE__, __SAFE_err, name, desc, #x); \
		exit(1); \
	}
