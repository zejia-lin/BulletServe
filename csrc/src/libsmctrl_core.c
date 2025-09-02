/**
 * Copyright 2022-2025 Joshua Bakita
 * Library to control SM masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 *
 * This file implements partitioning via three different mechanisms:
 * - Modifying the QMD/TMD immediately prior to upload
 * - Changing a field in CUDA's stream struct that CUDA applies to the QMD/TMD
 * This table shows the mechanism used with each CUDA version:
 *   +-----------+---------------+---------------+--------------+
 *   |  Version  |  Global Mask  |  Stream Mask  |  Next Mask   |
 *   +-----------+---------------+---------------+--------------+
 *   | 8.0-12.8  | TMD/QMD Hook  | stream struct | TMD/QMD Hook |
 *   | 6.5-7.5   | TMD/QMD Hook  | N/A           | TMD/QMD Hook |
 *   +-----------+---------------+---------------+--------------+
 * "N/A" indicates that a mask type is unsupported on that CUDA version.
 * Please contact the authors if support is needed for a particular feature on
 * an older CUDA version. Support for those is unimplemented, not impossible.
 *
 * An old implementation of this file effected the global mask on CUDA 10.2 by
 * changing a field in CUDA's global struct that CUDA applies to the QMD/TMD.
 * That implementation was extraordinarily complicated, and was replaced in
 * 2024 with a more-backward-compatible way of hooking the TMD/QMD.
 * View the old implementation via Git: `git show aa63a02e:libsmctrl.c`.
 */
#include <cuda.h>

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "libsmctrl.h"

// In functions that do not return an error code, we favor terminating with an
// error rather than merely printing a warning and continuing.
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

/*** QMD/TMD-based SM Mask Control via Debug Callback. ***/

// Tested working on x86_64 CUDA 6.5, 9.1, and various 10+ versions
// (No testing attempted on pre-CUDA-6.5 versions)
// Values for the following three lines can be extracted by tracing CUPTI as
// it interects with libcuda.so to set callbacks.
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
// These callback descriptors appear to intercept the TMD/QMD late enough that
// CUDA has already applied the per-stream mask from its internal data
// structures, allowing us to override it with the next mask.
#define QMD_DOMAIN 0xb
#define QMD_PRE_UPLOAD 0x1
// Global mask (applies across all threads)
static uint64_t g_sm_mask = 0;
// Next mask (applies per-thread)
static __thread uint64_t g_next_sm_mask = 0;
// Flag value to indicate if setup has been completed
static bool sm_control_setup_called = false;

// v1 has been removed---it intercepted the TMD/QMD too early, making it
// impossible to override the CUDA-injected stream mask with the next mask.
static void control_callback_v2(void *ukwn, int domain, int cbid, const void *in_params) {
	// ***Only tested on platforms with 64-bit pointers.***
	// The first 8-byte element in `in_params` appears to be its size. `in_params`
	// must have at least five 8-byte elements for index four to be valid.
	if (*(uint32_t*)in_params < 5 * sizeof(void*))
		abort(1, 0, "Unsupported CUDA version for callback-based SM masking. Aborting...");
	// The fourth 8-byte element in `in_params` is a pointer to the TMD. Note
	// that this fourth pointer must exist---it only exists when the first
	// 8-byte element of `in_params` is at least 0x28 (checked above).
	void* tmd = *((void**)in_params + 4);
	if (!tmd)
		abort(1, 0, "TMD allocation appears NULL; likely forward-compatibilty issue.\n");

	uint32_t *lower_ptr, *upper_ptr;

	// The location of the TMD version field seems consistent across versions
	uint8_t tmd_ver = *(uint8_t*)(tmd + 72);

	if (tmd_ver >= 0x40) {
		// TMD V04_00 is used starting with Hopper to support masking >64 TPCs
		lower_ptr = tmd + 304;
		upper_ptr = tmd + 308;
		// XXX: Disable upper 64 TPCs until we have ...next_mask_ext and
		//      ...global_mask_ext
		*(uint32_t*)(tmd + 312) = -1;
		*(uint32_t*)(tmd + 316) = -1;
		// An enable bit is also required
		*(uint32_t*)tmd |= 0x80000000;
	} else if (tmd_ver >= 0x16) {
		// TMD V01_06 is used starting with Kepler V2, and is the first to
		// support TPC masking
		lower_ptr = tmd + 84;
		upper_ptr = tmd + 88;
	} else {
		// TMD V00_06 is documented to not support SM masking
		abort(1, 0, "TMD version %04o is too old! This GPU does not support SM masking.\n", tmd_ver);
	}

	// Setting the next mask overrides both per-stream and global masks
	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}

	//fprintf(stderr, "Final SM Mask (lower): %x\n", *lower_ptr);
	//fprintf(stderr, "Final SM Mask (upper): %x\n", *upper_ptr);
}

static void setup_sm_control_callback() {
	int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
	int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
	uintptr_t* tbl_base;
	uint32_t my_hndl;
	// Avoid race conditions (setup should only run once)
	if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
		return;

#if CUDA_VERSION <= 6050
	// Verify supported CUDA version
	// It's impossible for us to run with a version of CUDA older than we were
	// built by, so this check is excluded if built with CUDA > 6.5.
	int ver = 0;
	cuDriverGetVersion(&ver);
	if (ver < 6050)
		abort(1, ENOSYS, "Global or next masking requires at least CUDA 6.5; "
		                 "this application is using CUDA %d.%d",
		                 ver / 1000, (ver % 100));
#endif

	// Set up callback
	cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
	uintptr_t subscribe_func_addr = *(tbl_base + 3);
	uintptr_t enable_func_addr = *(tbl_base + 6);
	subscribe = (typeof(subscribe))subscribe_func_addr;
	enable = (typeof(enable))enable_func_addr;
	int res = 0;
	res = subscribe(&my_hndl, control_callback_v2, NULL);
	if (res)
		abort(1, 0, "Error subscribing to launch callback. CUDA returned error code %d.", res);
	res = enable(1, my_hndl, QMD_DOMAIN, QMD_PRE_UPLOAD);
	if (res)
		abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
}

// Set default mask for all launches
void libsmctrl_set_global_mask(uint64_t mask) {
	setup_sm_control_callback();
	g_sm_mask = mask;
}

// Set mask for next launch from this thread
int libsmctrl_set_next_mask(uint64_t mask) {
	setup_sm_control_callback();
	g_next_sm_mask = mask;
	return 0;
}


/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

// Offsets for the stream struct on x86_64
// No offset appears to work with CUDA 6.5 (tried 0x0--0x1b4 w/ 4-byte step)
// 6.5 tested on 340.118
#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
// CUDA 9.0 and 9.1 use the same offset
// 9.1 tested on 390.157
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x244
// CUDA 10.0, 10.1 and 10.2 use the same offset
// 10.1 tested on 418.113
// 10.2 tested on 440.100, 440.82, 440.64, and 440.36
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
// 11.4 tested on 470.223.02
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
// 11.8 tested on 520.56.06
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset
// 12.0 tested on 525.147.05
#define CU_12_2_MASK_OFF 0x4e4
// 12.2 tested on 535.129.03
#define CU_12_3_MASK_OFF 0x49c
// 12.3 tested on 545.29.06
#define CU_12_4_MASK_OFF 0x4ac
// 12.4 tested on 550.54.14 and 550.54.15
#define CU_12_5_MASK_OFF 0x4ec
// CUDA 12.5 and 12.6 use the same offset
// 12.5 tested on 555.58.02
// 12.6 tested on 560.35.03
#define CU_12_7_MASK_OFF 0x4fc
// CUDA 12.7 and 12.8 use the same offset
// 12.7 tested on 565.77
// 12.8 tested on 570.124.06

// Offsets for the stream struct on Jetson aarch64
#define CU_9_0_MASK_OFF_JETSON 0x128
// 9.0 tested on Jetpack 3.x (TX2, Nov 2023)
#define CU_10_2_MASK_OFF_JETSON 0x24c
// 10.2 tested on Jetpack 4.x (AGX Xaver and TX2, Nov 2023)
#define CU_11_4_MASK_OFF_JETSON 0x394
// 11.4 tested on Jetpack 5.x (AGX Orin, Nov 2023)
// TODO: 11.8, 12.0, 12.1, and 12.2 on Jetpack 5.x via compatibility packages
#define CU_12_2_MASK_OFF_JETSON 0x50c
// 12.2 tested on Jetpack 6.x (AGX Orin, Dec 2024)
#define CU_12_4_MASK_OFF_JETSON 0x4c4
// 12.4 tested on Jetpack 6.x with cuda-compat-12-4 (AGX Orin, Dec 2024)
#define CU_12_5_MASK_OFF_JETSON 0x50c
// 12.5 tested on Jetpack 6.x with cuda-compat-12-5 (AGX Orin, Dec 2024)
#define CU_12_6_MASK_OFF_JETSON 0x514
// 12.6 tested on Jetpack 6.x with cuda-compat-12-6 (AGX Orin, Dec 2024)

// Used up through CUDA 11.8 in the stream struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
};

// Used starting with CUDA 12.0 in the stream struct
struct stream_sm_mask_v2 {
	uint32_t enabled;
	uint32_t mask[4];
};

// Check if this system has a Parker SoC (TX2/PX2 chip)
// (CUDA 9.0 behaves slightly different on this platform.)
// @return 1 if detected, 0 if not, -cuda_err on error
#if __aarch64__
int detect_parker_soc() {
	int cap_major, cap_minor, err, dev_count;
	if (err = cuDeviceGetCount(&dev_count))
		return -err;
	// As CUDA devices are numbered by order of compute power, check every
	// device, in case a powerful discrete GPU is attached (such as on the
	// DRIVE PX2). We detect the Parker SoC via its unique CUDA compute
	// capability: 6.2.
	for (int i = 0; i < dev_count; i++) {
		if (err = cuDeviceGetAttribute(&cap_minor,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		                               i))
			return -err;
		if (err = cuDeviceGetAttribute(&cap_major,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		                               i))
			return -err;
		if (cap_major == 6 && cap_minor == 2)
			return 1;
	}
	return 0;
}
#endif // __aarch64__

// Should work for CUDA 8.0 through 12.6
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
int libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	// When the old API is used on GPUs with over 64 TPCs, disable all TPCs >64
	uint128_t full_mask = -1;
	full_mask <<= 64;
	full_mask |= mask;
	return libsmctrl_set_stream_mask_ext(stream, full_mask);
}

int libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask = NULL;
	struct stream_sm_mask_v2* hw_mask_v2 = NULL;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
#if __x86_64__
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010: {
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
		break;
	}
	case 9020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_2_MASK_OFF);
		break;
	case 10000:
	case 10010:
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_0_MASK_OFF);
		break;
	case 11000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_0_MASK_OFF);
		break;
	case 11010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_1_MASK_OFF);
		break;
	case 11020:
	case 11030:
	case 11040:
	case 11050:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_2_MASK_OFF);
		break;
	case 11060:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_6_MASK_OFF);
		break;
	case 11070:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_7_MASK_OFF);
		break;
	case 11080:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF);
		break;
	case 12000:
	case 12010:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	case 12020:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF);
		break;
	case 12030:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_3_MASK_OFF);
		break;
	case 12040:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_4_MASK_OFF);
		break;
	case 12050:
	case 12060:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_5_MASK_OFF);
		break;
	case 12070:
	case 12080:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_7_MASK_OFF);
		break;
#elif __aarch64__
	case 9000: {
		// Jetson TX2 offset is slightly different on CUDA 9.0.
		// Only compile the check into ARM64 builds.
		// TODO: Always verify Jetson-board-only on aarch64.
		int is_parker;
		const char* err_str;
		if ((is_parker = detect_parker_soc()) < 0) {
			cuGetErrorName(-is_parker, &err_str);
			abort(1, 0, "While performing platform-specific "
			            "compatibility checks for stream masking, "
			            "CUDA call failed with error '%s'.", err_str);
		}

		if (!is_parker)
			abort(1, 0, "Not supported on non-Jetson aarch64.");
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF_JETSON);
		break;
	}
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_2_MASK_OFF_JETSON);
		break;
	case 11040:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_4_MASK_OFF_JETSON);
		break;
	case 12020:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF_JETSON);
		break;
	case 12040:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_4_MASK_OFF_JETSON);
		break;
	case 12050:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_5_MASK_OFF_JETSON);
		break;
	case 12060:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_6_MASK_OFF_JETSON);
		break;
#endif
	}

	// For experimenting to determine the right mask offset, set the MASK_OFF
	// environment variable (positive and negative numbers are supported)
	char* mask_off_str = getenv("MASK_OFF");
	if (mask_off_str) {
		int off = atoi(mask_off_str);
		fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.2 base %#x "
				"(total off: %#x)\n", off, CU_12_2_MASK_OFF, CU_12_2_MASK_OFF + off);
		if (CU_12_2_MASK_OFF + off < 0)
			abort(1, 0, "Total offset cannot be less than 0! Aborting...");
		// +4 bytes to convert a mask found with this for use with hw_mask
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF + off);
	}

	// Mask layout changed with CUDA 12.0 to support large Hopper/Ada GPUs
	if (hw_mask) {
		hw_mask->upper = mask >> 32;
		hw_mask->lower = mask;
	} else if (hw_mask_v2) {
		hw_mask_v2->enabled = 1;
		hw_mask_v2->mask[0] = mask;
		hw_mask_v2->mask[1] = mask >> 32;
		hw_mask_v2->mask[2] = mask >> 64;
		hw_mask_v2->mask[3] = mask >> 96;
	} else {
		abort(1, 0, "Stream masking unsupported on this CUDA version (%d), and"
		            " no fallback MASK_OFF set!", ver);
	}
	return 0;
}

/* INFORMATIONAL FUNCTIONS */

// Read an integer from a file in `/proc`
static int read_int_procfile(char* filename, uint64_t* out) {
	char f_data[18] = {0};
	size_t ret;
	int fd = open(filename, O_RDONLY);
	if (fd == -1)
		return errno;
	ret = read(fd, f_data, 18);
	if (ret == -1)
		return errno;
	close(fd);
	*out = strtoll(f_data, NULL, 16);
	return 0;
}

// We support up to 64 TPCs, up to 12 GPCs per GPU, and up to 16 GPUs.
// TODO: Handle GPUs with greater than 64 TPCs (e.g. some H100 variants)
static uint64_t tpc_mask_per_gpc_per_dev[16][12];
// Output mask is vtpc-indexed (virtual TPC)
int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev) {
	uint32_t i, j, vtpc_idx = 0;
	uint64_t gpc_mask, num_tpc_per_gpc, max_gpcs, gpc_tpc_mask;
	int err;
	char filename[100];
	*num_enabled_gpcs = 0;
	// Maximum number of GPCs supported for this chip
	snprintf(filename, 100, "/proc/gpu%d/num_gpcs", dev);
	if (err = read_int_procfile(filename, &max_gpcs)) {
		fprintf(stderr, "libsmctrl: nvdebug module must be loaded into kernel before "
				"using libsmctrl_get_*_info() functions\n");
		return err;
	}
	// TODO: handle arbitrary-size GPUs
	if (dev > 16 || max_gpcs > 12) {
		fprintf(stderr, "libsmctrl: GPU possibly too large for preallocated map!\n");
		return ERANGE;
	}
	// Set bit = disabled GPC
	snprintf(filename, 100, "/proc/gpu%d/gpc_mask", dev);
	if (err = read_int_procfile(filename, &gpc_mask))
		return err;
	snprintf(filename, 100, "/proc/gpu%d/num_tpc_per_gpc", dev);
	if (err = read_int_procfile(filename, &num_tpc_per_gpc))
		return err;
	// For each enabled GPC
	for (i = 0; i < max_gpcs; i++) {
		// Skip this GPC if disabled
		if ((1 << i) & gpc_mask)
			continue;
		(*num_enabled_gpcs)++;
		// Get the bitstring of TPCs disabled for this GPC
		// Set bit = disabled TPC
		snprintf(filename, 100, "/proc/gpu%d/gpc%d_tpc_mask", dev, i);
		if (err = read_int_procfile(filename, &gpc_tpc_mask))
			return err;
		uint64_t* tpc_mask = &tpc_mask_per_gpc_per_dev[dev][*num_enabled_gpcs - 1];
		*tpc_mask = 0;
		for (j = 0; j < num_tpc_per_gpc; j++) {
				// Skip disabled TPCs
				if ((1 << j) & gpc_tpc_mask)
					continue;
				*tpc_mask |= (1ull << vtpc_idx);
				vtpc_idx++;
		}
	}
	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev[dev];
	return 0;
}

int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev) {
	uint32_t num_gpcs;
	uint64_t* tpcs_per_gpc;
	int res;
	if (res = libsmctrl_get_gpc_info(&num_gpcs, &tpcs_per_gpc, dev))
		return res;
	*num_tpcs = 0;
	for (int gpc = 0; gpc < num_gpcs; gpc++) {
		*num_tpcs += __builtin_popcountl(tpcs_per_gpc[gpc]);
	}
	return 0;
}

// @param dev Device index as understood by CUDA **can differ from nvdebug idx**
// This implementation is fragile, and could be incorrect for odd GPUs
int libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev) {
	int num_sms, major, minor, res = 0;
	const char* err_str;
	if (res = cuInit(0))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev))
		goto abort_cuda;
	// SM masking only works on sm_35+
	if (major < 3 || (major == 3 && minor < 5))
		return ENOTSUP;
	// Everything newer than Pascal (as of Hopper) has 2 SMs per TPC, as well
	// as the P100, which is uniquely sm_60
	int sms_per_tpc;
	if (major > 6 || (major == 6 && minor == 0))
		sms_per_tpc = 2;
	else
		sms_per_tpc = 1;
	// It looks like there may be some upcoming weirdness (TPCs with only one SM?)
	// with Hopper
	if (major >= 9)
		fprintf(stderr, "libsmctrl: WARNING, TPC masking is untested on Hopper,"
				" and will likely yield incorrect results! Proceed with caution.\n");
	*num_tpcs = num_sms/sms_per_tpc;
	return 0;
abort_cuda:
	cuGetErrorName(res, &err_str);
	fprintf(stderr, "libsmctrl: CUDA call failed due to %s. Failing with EIO...\n", err_str);
	return EIO;
}

int libsmctrl_make_mask(uint64_t *result, uint32_t low, uint32_t high_exlusive) {
  uint64_t mask = 0ULL;
  int num_enabled = high_exlusive - low;
  for (int i = 0; i < num_enabled; ++i) {
    mask = (mask << 1) | 1ULL;
  }
  for (int i = 0; i < low; ++i) {
    mask <<= 1;
  }
  *result = ~mask;
  return 0;
}

int libsmctrl_make_mask_ext(uint128_t *result, uint32_t low, uint32_t high_exlusive) {
  uint128_t mask = 0ULL;
  int num_enabled = high_exlusive - low;
  for (int i = 0; i < num_enabled; ++i) {
    mask = (mask << 1) | 1ULL;
  }
  for (int i = 0; i < low; ++i) {
    mask <<= 1;
  }
  *result = ~mask;
  return 0;
}