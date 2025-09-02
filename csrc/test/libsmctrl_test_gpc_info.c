// Copyright 2024 Joshua Bakita
#define _GNU_SOURCE
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libsmctrl.h"

int main(int argc, char** argv) {
	uint32_t num_gpcs = 0, num_tpcs = 0;
	uint64_t* masks = NULL;
	int res;
	int gpu_id = 0;
	// Optionally support specifying the GPU ID to query via an argument
	// Important: This GPU ID must match the ID used by the nvdebug module. See
	//            the documentation on libsmctrl_get_gpc_info() for details.
	if (argc > 2 || (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))) {
		fprintf(stderr, "Usage: %s <nvdebug GPU ID>\n", argv[0]);
		return 1;
	}
	if (argc > 1)
		gpu_id = atoi(argv[1]);
	if ((res = libsmctrl_get_gpc_info(&num_gpcs, &masks, gpu_id)) != 0)
		error(1, res, "libsmctrl_get_gpc_info() failed");
	printf("%s: GPU%d has %d enabled GPCs.\n", program_invocation_name, gpu_id, num_gpcs);
	for (int i = 0; i < num_gpcs; i++) {
		num_tpcs += __builtin_popcountl(masks[i]);
		printf("%s: Mask of %d TPCs associated with GPC %d: %#018lx\n", program_invocation_name, __builtin_popcountl(masks[i]), i, masks[i]);
	}
	printf("%s: Total of %u enabled TPCs.\n", program_invocation_name, num_tpcs);
	return 0;
}
