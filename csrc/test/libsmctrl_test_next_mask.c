// Copyright 2024 Joshua Bakita
// Test libsmctrl_set_next_mask().
// All types of partitioning use the same test, so this file is trival.

#include "libsmctrl_test_mask_shared.h"

int main() {
  return test_constrained_size_and_location(PARTITION_NEXT);
}

