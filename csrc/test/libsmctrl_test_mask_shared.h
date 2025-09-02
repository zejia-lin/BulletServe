// Copyright 2023 Joshua Bakita
#ifdef __cplusplus
extern "C" {
#endif

enum partitioning_type {
  PARTITION_GLOBAL,
  PARTITION_STREAM,
  PARTITION_STREAM_OVERRIDE,
  PARTITION_NEXT,
  PARTITION_NEXT_OVERRIDE,
};

extern int test_constrained_size_and_location(enum partitioning_type part_type);

#ifdef __cplusplus
}
#endif
