import ctypes
import sys

from sglang.srt.bullet_utils import TOTAL_TPCS

if sys.version_info < (3, 11):
    import enum
    from enum import Enum

    class StrEnum(str, Enum):
        """
        Enum where members are also (and must be) strings
        """

        def __new__(cls, value):
            if not isinstance(value, str):
                raise TypeError(f"{value!r} is not a string")
            return str.__new__(cls, value)

        def __str__(self):
            return str(self.value)

        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

    enum.StrEnum = StrEnum
else:
    from enum import StrEnum


import inspect
import os
from typing import Callable, Literal, Tuple

# import joblib
import numpy as np
import torch
import logging
import enum

from sglang.srt.bullet.shared_nparray import SharedNPArray
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS_ESTIMATION
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


class SelectedSMPolicy(enum.StrEnum):
    DEFAULT = "default"
    FIXED = "fixed"
    DYNAMIC = "dynamic"


def prop(onset: Callable[[object, int], None]) -> int:
    if not hasattr(prop, "_property_idx"):
        prop._property_idx = 0
    else:
        prop._property_idx += 1

    idx = prop._property_idx

    def getter(self):
        return self.array[idx]

    def setter(self, value):
        self.array[idx] = value
        if onset is not None:
            onset(self, value)

    return property(getter, setter)


class OnlineAverage:
    def __init__(self):
        self.hist = []
        self.hist_sum = 0
        self.avg = 0

    def add(self, value):
        if value == 0:
            return
        self.hist.append(value)
        self.hist_sum += value
        self.avg = self.hist_sum / len(self.hist)

    def clear(self):
        self.hist = []
        self.hist_sum = 0
        self.avg = 0


def onset_prefill_size(self: "SharedManager", value):
    if value == 0:
        return
    self.avg_prefill.add(value)


class SharedManager:

    prefill_size = prop(onset_prefill_size)
    decode_size = prop(None)
    decode_total_context = prop(None)
    rem_total_token_offset = prop(None)
    prefill_num_tpcs = prop(None)
    decode_num_tpcs = prop(None)
    num_queue_reqs = prop(None)
    rem_layers = prop(None)

    DTYPE = np.int32
    ARRAY_SIZE = prop._property_idx + 1

    def __init__(self, args: ServerArgs, *, create=False):
        self.shm_name = f"shared_array_{args.launcher_pid}"
        if create:
            initial_array = np.zeros(self.ARRAY_SIZE, dtype=self.DTYPE)
            self.shared = SharedNPArray.create(initial_array, self.shm_name)
        else:
            self.shared = SharedNPArray.rebuild(self.shm_name, (self.ARRAY_SIZE,), np.dtype(self.DTYPE).str)
        if hasattr(args, "predictor_param_file") and args.predictor_param_file:
            try:
                self.lib = ctypes.CDLL(args.predictor_param_file)
            except Exception as e:
                raise OSError(f"Failed to load predictor: {e}")
        else:
            self.lib = None
        self.array = self.shared.array
        self.args = args
        self.previous_decode_entry = None
        self.avg_prefill = OnlineAverage()
        self.avg_prefill_norm_ms = OnlineAverage()  # prefill normalized latency (include queue time)
        logger.info(f"Shared Manager initialized with shm_name: {self.shm_name}")

    def _setup_function_signatures(lib):
        # Setup predic_duration function
        lib.predic_duration.argtypes = [
            ctypes.c_char_p,  # const char* phase
            ctypes.c_int,  # int prefill_tpc
            ctypes.c_int,  # int prefill_size
            ctypes.c_int,  # int decode_size
            ctypes.c_int,  # int decode_total_context
            ctypes.c_int,  # int decode_tpc
            ctypes.c_int,  # int rem_layers
        ]
        lib.predic_duration.restype = ctypes.c_double

        lib.set_adaptive_num_tpcs.argtypes = [
            ctypes.c_int,  # int queue_ms
            ctypes.c_char_p,  # const char* phase
            ctypes.c_int,  # int prefill_tpc
            ctypes.c_int,  # int prefill_size
            ctypes.c_int,  # int decode_size
            ctypes.c_int,  # int decode_total_context
            ctypes.c_int,  # int decode_tpc
            ctypes.c_int,  # int rem_layers
        ]
        lib.set_adaptive_num_tpcs.restype = ctypes.c_int

    def predict_duration(
        self,
        phase,
        prefill_tpc=None,
        prefill_size=None,
        decode_size=None,
        decode_total_context=None,
        decode_tpc=None,
    ):
        prefill_tpc = prefill_tpc if prefill_tpc is not None else self.prefill_num_tpcs
        prefill_size = prefill_size if prefill_size is not None else self.prefill_size
        decode_size = decode_size if decode_size is not None else self.decode_size
        decode_total_context = (
            decode_total_context
            if decode_total_context is not None
            else self.decode_total_context / self.decode_size
        )
        decode_tpc = decode_tpc if decode_tpc is not None else self.decode_num_tpcs

        if not self.lib:
            return -1

        return self.lib.predic_duration(
            phase.encode("utf-8"), prefill_tpc, prefill_size, decode_size, decode_total_context, decode_tpc
        )

    def set_real_norm_ms(self, phase, duration_ms):
        if phase == "prefill":
            self.avg_prefill_norm_ms.add(duration_ms)
        # self.previous_decode_entry.append(duration_ms)

    def set_adaptive_decode_num_tpcs(self, queue_ms) -> Tuple[int, SelectedSMPolicy]:
        if self.prefill_size == 0:
            self.decode_num_tpcs = TOTAL_TPCS
            return TOTAL_TPCS, SelectedSMPolicy.DEFAULT

        if not self.args.enable_sm_partition:
            self.decode_num_tpcs = self.args.fixed_decode_tpcs
            return self.args.fixed_decode_tpcs, SelectedSMPolicy.FIXED

        return self.lib.set_adaptive_num_tpcs(
            int(queue_ms),
            b"decode",
            int(self.prefill_num_tpcs),
            int(self.prefill_size),
            int(self.decode_size),
            int(self.decode_total_context),
            int(self.decode_num_tpcs),
            int(self.rem_layers),
        ), SelectedSMPolicy.DYNAMIC

    def set_adaptive_prefill_num_tpcs(self, queue_ms) -> Tuple[int, SelectedSMPolicy]:
        if self.decode_size == 0:
            self.prefill_num_tpcs = TOTAL_TPCS
            return TOTAL_TPCS, SelectedSMPolicy.DEFAULT

        if not self.args.enable_sm_partition:
            self.prefill_num_tpcs = self.args.fixed_prefill_tpcs
            return self.args.fixed_prefill_tpcs, SelectedSMPolicy.FIXED

        return self.lib.set_adaptive_num_tpcs(
            int(queue_ms),
            b"prefill",
            int(self.prefill_num_tpcs),
            int(self.prefill_size),
            int(self.decode_size),
            int(self.decode_total_context),
            int(self.decode_num_tpcs),
            int(self.rem_layers),
        ), SelectedSMPolicy.DYNAMIC

    def update_running_tokens(self, batch: ScheduleBatch, new_token_ratio: float):
        if batch is not None and batch.reqs is not None:
            self.rem_total_token_offset = sum(
                [
                    min(
                        (r.sampling_params.max_new_tokens - len(r.output_ids)),
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    )
                    * new_token_ratio
                    for r in batch.reqs
                ]
            )

    def clear(self):
        self.avg_prefill.clear()
        self.avg_prefill_norm_ms.clear()


# Example Usage
def main():
    manager = SharedManager(create=True)

    manager.prefill_size = 5
    manager.decode_size = 10
    manager.decode_num_tpc = 2

    print("After Setting:")
    print("prefill_size =", manager.prefill_size)
    print("decode_size =", manager.decode_size)
    print("decode_num_tpc =", manager.decode_num_tpc)

    manager_rebuild = SharedManager(create=False)
    print("\nAfter Rebuilding:")
    print("prefill_size =", manager_rebuild.prefill_size)
    print("decode_size =", manager_rebuild.decode_size)
    print("decode_num_tpc =", manager_rebuild.decode_num_tpc)

    manager_rebuild.prefill_size = 500
    manager_rebuild.decode_size = 1000
    manager_rebuild.decode_num_tpc = 200

    print("\nAfter Updating:")
    print("prefill_size =", manager.prefill_size)
    print("decode_size =", manager.decode_size)
    print("decode_num_tpc =", manager.decode_num_tpc)

    print(manager, manager_rebuild)


if __name__ == "__main__":
    main()
