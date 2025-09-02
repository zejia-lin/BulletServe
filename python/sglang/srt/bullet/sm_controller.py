import ctypes, ctypes.util
import os
import logging
from functools import wraps
import time
from typing import Literal, Callable, Any
from dataclasses import dataclass
import argparse
from enum import Enum

import pandas as pd
import torch

from sglang.srt.bullet_utils import BASE, TOTAL_TPCS
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


@dataclass
class ScheduleBudget:
    prefill_ratio: float
    decode_ratio: float


def _check_ret_code(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rets = func(*args, **kwargs)
        if isinstance(rets, tuple):
            ret_code = rets[0]
            ret_res = rets[1]
        else:
            ret_code = rets or 0
            ret_res = None
        if ret_code != 0:
            raise OSError(ret_code, f"{os.strerror(ret_code)} in {func.__name__}")
        return ret_res
    return wrapper


class c_uint128(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]
    
    def __init__(self, value=0):
        super().__init__()
        if isinstance(value, int):
            self.low = value & 0xFFFFFFFFFFFFFFFF
            self.high = (value >> 64) & 0xFFFFFFFFFFFFFFFF
        else:
            self.low = 0
            self.high = 0
    
    @property
    def value(self):
        return self.low | (self.high << 64)



class _LibSMCtrl:
    def __init__(self, libsmctrl_path):
        device_props = torch.cuda.get_device_properties()
        self.total_sms = device_props.multi_processor_count
        if not ctypes.util.find_library("libsmctrl"):
            libsmctrl_path = f"{BASE}/csrc/build/libsmctrl.so"
        try:
            self.lib = ctypes.CDLL(libsmctrl_path)
        except Exception as e:
            raise OSError(f"Failed to find or load libsmctrl.so: {e}, please reference to README to build it.")

    @_check_ret_code
    def set_global_mask(self, mask: int) -> None:
        if self.total_sms >= 128:
            logger.error(f"PYSMCTRL: total_sms {self.total_sms} >= 128, this method is problematic")
            raise ValueError(f"PYSMCTRL: total_sms {self.total_sms} >= 128, this method is problematic")
        return self.lib.libsmctrl_set_global_mask(ctypes.c_uint64(mask))

    @_check_ret_code
    def set_stream_mask(self, stream: torch.cuda.Stream, mask: int) -> None:
        if self.total_sms < 128:
            return self.lib.libsmctrl_set_stream_mask(ctypes.c_void_p(stream.cuda_stream), ctypes.c_uint64(mask))
        else:
            return self.lib.libsmctrl_set_stream_mask_ext(ctypes.c_void_p(stream.cuda_stream), c_uint128(mask))

    @_check_ret_code
    def set_next_mask(self, mask: int) -> None:
        return self.lib.libsmctrl_set_next_mask(mask)
    
    @_check_ret_code
    def get_tpc_count(self, cuda_dev: int) -> int:
        num_tpcs = ctypes.c_uint32()
        ret = self.lib.libsmctrl_get_tpc_info_cuda(ctypes.byref(num_tpcs), cuda_dev)
        return ret, num_tpcs.value

    @_check_ret_code
    def make_mask(self, low: int, high_exclusive: int) -> int:
        result = ctypes.c_uint64()
        ret = self.lib.libsmctrl_make_mask(ctypes.byref(result), low, high_exclusive)
        return ret, result.value
    
    @_check_ret_code
    def validate_stream_mask(self, stream: torch.cuda.Stream, low: int, high_exclusive: int, echo = False) -> None:
        stream_ptr = ctypes.c_void_p(stream.cuda_stream)
        ret = self.lib.libsmctrl_validate_stream_mask(stream_ptr, low, high_exclusive, echo)
        return ret


class SMController():
    def __init__(self, libsmctrl_path: str = "libsmctrl.so"):
        if is_hip():
            self.set_stream_mask = lambda stream, low, hi: None
            return
        self.enabled = True
        self.lib = _LibSMCtrl(libsmctrl_path)
        self.total_tpcs = self.lib.get_tpc_count(torch.cuda.current_device())

    def set_global_mask(self, num_tpcs: int):
        logger.debug(f"PYSMCTRL set global {num_tpcs * 2} SMs")
        mask = self.lib.make_mask(0, num_tpcs)
        self.lib.set_global_mask(mask)

    def set_stream_mask(self, stream: torch.cuda.Stream, low: int, high_exclusive: int, reversed=False):
        if reversed:
            low, high_exclusive = self.total_tpcs - high_exclusive, self.total_tpcs - low
            logger.debug(f"PYSMCTRL set {high_exclusive * 2}-{low * 2} SMs for stream {stream}")
        else:
            logger.debug(f"PYSMCTRL set {low * 2}-{high_exclusive * 2} SMs for stream {stream}")
        mask = self.lib.make_mask(low, high_exclusive)
        self.lib.set_stream_mask(stream, mask)
        return mask
        
    def validate_all_masks(self, echo = False):
        self.lib.validate_stream_mask(self.prefill_stream, *self.prefill_range, echo)
        self.lib.validate_stream_mask(self.decode_stream, *self.decode_range, echo)


if __name__ == "__main__":
    smctrl = SMController()
    smctrl.set_stream_mask(smctrl.prefill_stream, 0, TOTAL_TPCS)
    repeat = 10000
    times = []
    for i in range(repeat):
        st = time.perf_counter()
        a = list()
        ed = time.perf_counter()
        t = ed - st
        times.append(t * 1000)
    # ed = time.perf_counter()
    # t = ed - st
    # print(t, repeat, f"{t / repeat * 1000} ms")
    print(pd.DataFrame(times).describe(percentiles=[0.5, 0.9, 0.99]))