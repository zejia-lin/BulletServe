import dataclasses
from datetime import datetime
import os
from collections import deque
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Literal, Union
import logging

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


@dataclass
class PredictorInfoEntry:

    _: dataclasses.KW_ONLY
    phase: Union[Literal["prefill"], Literal["decode"]]
    prefill_len: int
    decode_bs: int
    decode_tokens: int
    prefill_tpc: int
    decode_tpc: int
    predict_ms: float  # predicted elapsed time
    start_timstamp: float = None
    end_timestamp: float = None
    policy: str = None
    prefill_len_after: int = None
    gpu_ms: float = None
    rids: List[str] = None
    # recv_overhead: float = None

    def __post_init__(self):
        self.start_timstamp = time.time()
        if self.phase not in ["prefill", "decode"]:
            raise ValueError("phase must be either 'prefill' or 'decode'")

    def __lt__(self, other: "PredictorInfoEntry"):
        pass


class PredictorInfos:
    def __init__(self, *, enabled: bool = True):
        self.entries: List[PredictorInfoEntry] = []
        self.enabled = enabled

    def add_entry(self, entry: PredictorInfoEntry, end_timestamp, gpu_ms=-1):
        if not self.enabled:
            return
        entry.end_timestamp = end_timestamp
        entry.gpu_ms = gpu_ms
        self.entries.append(entry)

    def dump(self, filename, echo=False):
        if not self.enabled:
            return False
        if len(self.entries) == 0:
            if echo:
                logger.info("No predictor infos to dump.")
            return False

        logger.info(filename)
        if not os.path.exists(os.path.dirname(filename)):
            logger.info(f"Make directory {os.path.dirname(filename)}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        write_header = not os.path.exists(filename)
        towrite = []
        with open(filename, "a") as fout:
            if write_header:
                towrite.append(",".join(dataclasses.asdict(self.entries[0]).keys()))
            else:
                towrite.append("\n")
            for entry in self.entries:
                towrite.append(",".join(map(str, dataclasses.asdict(entry).values())))
            fout.write("\n".join(towrite))
        return True

    def clear(self):
        self.entries = []


@dataclass
class ReqTimingState:
    """Timing of a request. All times are in ms"""

    _: dataclasses.KW_ONLY
    input_len: int
    arrive_tstamp: float  # the timestamp the request arrives at recv_requests engine loop
    output_len: int = 0
    queue_time: float = None
    ttft: float = None
    ttft_estimate: float = None  # estimated total time, including waiting time
    ttft_norm_estimate: float = None
    ttft_gpu: float = None
    ttft_gpu_estimate: float = None  # estimated time of GPU execution, not include waiting time
    decode_duration: float = None
    tpot: float = None
    transmit_time: float = None
    recv_tstamp: float = None

    def __post_init__(self):
        self.ttft_gpu_estimate = -1

    def estimate_ttft(self, tstamp):
        self.queue_time = (tstamp - self.arrive_tstamp) * 1000
        self.ttft_estimate = self.ttft_gpu_estimate + self.queue_time
        self.ttft_norm_estimate = self.ttft_estimate / self.input_len

    def step(self, tstamp, forward_mode: ForwardMode, gpu_time=None):
        if forward_mode.is_prefill():
            self.update_ttft(tstamp, gpu_time)
        else:
            self.update_tpot(tstamp)

    def update_ttft(self, tstamp, gpu_time):
        # logger.info(f"tstamp {tstamp}, arrive_tstamp {self.arrive_tstamp}")
        self.ttft = (tstamp - self.arrive_tstamp) * 1000
        self.ttft_gpu = gpu_time
        ttft_norm = self.ttft / self.input_len
        self.output_len = 1 if self.output_len == 0 else self.output_len

    def update_tpot(self, tstamp):
        self.output_len += 1
        self.decode_duration = tstamp - self.arrive_tstamp - self.ttft
        self.tpot = self.decode_duration / (self.output_len - 1) * 1000
        # logger.info(f"Decode {self.input_len}+{self.output_len} tokens in {self.decode_duration:.3f} s, "
        #             f"TPOT {self.tpot:.3f} ms")

    def key_for_sort(self, ttft_norm_slo):
        return -self.ttft_norm_estimate


@dataclass
class ReqTimingStateDict:
    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self.dct: Dict[str, ReqTimingState] = {}
        self.MAXLEN = 3
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.MAXLEN)]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.MAXLEN)]
        self.idx = 0
        self.que = deque(maxlen=self.MAXLEN)

    def __setitem__(self, key, value):
        self.dct[key] = value

    def __getitem__(self, key):
        return self.dct[key]

    def begin_forward(self, stream, rids: List[str], enable_gpu_timing=False):
        if not self.enabled:
            return
        self.idx = (self.idx + 1) % self.MAXLEN
        start = self.starts[self.idx]
        if enable_gpu_timing:
            start.record(stream)
        self.que.append((self.idx, rids))

    def end_forward(self, stream, enable_gpu_timing=False):
        if not self.enabled:
            return
        end = self.ends[self.que[0][0]]
        if enable_gpu_timing:
            end.record(stream)

    def step(self, phase: ForwardMode, enable_gpu_timing=False) -> float:
        """Update the timings for current requests, return GPU elapsed ms of CUDA events"""
        if not self.enabled:
            return -1
        _t = time.time()
        if not self.que:
            return -1
        idx, rids = self.que.popleft()
        start = self.starts[idx]
        end = self.ends[idx]
        elapsed_ms = -1
        if enable_gpu_timing:
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        for rid in rids:
            if rid in self.dct:
                self.dct[rid].step(_t, phase, elapsed_ms)
        return elapsed_ms

    def dump(self, filename, clear=True):
        if not self.enabled:
            return
        if not self.dct:
            return

        field_names = ["rid"] + [field.name for field in dataclasses.fields(ReqTimingState)]
        write_header = not os.path.exists(filename)
        if write_header:
            header = [",".join(field_names)]
        else:
            header = ["\n"]
        lines = header + [
            f"{rid}," + ",".join(map(str, dataclasses.asdict(entry).values()))
            for rid, entry in self.dct.items()
        ]

        # Single write operation
        with open(filename, "a") as fout:
            fout.write("\n".join(lines) + "\n")

        if clear:
            self.clear()

    def clear(self):
        if not self.enabled:
            return
        keys = list(self.dct.keys())[:-10]
        for key in keys:
            self.dct.pop(key)
