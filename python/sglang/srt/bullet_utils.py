from dataclasses import dataclass, fields
import dataclasses
from functools import wraps
import os
import random
import time
import shutil
from typing import List
import torch
import json

import psutil

from sglang.srt.utils import is_port_available
import subprocess
import re
from sglang.srt.utils import is_hip

def get_parant_dir(path: str, level: int=1):
    for i in range(level):
        path = os.path.dirname(path)
    return path


BASE = get_parant_dir(__file__, 4)


def get_gpu_type():
    """Detect GPU type (A100 or H100) using nvidia-smi output, without torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        gpu_names = result.stdout.strip().split('\n')
        for name in gpu_names:
            lname = name.lower()
            match = re.search(r"nvidia [a-z]\d+", lname)
            if match:
                return match.group(0).split()[1]
        return "unknown"
    except Exception:
        return "unknown"


def get_num_tpcs():
    """Get the number of TPCs based on the GPU type."""
    if is_hip():
        return 120
    gpu_type = get_gpu_type()
    tpcs = {
        "a100": 54,
        "h100": 66,
        "a800": 54,
        "h20": 39
    }
    return tpcs[gpu_type]


TOTAL_TPCS = get_num_tpcs()


def allocate_port_from_base(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            break
        if port < 60000:
            port += 42
        else:
            port -= 43


class FakeContext:
    def __init__(self, real_ctx) -> None:
        self.real_ctx = real_ctx

    def enable(self):
        self.__enter__ = self.real_ctx.__enter__
        self.__exit__ = self.real_ctx.__exit__
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def with_self_context(name: str):
    def decorator(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            with getattr(self, name):
                return f(self, *args, **kwargs)

        return wrapped

    return decorator


def with_lock():
    return with_self_context("_lock")


class with_context:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            with self.ctx:
                return f(*args, **kwargs)

        return wrapper


def make_noop(obj, name):
    """make an no-op method for an object"""
    setattr(obj, name, lambda *args, **kwargs: None)


def auto_increment_class(cls):
    """Auto increment the fields of a class similar to an Enum. This is for passing msg types in ZMQ"""
    cls = dataclass(cls)

    def init(self, *args, **kwargs):
        raise NotImplementedError("Cannot instantiate the auto_increment_class class")

    cls.__init__ = init
    unique_values = set()
    for idx, f in enumerate(fields(cls)):
        if idx in unique_values:
            raise ValueError(f"Duplicate value detected for attribute {f.name}")
        setattr(cls, f.name, idx)
        unique_values.add(idx)
    return cls


def make_timestamped_dir(basedir: str):
    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    new_dir = os.path.join(basedir, cur_time)
    os.mkdir(new_dir)
    return new_dir


def create_dataclass_from_dict(cls, dct):
    valid_fields = {field.name for field in dataclasses.fields(cls)}
    filtered_dct = {k: v for k, v in dct.items() if k in valid_fields}
    return cls(**filtered_dct)


def kill_process_on_port(port: int, sudo: bool):
    """Kill the process that is listening on the given port."""
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port:
                    if sudo:
                        os.system(f"sudo kill -9 {proc.info['pid']}")
                    else:
                        proc.kill()
                    print(
                        f"Killed process {proc.info['name']} (PID: {proc.info['pid']}) listening on port {port}"
                    )
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print(f"No process found listening on port {port}")


def remove_required_arg_from_parser(parser, name: str):
    for i in range(len(parser._actions)):
        if name in parser._actions[i].option_strings:
            parser._actions[i].required = False


class CudaTimer:

    def __init__(self, names: List):
        self.records = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.current_names = []
        
    def __call__(self, name: str):
        self.current_names.append(name)
        return self

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        name = self.current_names.pop()
        self.records.append((name, self.elapsed_time_ms))

    def dump(self, filename: str):
        with open(filename, "a+") as f:
            json.dump(self.records, f)
