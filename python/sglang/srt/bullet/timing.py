import functools
from typing import Dict, List, Tuple, Optional
import torch
from collections import defaultdict, deque
from contextlib import contextmanager
import json

import pandas as pd


__all__ = ["GLOBAL_TIMER", "GlobalTimer"]


class GlobalTimer:
    """Asynchronous global timer using pre-allocated CUDA events."""

    def __init__(self, max_ops: int = 10000):
        self.enabled = False
        self.max_events = max_ops
        self.forward_id = 0
        self.results = []
        self.extra_info = {}

        # Pre-allocate CUDA events
        self.event_pool = deque()
        self.used_events = []  # Store (start_event, end_event, metadata) tuples

        # Initialize event pool
        self._init_event_pool()

    def _init_event_pool(self):
        """Pre-allocate CUDA events for timing."""
        for _ in range(self.max_events * 2):  # Need pairs of start/end events
            event = torch.cuda.Event(enable_timing=True)
            self.event_pool.append(event)

    def _get_event_pair(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """Get a pair of events from the pool."""
        if len(self.event_pool) < 2:
            # If pool is exhausted, synchronize and recycle events
            self._synchronize_and_process()

        start_event = self.event_pool.popleft()
        end_event = self.event_pool.popleft()
        return start_event, end_event

    def _return_events(self, start_event: torch.cuda.Event, end_event: torch.cuda.Event):
        """Return events to the pool for reuse."""
        self.event_pool.append(start_event)
        self.event_pool.append(end_event)

    def enable(self):
        """Enable timing collection."""
        self.enabled = True

    def disable(self):
        """Disable timing collection."""
        self.enabled = False

    def clear(self):
        """Clear all recorded timings and reset state."""
        # Process any remaining events first
        if self.used_events:
            self._synchronize_and_process()

        self.results.clear()
        self.forward_id = 0
        self.extra_info.clear()

    def set_extra_info(self, info: dict):
        """Set extra info saved in the results"""
        self.extra_info = info

    @contextmanager
    def time_operation(
        self, operation_name: str, input_tensor: torch.Tensor = None, is_last_op: bool = False
    ):
        """Context manager to time an operation using CUDA events (asynchronous)."""
        if not self.enabled:
            yield
            return

        # Get event pair from pool
        start_event, end_event = self._get_event_pair()

        # Record batch size if tensor is provided
        batch_size = input_tensor.shape[0] if input_tensor is not None else -1

        # Store metadata for later processing
        metadata = {
            "op_type": operation_name,
            "batch_size": batch_size,
            "run_id": self.forward_id,
            "is_last_op": is_last_op,
            **self.extra_info
        }

        # Record start event
        start_event.record()

        try:
            yield
        finally:
            # Record end event (asynchronous)
            end_event.record()

            # Store events and metadata for later processing
            self.used_events.append((start_event, end_event, metadata))

            # self._synchronize_and_process()
            # print(metadata)

            # If this is the last operation of a forward pass, increment run_id
            if is_last_op:
                self.forward_id += 1

    def _synchronize_and_process(self, device=None):
        """Synchronize all pending events and calculate elapsed times."""
        if not self.used_events:
            return

        # Synchronize all events at once
        torch.cuda.synchronize(device)

        # Process all stored events
        for start_event, end_event, metadata in self.used_events:
            elapsed_time = start_event.elapsed_time(end_event)

            # Store result
            res = {
                "time_ms": elapsed_time,
                **metadata
            }
            res.pop("is_last_op", None)  # Remove is_last_op from metadata
            self.results.append(res)

            # Return events to pool
            self._return_events(start_event, end_event)

        # Clear used events list
        self.used_events.clear()

    def synchronize(self, device=None):
        """Manually synchronize and process all pending timing data."""
        self._synchronize_and_process(device)

    def decorator_all_reduce(self, operation_name="tp_all_reduce"):
        """Decorator to time all-reduce operations."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                with self.time_operation(operation_name, args[0] if args else None):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_stats(self):
        """Get timing statistics for all operations."""
        # Make sure all pending events are processed
        self._synchronize_and_process()

        # Group by operation type
        operation_data = defaultdict(list)
        batch_data = defaultdict(list)

        for result in self.results:
            operation_data[result["op_type"]].append(result["time_ms"])
            if result["batch_size"] > 0:
                batch_data[result["op_type"]].append(result["batch_size"])

        stats = {}
        for operation, times in operation_data.items():
            if times:
                batches = batch_data[operation]
                stats[operation] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "avg_batch_size": sum(batches) / len(batches) if batches else 0,
                    "min_batch_size": min(batches) if batches else 0,
                    "max_batch_size": max(batches) if batches else 0,
                }
        return stats

    def get_raw_data(self) -> List[Dict]:
        """Get raw timing data as a list of dictionaries."""
        # Make sure all pending events are processed
        self._synchronize_and_process()
        return self.results

    def get_dataframe(self):
        """Get timing data as a pandas DataFrame."""
        data = self.get_raw_data()
        return pd.DataFrame(data)

    def get_json_data(self):
        """Get timing data as a space-efficient JSON object grouped by run_id."""
        # Make sure all pending events are processed
        self._synchronize_and_process()

        # Group data by run_id
        if len(self.results) == 0:
            return []
        keys = list(self.results[0].keys())
        keys.remove("run_id")
        grouped_data = defaultdict(lambda: {key: [] for key in keys})

        for result in self.results:
            run_id = result["run_id"]
            for key in keys:
                grouped_data[run_id][key].append(result[key])

        # Convert to the desired format
        json_data = []
        for run_id, data in grouped_data.items():
            json_data.append(
                {
                    "run_id": run_id,
                    **data,
                }
            )

        return json_data

    def get_jsonl(self):
        """Get timing data as JSONL format (one JSON object per line)."""

        json_data = self.get_json_data()
        jsonl_lines = []

        for entry in json_data:
            jsonl_lines.append(json.dumps(entry))

        return "\n".join(jsonl_lines)

    def print_stats(self):
        """Print timing statistics."""
        stats = self.get_stats()
        if not self.enabled:
            print("Global timer is not enabled")
            return
        if not stats:
            print("No timing data collected")
            return

        print("\n=== Global Timer Statistics ===")
        for operation, stat in stats.items():
            print(f"{operation}:")
            print(f"  Count: {stat['count']}")
            print(f"  Total: {stat['total_ms']:.2f}ms")
            print(f"  Average: {stat['avg_ms']:.2f}ms")
            print(f"  Min: {stat['min_ms']:.2f}ms")
            print(f"  Max: {stat['max_ms']:.2f}ms")
            print(f"  Avg Batch Size: {stat['avg_batch_size']:.1f}")
            print(f"  Min Batch Size: {stat['min_batch_size']}")
            print(f"  Max Batch Size: {stat['max_batch_size']}")

    def get_pending_count(self) -> int:
        """Get number of pending (unsynchronized) timing operations."""
        return len(self.used_events)

    def __del__(self):
        """Cleanup: synchronize any remaining events before destruction."""
        if hasattr(self, "used_events") and self.used_events:
            try:
                self._synchronize_and_process()
            except:
                pass  # Ignore errors during cleanup


# Global timer instance
GLOBAL_TIMER = GlobalTimer(max_ops=1000)
