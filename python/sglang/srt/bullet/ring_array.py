import os
import numpy as np
from typing import Callable, Optional, Union, Any, List, Tuple, TypeVar, Generic
import logging
import torch
import traceback

from sglang.srt.bullet.shared_nparray import SharedNPArray


logger = logging.getLogger(__name__)

T = TypeVar("T")


def wrapped_list_constructor(n: Tuple[int], dtype: Any) -> List[T]:
    assert len(n) == 1
    return [0] * n[0]


class SharedRingArray(Generic[T]):
    """
    A ring array implementation using SharedNPArray as the underlying container.

    This data structure acts like a fixed-size queue where new elements are added
    to the end and retrieved from the beginning. When the array is full, adding
    new elements overwrites the oldest elements.

    The underlying array is stored in shared memory, making it accessible
    across different processes.
    """

    def __init__(
        self,
        capacity: int,
        dtype: Any,
        *,
        name: str,
        create: bool,
        container_constructor: Callable[[int, Any], Any] = np.zeros,
    ):
        """
        Initialize an empty SharedRingArray.

        Args:
            capacity: Maximum number of elements the array can hold (used only for empty creation)
            dtype: Data type of the array elements (numpy dtype)
        """
        self.capacity = capacity
        self.dtype = dtype
        self.container_constructor = container_constructor
        if create:
            self.shared_array: SharedNPArray = SharedNPArray.create(
                np.array(range(capacity), dtype=dtype), name=name
            )
            self.metadata = SharedNPArray.create(np.array([0, capacity], dtype=np.int32), name=name + "_metadata")
            self.start_idx = 0
            self._available_size = capacity
        else:
            self.shared_array = SharedNPArray.rebuild(name, (capacity,), np.dtype(dtype).str)
            self.metadata = SharedNPArray.rebuild(name + "_metadata", (2,), np.dtype(np.int32).str)

        self.array = self.shared_array.array

    @property
    def start_idx(self) -> int:
        return self.metadata.array[0]

    @start_idx.setter
    def start_idx(self, value: int) -> None:
        self.metadata.array[0] = value

    @property
    def _available_size(self) -> int:
        return self.metadata.array[1]

    @_available_size.setter
    def _available_size(self, value: int) -> None:
        # traceback.print_stack()
        # logger.info(f"Setting available size from {self._available_size} to {value}")
        self.metadata.array[1] = value

    def share(self):
        """
        Get information needed to rebuild this array in another process.

        Returns:
            Tuple of (shm_name, shape, dtype_str)
        """
        return self.shared_array.share(), self.metadata.share()

    def __len__(self) -> int:
        """Return the **available** size of the array."""
        # traceback.print_stack(limit=5)
        # logger.info(f"@@@@@ __len__: {self._available_size}")
        return self._available_size

    def is_empty(self) -> bool:
        """Check if the array is empty."""
        return self._available_size == self.capacity

    def is_full(self) -> bool:
        """Check if the array is full."""
        return self._available_size == 0

    def push(self, values: Union[T, List[T], np.ndarray, torch.Tensor]) -> None:
        """
        Add element(s) to the end of the ring array.
        If the array is full, the oldest elements are overwritten.

        Args:
            values: Single value or array of values to add
        """
        # Convert single value to array if needed
        if not isinstance(values, (list, np.ndarray, torch.Tensor)):
            values = [values]

        n = len(values)
        # traceback.print_stack(limit=5)
        # logger.info(f"@@@@@ pushing {n} elements")
        
        # logger.info(f"available_size={self._available_size}, capacity={self.capacity}, n={n}")

        # If adding would exceed capacity, adjust start_idx and size
        if self._available_size + n > self.capacity:
            # Calculate how many old elements will be overwritten
            overflow = self._available_size + n - self.capacity
            self.start_idx = (self.start_idx + overflow) % self.capacity
            self._available_size = min(self._available_size + n, self.capacity)
        else:
            self._available_size += n

        # Calculate where to place the new values
        end_idx = (self.start_idx + self._available_size - n) % self.capacity
        # logger.info(f"start_idx={self.start_idx}, end_idx={end_idx}, n={n}, "
        #             f"available_size={self._available_size}, capacity={self.capacity}")

        # Handle wrap-around case
        if end_idx + n > self.capacity:
            # Split the insertion into two parts
            first_part_size = self.capacity - end_idx
            self.array[end_idx:] = values[:first_part_size]
            self.array[: n - first_part_size] = values[first_part_size:]
        else:
            # No wrap-around needed
            # logger.info(f"pushing values shape={len(values)}, end_idx={end_idx}, n={n}, "
            #             f"selected shape={self.array[end_idx : end_idx + n].shape}"
            #             f"start_idx={self.start_idx}, available_size={self._available_size}")
            self.array[end_idx : end_idx + n] = values

        # logger.info(f"pushed values shape={len(values)}, end_idx={end_idx}, n={n}, "
        #             f"selected shape={self.array[end_idx : end_idx + n].shape}"
        #             f"start_idx={self.start_idx}, available_size={self._available_size}")

    def pop(self, n: int = 1):
        """
        Remove and return the first n elements from the ring array.

        Args:
            n: Number of elements to pop

        Returns:
            Numpy array containing the popped elements

        Raises:
            ValueError: If trying to pop more elements than currently in the array
        """

        # Handle wrap-around case
        if self.start_idx + n > self.capacity:
            result = self.container_constructor((n,), dtype=self.dtype)
            first_part_size = self.capacity - self.start_idx
            result[:first_part_size] = self.array[self.start_idx :]
            result[first_part_size:] = self.array[: n - first_part_size]
        else:
            result = self.array[self.start_idx : self.start_idx + n]

        # Update start_idx and size
        self.start_idx = (self.start_idx + n) % self.capacity
        self._available_size -= n

        return result

    def clear(self, initializer=None) -> None:
        """Reset the ring array to empty state."""
        if initializer is None:
            self.array[:] = np.array(range(self.capacity))
        else:
            self.array[:] = initializer
        self.start_idx = 0
        self._available_size = self.capacity

    def get_all(self) -> np.ndarray:
        """
        Return all elements currently in the ring array in correct order.

        Returns:
            Numpy array containing all elements
        """
        if self.is_full():
            return self.container_constructor(0, dtype=self.dtype)

        result = self.container_constructor((self._available_size,), dtype=self.dtype)

        # Handle wrap-around case
        if self.start_idx + self._available_size > self.capacity:
            first_part_size = self.capacity - self.start_idx
            result[:first_part_size] = self.array[self.start_idx :]
            result[first_part_size:] = self.array[: self._available_size - first_part_size]
        else:
            result[:] = self.array[self.start_idx : self.start_idx + self._available_size]

        return result

    def __del__(self):
        """Clean up shared memory resources."""
        del self.shared_array
        del self.metadata


class GPUSharedRingArray(SharedRingArray):
    """
    A GPU version of the SharedRingArray.
    This class is a placeholder and does not implement any GPU-specific functionality.
    """

    def __init__(
        self, capacity: int, dtype, device, *, name: str, create: bool, container_constructor=torch.zeros
    ):
        self.capacity = capacity
        self.dtype = dtype
        self.device = device
        self.is_owner = create
        self.container_constructor = container_constructor
        self.array = None
        if create:
            self.metadata = SharedNPArray.create(np.array([0, capacity], dtype=np.int32), name=name + "_metadata")
            self.start_idx = 0
            self._available_size = capacity
        else:
            self.metadata = SharedNPArray.rebuild(name + "_metadata", (2,), np.dtype(np.int32).str)

    def clear(self):
        self.array[:] = torch.arange(1, self.capacity + 1, dtype=self.dtype, device=self.device)
        self.start_idx = 0
        self._available_size = self.capacity

    def __del__(self):
        """Clean up shared memory resources."""
        del self.metadata


# Example usage
def main():
    ring = GPUSharedRingArray(10, dtype=torch.int64, device="cuda", name="gpu_ring", create=True)
    print(ring.array)

    # Create a shared ring array
    ring = SharedRingArray(10, dtype=np.int32, name="my_ring", create=True)

    idx = ring.pop(4)
    print(f"Initial pop: {idx}")

    ring.push([1, 2])
    print(f"After pushing [1,2]: {ring.get_all()}")

    r2 = SharedRingArray(10, dtype=np.int32, name="my_ring", create=False)
    print(f"Rebuilt ring: {r2.get_all()}")

    idx = ring.pop(8)
    print(f"After popping 8: {idx}, ring={ring.get_all()}")

    ring.push([6])
    print(f"After pushing [6]: {ring.get_all()}")

    r2.push([4, 2, 9])
    print(f"After pushing [4,6,9]: {ring.get_all()}")

    r2.clear()
    print(f"After clearing: {ring.get_all()}")


if __name__ == "__main__":
    main()
