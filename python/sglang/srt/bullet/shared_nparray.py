import logging
import time
from multiprocessing import shared_memory
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SharedNPArray:

    def __init__(self):
        self.array: np.ndarray = None
        self.shm: shared_memory.SharedMemory = None
        self.is_owner = False

    @classmethod
    def create(cls, array: np.ndarray, name: str=None):
        self = cls()

        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=array.nbytes)
        except FileExistsError:
            # If the shared memory already exists, unlink it and create a new one
            logging.warning(f"Shared memory {name} already exists. Unlinking and creating a new one.")
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=array.nbytes)
        
        self.is_owner = True
        self.array = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)
        self.array[:] = array[:]

        return self

    @classmethod
    def rebuild(cls, shm_name: str, shape: Tuple[int], dtype_str: str):
        dtype = np.dtype(dtype_str)
        counter = 0
        while True:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                break
            except FileNotFoundError:
                counter += 1
                logger.warning(f"Shared memory {shm_name} not found, retrying {counter} time...")
                time.sleep(1)

        try:
            shared_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        except Exception as e:
            shm.close()
            raise e

        self = cls()
        self.array = shared_np
        self.shm = shm
        self.is_owner = False

        return self

    def share(self) -> Tuple[str, Tuple[int], str]:
        return self.shm.name, self.array.shape, self.array.dtype.str

    def __del__(self):
        try:
            self.shm.close()
            if self.is_owner:
                self.shm.unlink()
        except Exception as e:
            pass


def create_shared_nparray(input_list: list, dtype) -> Tuple[str, Tuple[int], str]:

    input_np = np.array(input_list, dtype=dtype)
    shape = input_np.shape
    dtype_str = input_np.dtype.str
    total_size = input_np.nbytes
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    try:
        shared_np = np.ndarray(shape, dtype=input_np.dtype, buffer=shm.buf)
        shared_np[:] = input_np[:]
    except Exception as e:
        shm.close()
        shm.unlink()
        raise e

    return shm.name, shape, dtype_str


def rebuild_shared_nparray(shm_name: str, shape: Tuple[int], dtype_str: str) -> np.ndarray:
    dtype = np.dtype(dtype_str)
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        shared_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    except Exception as e:
        shm.close()
        raise e

    return shared_np, shm  # the shm object should be referenced to prevent GC


def main():
    # Create a shared numpy array
    array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    shared_array = SharedNPArray.create(array)
    shm_name, shape, dtype_str = shared_array.share()

    print(f"Shared memory name: {shm_name}")
    print(f"Array shape: {shape}")
    print(f"Array dtype: {dtype_str}")

    # Rebuild the shared numpy array
    rebuilt_array = SharedNPArray.rebuild(shm_name, shape, dtype_str)
    print(f"Rebuilt array: {rebuilt_array.array}")

    # Clean up
    del shared_array
    del rebuilt_array


if __name__ == "__main__":
    main()
