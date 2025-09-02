import fcntl
import mmap
import multiprocessing
import os
import struct
import tempfile
import logging
import time


logger = logging.getLogger(__name__)


class SharedMemoryMutex:
    def __init__(self, file_path: str, size: int = 4):
        """
        Initialize the `reentrantable` shared memory and file locking mechanism.

        Args:
            file_path (str): Path to the shared memory file.
            size (int): Size of the shared memory (default is 4 bytes for an integer).
        """
        self.file_path = file_path
        self.size = size
        self._locked_count = 0
        self.file = open(self.file_path, "r+b" if os.path.exists(self.file_path) else "w+b")
        if os.path.getsize(self.file_path) == 0:
            self.file.write(b"\x00" * self.size)
            self.file.flush()

        self.shared_memory = mmap.mmap(self.file.fileno(), self.size)
        logger.info(f"Shared memory file {self.file_path} mapped to {self.shared_memory}")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False

    def acquire(self):
        if self._locked_count == 0:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
            self._locked_count = 1

    def release(self):
        self._locked_count -= 1
        if self._locked_count == 0:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)

    def read(self) -> int:
        """Read the integer value from shared memory."""
        self.shared_memory.seek(0)
        data = self.shared_memory.read(self.size)
        return struct.unpack("i", data)[0]

    def write(self, value: int):
        """Write an integer value to shared memory."""
        self.shared_memory.seek(0)
        self.shared_memory.write(struct.pack("i", value))
        self.shared_memory.flush()

    def locked_read(self):
        with self:
            return self.read()

    def locked_write(self, value):
        with self:
            self.write(value)

    def close(self):
        """Close the shared memory and file."""
        self.shared_memory.close()
        self.file.close()

    def __del__(self):
        """Ensure that resources are cleaned up when the object is deleted."""
        self.close()


class SharedAtomicInt(SharedMemoryMutex):
    def __init__(self, file_path: str):
        super().__init__(file_path, 4)

    def register_as_property(self, obj, property_name):
        """Register the atomic integer as a property of an object."""
        raise NotImplementedError("This method has bug, it cannot set property correctly.")
        setattr(obj, property_name, property(self.locked_read, self.locked_write))

    def _update(self, value, operation):
        with self:
            counter = self.read()
            counter = operation(counter, value)
            self.write(counter)
            return counter

    def __add__(self, value):
        return self.locked_read() + value

    def __iadd__(self, value):
        return self._update(value, lambda x, y: x + y)

    def __isub__(self, value):
        return self._update(value, lambda x, y: x - y)


if __name__ == "__main__":

    SHARED_FILE_PATH = "/tmp/shared_memory_file"
    NUM_WORKERS = 10
    NUM_ITERATIONS = 10000

    def worker_task(worker_id):
        """Worker task to read and write shared memory NUM_ITERATIONS times."""
        shared_memory_mutex = SharedMemoryMutex(SHARED_FILE_PATH)

        for _ in range(NUM_ITERATIONS):
            with shared_memory_mutex:  # Acquire mutex lock
                # Critical section: read, modify, and write to shared memory
                counter = shared_memory_mutex.read()
                counter += 1
                shared_memory_mutex.write(counter)

        shared_memory_mutex.close()
        print(f"Worker {worker_id} finished")

    def benchmark():
        """Launch multiple workers and measure the total time taken."""
        # Initialize the shared memory to zero before starting
        shared_memory_mutex = SharedMemoryMutex(SHARED_FILE_PATH)
        shared_memory_mutex.write(0)
        shared_memory_mutex.close()

        # Start timing the execution
        start_time = time.time()

        # Launch worker processes
        processes = []
        for worker_id in range(NUM_WORKERS):
            process = multiprocessing.Process(target=worker_task, args=(worker_id,))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Stop timing the execution
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Total elapsed time: {elapsed_time:.4f} s")
        print(f"Each r/w time: {elapsed_time / NUM_ITERATIONS / NUM_WORKERS * 1000:.4f} ms")

        # Check the final counter value
        shared_memory_mutex = SharedMemoryMutex(SHARED_FILE_PATH)
        final_value = shared_memory_mutex.read()
        print(f"Final counter value: {final_value}")
        shared_memory_mutex.close()

    benchmark()
