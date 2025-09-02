from functools import partial
import logging
import os
import time
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.bullet.ring_array import GPUSharedRingArray, SharedRingArray
from sglang.srt.bullet.rpc_server import ZMQClient, build_tensor_from_dict, share_tensor_to_dict
from sglang.srt.bullet.smem_mutex import SharedAtomicInt
from sglang.srt.bullet.ipc_chat import MsgClient
from sglang.srt.bullet.constants import IPCGroup, MsgTy, ShmName
from sglang.srt.mem_cache.memory_pool import GB, MHATokenToKVPool, ReqToTokenPool


logger = logging.getLogger(__name__)


class ReqToTokenPoolRemoteClient:
    def __init__(self, req_to_token_dct: Union[Dict, torch.Tensor], create: bool):
        gpu_id = torch.cuda.current_device()
        if isinstance(req_to_token_dct, torch.Tensor):
            self.req_to_token = req_to_token_dct
        else:
            self.req_to_token = build_tensor_from_dict(req_to_token_dct)
        shape = self.req_to_token.shape
        self.size = shape[0]
        self.max_context_len = shape[1]
        self.device = self.req_to_token.device
        self.free_slots = SharedRingArray(self.size, np.int64, name=ShmName.req_pool(gpu_id), create=create)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots.pop(need_size)
        return select_index

    def free(self, free_index: Union[int, List[int]]):
        self.free_slots.push(free_index)

    def clear(self):
        self.free_slots.clear()


class FusedReqToTokenPoolAndMHATokenToKVPoolRemoteServer:

    def __init__(
        self,
        *,
        reqpool_size: int,
        max_context_len: int,
        kvcache_size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        gpu_list: List[int],
        launcher_pid: str,
        start_layer: int,
        end_layer: int
    ):
        # Token to kv pool
        self.size = kvcache_size
        self.max_context_len = max_context_len
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.atomic_lock_name = f"/tmp/bullet_mempool.lock_{launcher_pid}"

        if gpu_list is None:
            gpu_list = [torch.cuda.current_device()]
        self.k_buffers = [None] * torch.cuda.device_count()
        self.v_buffers = [None] * torch.cuda.device_count()
        self.mem_states = [None] * torch.cuda.device_count()

        logger.info(f"Initializing remote memory pool server on GPUs: {gpu_list}")
        logger.info(
            f"KV Cache size: {kvcache_size}, page size: {page_size}, max context len: {max_context_len}, "
            f"head num: {head_num}, head dim: {head_dim}, layer num: {layer_num}"
        )
        for gpu_id in gpu_list:
            with torch.cuda.device(gpu_id):
                self.mem_states[gpu_id] = torch.arange(1, kvcache_size + 1, dtype=torch.int64, device="cuda")
                self.k_buffers[gpu_id] = [
                    torch.empty((kvcache_size + page_size, head_num, head_dim), dtype=dtype, device="cuda")
                    for _ in range(layer_num)
                ]
                self.v_buffers[gpu_id] = [
                    torch.empty((kvcache_size + page_size, head_num, head_dim), dtype=dtype, device="cuda")
                    for _ in range(layer_num)
                ]

        self.atomic = SharedAtomicInt(self.atomic_lock_name)
        self.atomic.locked_write(kvcache_size)

        # req to token pool
        self.pools = [None] * torch.cuda.device_count()
        if gpu_list is None:
            gpu_list = [torch.cuda.current_device()]
        for gpu_id in gpu_list:
            with torch.cuda.device(gpu_id):
                self.pools[gpu_id] = torch.empty(
                    (reqpool_size, max_context_len), dtype=torch.int32, device="cuda"
                )

        logger.info("Remote memory pool server initialized.")

    def share_all_tensors(self, gpu_id: int):
        mem_state = [share_tensor_to_dict(self.mem_states[gpu_id])]
        k_buffer = [share_tensor_to_dict(i) for i in self.k_buffers[gpu_id]]
        v_buffer = [share_tensor_to_dict(i) for i in self.v_buffers[gpu_id]]
        return mem_state, k_buffer, v_buffer

    def get_all_tensors(self, gpu_id: int):
        mem_state = self.mem_states[gpu_id]
        k_buffer = self.k_buffers[gpu_id]
        v_buffer = self.v_buffers[gpu_id]
        return mem_state, k_buffer, v_buffer

    def get_config(self):
        return {
            "size": self.size,
            "page_size": self.page_size,
            "dtype": self.dtype,
            "head_num": self.head_num,
            "head_dim": self.head_dim,
            "layer_num": self.layer_num,
            "device": "cuda",  # TODO(lzj): backend device is hard coded to cuda
        }

    def share_req_to_token(self, gpu_id: int):
        return share_tensor_to_dict(self.pools[gpu_id])

    def get_req_to_token(self, gpu_id: int):
        return self.pools[gpu_id]


class MHATokenToKVPoolRemoteClient(MHATokenToKVPool):

    def __init__(self, config: dict, kvtensors, launcher_pid: str, start_layer: int):
        self.atomic = SharedAtomicInt(f"/tmp/bullet_mempool.lock_{launcher_pid}")
        self.start_layer = start_layer
        if self.atomic.locked_read() == 0:
            raise ValueError("The shared memory is not initialized!")
        self._construct_from_shared_tensors(config, kvtensors)
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def _warmup(self):
        with self.atomic:
            self.mem_state[:] = self.mem_state[:]
            for layer_id in range(self.layer_num):
                key, value = self.get_kv_buffer(layer_id)
                key[:] = 0
                value[:] = 0

    def _construct_from_shared_tensors(self, config, kvtensors):
        """Override the _create_buffers method to create shared buffers."""
        with self.atomic:
            logger.info(f"Get KV Cache config: {config}")
            self.size = config["size"]
            self.page_size = config["page_size"]
            self.dtype = config["dtype"]
            self.head_num = config["head_num"]
            self.head_dim = config["head_dim"]
            self.layer_num = config["layer_num"]
            self.device = config["device"]
            self.enable_memory_saver = False

            if self.dtype == torch.float8_e5m2:
                # NOTE: Store as torch.uint8 because Tensor index_put is not implemented for torch.float8_e5m2
                self.store_dtype = torch.uint8
            else:
                self.store_dtype = self.dtype

            # Construct shared tensors
            self.mem_state, self.k_buffer, self.v_buffer = self._reconstruct_shared_tensors(kvtensors)

            # Data fro MHATokenToKVPool
            self.layer_transfer_counter = None
            self.capture_mode = False
            self.device_module = torch.get_device_module(self.device)
            self.alt_stream = self.device_module.Stream()

            k_size, v_size = self.get_kv_size_bytes()
            logger.info(
                f"KV Cache received. #tokens: {self.size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
            )
            self._warmup()

    def _reconstruct_shared_tensors(self, kvtensors):
        if isinstance(kvtensors[0], list):
            mem_state, k_buffer, v_buffer = kvtensors
            mem_state = build_tensor_from_dict(mem_state[0])
            k_buffer = [build_tensor_from_dict(i) for i in k_buffer]
            v_buffer = [build_tensor_from_dict(i) for i in v_buffer]
            return mem_state, k_buffer, v_buffer
        else:
            return kvtensors


class TokenToKVPoolAllocatorRemoteClient(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: MHATokenToKVPoolRemoteClient,
        *,
        create: bool,
        launcher_pid: str
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = 1
        self.gpu_id = torch.cuda.current_device()
        self._kvcache = kvcache
        self.atomic = SharedAtomicInt(f"/tmp/bullet_mempool.lock_{launcher_pid}")

        self.free_slots = GPUSharedRingArray(
            self.size,
            torch.int64,
            device,
            name=ShmName.kv_pool(self.gpu_id),
            create=create,
            container_constructor=partial(torch.empty, device=device),
        )
        self.free_slots.array = kvcache.mem_state

        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self):
        return len(self.free_slots)

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        with self.atomic:
            if need_size > len(self.free_slots):
                return None

            select_index = self.free_slots.pop(need_size)
            return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            with self.atomic:
                self.free_slots.push(free_index)
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def backup_state(self):
        raise NotImplementedError("This method is not implemented.")

    def restore_state(self, free_slots):
        raise NotImplementedError("This method is not implemented.")

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots.clear()
        self.is_not_in_free_group = True
        self.free_group = []
