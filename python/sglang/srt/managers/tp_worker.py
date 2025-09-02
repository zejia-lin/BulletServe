# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

import logging
import threading
import time
from typing import Optional, Tuple, Union

import torch

from sglang.srt.bullet_utils import TOTAL_TPCS
from sglang.srt.bullet.observability import (
    PredictorInfoEntry,
    PredictorInfos,
    ReqTimingState,
    ReqTimingStateDict,
)
from sglang.srt.bullet.shared_mng import SelectedSMPolicy, SharedManager
from sglang.srt.bullet.sm_controller import SMController
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed

logger = logging.getLogger(__name__)


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path if not is_draft_worker else server_args.speculative_draft_model_path
            ),
            is_draft_model=is_draft_worker,
        )

        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert self.max_req_len > 0 and self.max_req_input_len > 0, "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        self.hicache_layer_transfer_counter = None

        # Bullet related
        self.server_args = server_args
        if True: #server_args.enable_bullet_engine:
            self.smctrl = SMController()
            self.predictor_infos = PredictorInfos()
            self.timing_states = ReqTimingStateDict(enabled=server_args.enable_record_timing)
            self.is_bullet_prefill = server_args.is_bullet_prefill
            self.is_bullet_decode = server_args.is_bullet_decode
            self.forward_stream = torch.cuda.Stream(priority=-1 if server_args.is_bullet_decode else 0)
            self.pp_first_rank = self.pp_group.is_first_rank
            self.pp_last_rank = self.pp_group.is_last_rank
            self.last_num_tpc = 0
            if server_args.enable_bullet_engine and server_args.disable_overlap_schedule:
                self.forward_batch_generation = self.forward_batch_generation_bullet
            if server_args.is_bullet_prefill and self.tp_rank == 0:
                self.shared_mng = SharedManager(server_args, create=True)
                # self.forward_batch_generation = self.forward_batch_generation_prefill
            else:
                self.shared_mng = SharedManager(server_args, create=False)

    def register_hicache_layer_transfer_counter(self, counter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return getattr(self.model_runner.attention_tp_group, "cpu_group", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def update_bullet_before_forward(self, model_worker_batch: ModelWorkerBatch, create_entry: bool = True):
        """
        1. Update shared manager states
        2. Set adaptive TPCs
        3. Record predictor info
        """

        if not self.server_args.enable_bullet_engine:
            return None

        # Record states
        num_forward_tokens = model_worker_batch.input_ids.size(0)
        self.shared_mng.rem_layers = self.model_runner.model.model.end_layer - self.model_runner.model.model.start_layer
        if self.tp_rank == 0 and create_entry:
            if model_worker_batch.forward_mode == ForwardMode.EXTEND:
                self.shared_mng.prefill_size = num_forward_tokens
            elif model_worker_batch.forward_mode == ForwardMode.DECODE:
                self.shared_mng.decode_size = num_forward_tokens
                self.shared_mng.decode_total_context = model_worker_batch.total_tokens
            elif (
                model_worker_batch.forward_mode == ForwardMode.IDLE
                or model_worker_batch.forward_mode == ForwardMode.DUMMY_FIRST()
            ):
                pass
            else:
                raise ValueError(f"Forward mode {model_worker_batch.forward_mode} is not supported.")

        # Set adaptive TPCs
        if model_worker_batch.forward_mode.is_extend():
            num_tpcs, policy = self.shared_mng.set_adaptive_prefill_num_tpcs(
                model_worker_batch.longest_queue_ms
            )
            # logger.info(f"Prefill TPC {num_tpcs}, policy {policy}")
        elif model_worker_batch.forward_mode.is_decode():
            num_tpcs, policy = self.shared_mng.set_adaptive_decode_num_tpcs(
                model_worker_batch.longest_queue_ms
            )
        else:
            num_tpcs, policy = TOTAL_TPCS, SelectedSMPolicy.DEFAULT

        if num_tpcs != self.last_num_tpc:
            self.smctrl.set_stream_mask(
                self.forward_stream,
                0,
                num_tpcs,
                reversed=self.is_bullet_decode and not self.server_args.disable_decode_tpc_reverse,
            )
            self.last_num_tpc = num_tpcs

        # Record predictor info
        if self.worker.tp_rank == 0 and create_entry:
            phase = "prefill" if model_worker_batch.forward_mode == ForwardMode.EXTEND else "decode"
            entry = PredictorInfoEntry(
                phase=phase,
                prefill_len=self.worker.shared_mng.prefill_size,
                decode_bs=self.worker.shared_mng.decode_size,
                decode_tokens=self.worker.shared_mng.decode_total_context,
                prefill_tpc=self.worker.shared_mng.prefill_num_tpcs,
                decode_tpc=self.worker.shared_mng.decode_num_tpcs,
                predict_ms=self.worker.shared_mng.predict_duration(phase),
                start_timstamp=time.time(),
                policy=policy,
                rids=model_worker_batch.rids,
            )
        else:
            entry = None

        return entry

    def update_bullet_after_forward(self, entry: PredictorInfoEntry):

        if entry is None:
            return

        if not self.server_args.enable_bullet_engine or self.tp_rank != 0:
            return

        gpu_ms = self.worker.timing_states.step(
            ForwardMode.EXTEND if self.worker.is_bullet_prefill else ForwardMode.DECODE,
            enable_gpu_timing=self.server_args.enable_gpu_timing,
        )
        _t = time.time()
        _len = entry.prefill_len if self.worker.is_bullet_prefill else 1
        entry.prefill_len_after = self.worker.shared_mng.prefill_size
        self.worker.predictor_infos.add_entry(entry, _t, gpu_ms)
        self.worker.shared_mng.set_real_norm_ms(entry.phase, (_t - entry.start_timstamp) * 1000 / _len)

        if self.worker.is_bullet_prefill:
            self.worker.shared_mng.prefill_size = 0
            self.worker.shared_mng.prefill_num_tpcs = TOTAL_TPCS
        elif self.worker.is_bullet_decode:
            self.worker.shared_mng.decode_size = 0
            self.worker.shared_mng.decode_total_context = 0
            self.worker.shared_mng.decode_num_tpcs = TOTAL_TPCS
        else:
            raise ValueError("Invalid mode, should never reach here.")

    def model_forward_with_bullet(
        self,
        model_worker_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        self.timing_states.begin_forward(
            self.forward_stream, model_worker_batch.rids, enable_gpu_timing=self.server_args.enable_gpu_timing
        )
        with torch.cuda.stream(self.forward_stream):
            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )
        self.timing_states.end_forward(
            self.forward_stream, enable_gpu_timing=self.server_args.enable_gpu_timing
        )
        return logits_output, can_run_cuda_graph

    def forward_batch_generation_bullet(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool]:

        pp_proxy_tensors = None
        entry = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(all_gather_group=self.get_attention_tp_group())
            )
        else:
            entry = self.update_bullet_before_forward(model_worker_batch)

        # torch.cuda.synchronize()

        if self.pp_group.is_last_rank:
            logits_output, can_run_cuda_graph = self.model_forward_with_bullet(
                model_worker_batch, pp_proxy_tensors=pp_proxy_tensors
            )
            if launch_done is not None:
                launch_done.set()

            if skip_sample:
                next_token_ids = None
            else:
                with torch.cuda.stream(self.forward_stream):
                    next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)

        else:
            pp_proxy_tensors, can_run_cuda_graph = self.model_forward_with_bullet(
                model_worker_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            logits_output, next_token_ids = pp_proxy_tensors.tensors, None

        # torch.cuda.synchronize()

        self.update_bullet_after_forward(entry)

        return logits_output, next_token_ids, can_run_cuda_graph

    def layerwise_prefill_step_generator(self):
        for i in range(self.model_runner.num_prefill_steps):
            self.pp_group.is_first_rank = True if i == 0 else False
            self.pp_group.is_last_rank = True if i == self.model_runner.num_prefill_steps - 1 else False
            self.model_runner.model.cur_step = i
            yield i

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool]:
        if forward_batch is None:
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        if not self.pp_group.is_first_rank and pp_proxy_tensors is None:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(all_gather_group=self.get_attention_tp_group())
            )

        if self.pp_group.is_last_rank:
            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )
            if launch_done is not None:
                launch_done.set()

            if skip_sample:
                next_token_ids = None
            else:
                next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
            return logits_output, next_token_ids, can_run_cuda_graph
        else:
            pp_proxy_tensors, can_run_cuda_graph = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return pp_proxy_tensors.tensors, None, can_run_cuda_graph

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output, _ = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def update_weights_from_distributed(self, recv_req: UpdateWeightsFromDistributedReqInput):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names, recv_req.dtypes, recv_req.shapes, recv_req.group_name
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(recv_req.name, recv_req.truncate_size)
        return parameter

    def dump_predictor_infos(self, path: str, clear: bool):
        try:
            if self.server_args.enable_bullet_engine:
                if self.predictor_infos.dump(path):
                    logger.info(f"Predictor infos dumped to {path}")
                    if clear:
                        self.predictor_infos.clear()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to dump predictor infos: {e}")
            return False

    def dump_timing_states(self, path: str, clear: bool):
        try:
            if self.server_args.enable_bullet_engine:
                if self.timing_states.dump(path):
                    logger.info(f"Timing states dumped to {path}")
                    if clear:
                        self.timing_states.clear()
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to dump timing states: {e}")
            return False

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.lora_name, recv_req.lora_path)
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.lora_name)
        return result
