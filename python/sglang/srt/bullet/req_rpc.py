import pickle
import time
import traceback
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor

import zmq
import torch
from sglang.srt.bullet.observability import ReqTimingState
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)


def req_to_dict_v2(req: Req, tstamp: float):
    # logger.info(f"Origin Req: {req}")
    ret = {}

    # Input and output info
    ret["rid"] = req.rid
    req.origin_input_text
    req.origin_input_ids_unpadded
    ret["origin_input_ids"] = req.origin_input_ids
    ret["output_ids"] = req.output_ids
    req.fill_ids
    ret["session_id"] = req.session_id
    req.input_embeds
    ret["stream"] = req.stream

    # Sampling info
    ret["sampling_params"] = req.sampling_params
    req.custom_logit_processor
    req.return_hidden_states

    # Memory pool info
    ret["req_pool_idx"] = req.req_pool_idx

    # Check finish, skipped some
    ret["eos_token_ids"] = req.eos_token_ids

    # Incremental decoding, skipped
    # Multimodal inputs, skipped

    # Prefix info
    req.prefix_indices
    req.extend_input_len
    req.last_node
    # req.last_node_global

    # Is chunked, skipped
    # Retraction, skipped
    # Logprobs, skipped
    # Latency breakdown, skipped
    # Logprobs, return values, skipped
    # Embedding, return values, skipped
    # Constrained decoding, skipped

    # Cached tokens
    ret["cached_tokens"] = req.cached_tokens
    ret["already_computed"] = req.already_computed
    req.is_retracted

    # Lora, skipped
    # Disaggregation, skipped

    # Bulelt info
    ret["timing_state"] = {
        "arrive_tstamp": req.timing_state.arrive_tstamp,
        "queue_time": req.timing_state.queue_time,
        "ttft": req.timing_state.ttft,
        "ttft_gpu": req.timing_state.ttft_gpu,
        "ttft_gpu_estimate": req.timing_state.ttft_gpu_estimate,
        "transmit_tstamp": tstamp
    }

    return ret


def dict_to_req_v2(dct: Dict, tstamp: float, tokenizer):
    req = Req(
        rid=dct["rid"],
        origin_input_text=None,
        origin_input_ids=dct["origin_input_ids"],
        sampling_params=dct["sampling_params"],
        stream=dct["stream"],
    )
    req.output_ids = dct["output_ids"]
    req.fill_ids = (req.origin_input_ids + req.output_ids)[:-1]
    req.req_pool_idx = dct["req_pool_idx"]

    # Check finish, skipped some
    req.tokenizer = tokenizer
    req.eos_token_ids = dct["eos_token_ids"]

    # Incremental decoding, skipped
    # Multimodal inputs, skipped
    # Prefix info, skipped
    # Is chunked, skipped
    # Retraction, skipped
    # Logprobs, skipped
    # Latency breakdown, skipped
    # Logprobs, return values, skipped
    # Embedding, return values, skipped
    # Constrained decoding, skipped

    # Cached tokens
    req.cached_tokens = dct["cached_tokens"]
    req.already_computed = dct["already_computed"]

    # Bullet timing info
    req.timing_state = ReqTimingState(
        input_len=len(req.origin_input_ids),
        arrive_tstamp=dct["timing_state"]["arrive_tstamp"],
        queue_time=dct["timing_state"]["queue_time"],
        output_len=1,  # Will be updated later
        ttft=dct["timing_state"]["ttft"],
        ttft_gpu=dct["timing_state"]["ttft_gpu"],
        ttft_gpu_estimate=dct["timing_state"]["ttft_gpu_estimate"],
        transmit_time=dct["timing_state"]["transmit_tstamp"] - tstamp,
    )

    # logger.info(f"Reconstruct Req: {req}")

    return req


def schedule_batch_to_dict_v2(batch: ScheduleBatch):
    # logger.info(f"schedule_batch_to_dict_v2")
    # traceback.print_stack(limit=5)
    ret = {}
    ret["reqs"] = [req_to_dict_v2(req, time.time()) for req in batch.reqs]
    # ret["overlap"] = batch.enable_overlap
    # ret["forever"] = batch.is_forever_bg_loop
    return ret


def dict_to_schedule_batch_v2(
    dct: Dict,
    *,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    tree_cache,
    model_config,
    tokenizer,
    is_forever_bg_loop,
    enable_overlap,
):
    # Request, memory pool, and cache
    reqs = [dict_to_req_v2(req, time.time(), tokenizer) for req in dct["reqs"]]
    batch = ScheduleBatch.init_new(
        reqs,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache,
        model_config,
        enable_overlap=enable_overlap,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
        is_forever_bg_loop=is_forever_bg_loop,
    )

    # Batch configs
    batch.forward_mode = ForwardMode.DECODE

    # Sampling info, adapted from ScheduleBatch.prepare_for_extend()
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch,
        batch.model_config.vocab_size,
    )

    # Init tensors
    reqs = batch.reqs
    req_pool_indices = [r.req_pool_idx for r in reqs]
    seq_lens = [len(r.fill_ids) for r in reqs]
    output_ids = [r.output_ids[-1] for r in reqs]

    req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
        batch.device, non_blocking=True
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(batch.device, non_blocking=True)
    output_ids_tensor = torch.tensor(output_ids, dtype=torch.int64).to(batch.device, non_blocking=True)

    batch.seq_lens_sum = sum(seq_lens)
    batch.req_pool_indices = req_pool_indices_tensor
    batch.seq_lens = seq_lens_tensor
    batch.output_ids = output_ids_tensor

    return batch


def req_to_dict(req: Req, timestamp: float, *, require_debug_info=False, dump_prefix_indices=False):
    ret = {}
    ret["rid"] = req.rid
    ret["origin_input_ids"] = req.origin_input_ids
    ret["output_ids"] = req.output_ids
    ret["req_pool_idx"] = req.req_pool_idx
    ret["sampling_params"] = req.sampling_params

    ret["last_node_id"] = 0  # req.last_node.uid if req.last_node_id is None else req.last_node_id
    ret["extend_input_len"] = req.extend_input_len
    # ret["arrive_tstamp"] = req.debug_info.arrive_tstamp
    # ret["inengine_tstamp"] = req.debug_info.inengine_tstamp
    # ret["ttft"] = timestamp - req.debug_info.inengine_tstamp
    # ret["transmit_tstamp"] = timestamp

    if dump_prefix_indices:
        ret["prefix_indices"] = req.prefix_indices.tolist()
    else:
        ret["prefix_len"] = len(req.prefix_indices)

    if require_debug_info:
        ret["debug_info"] = req.debug_info
    return ret


def dict_to_req(dct: Dict, tstamp: float, *, lookup_tree_node_id=False, dump_prefix_indices=False):
    req = Req(
        rid=dct["rid"],
        origin_input_text=None,
        origin_input_ids=dct["origin_input_ids"],
        sampling_params=dct["sampling_params"],
        stream=True,
    )
    req.output_ids = dct["output_ids"]
    req.fill_ids = (req.origin_input_ids + req.output_ids)[:-1]
    req.req_pool_idx = dct["req_pool_idx"]

    req.last_node_id = dct["last_node_id"]
    req.extend_input_len = dct["extend_input_len"]
    # req.debug_info = dct.get("debug_info", ReqDebugInfo(dct["arrive_tstamp"], 0, dct["inengine_tstamp"]))
    # req.debug_info.ttft = dct["ttft"]
    # req.timing_state = ReqTimingState(
    #     input_len=len(req.origin_input_ids),
    #     arrive_tstamp=dct["arrive_tstamp"],
    #     output_len=1,
    #     ttft=dct["ttft"],
    #     transmit_tstamp=dct["transmit_tstamp"],
    #     recv_tstamp=tstamp
    # )

    if lookup_tree_node_id:
        req.last_node = TreeNode.id_to_node[req.last_node_id]

    if dump_prefix_indices:
        req.prefix_indices = torch.tensor(dct["prefix_indices"], dtype=torch.int32, pin_memory=True)
    else:
        req.prefix_indices = [0] * dct["prefix_len"]

    return req


def dump_batch(batch: ScheduleBatch):
    ts = time.time()
    reqs = [req_to_dict(req, ts) for req in batch.reqs]
    return pickle.dumps(reqs)


def load_batch(buf, req_to_token_pool, token_to_kv_pool_allocator, tree_cache, model_config):
    reqs_dict = pickle.loads(buf)
    tstamp = time.time()
    reqs = [dict_to_req(dct, tstamp) for dct in reqs_dict]
    batch = ScheduleBatch.init_new(
        reqs,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache,
        model_config,
        enable_overlap=True,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )

    # See ScheduleBatch.merge() to see why these fields are needed
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, model_config.vocab_size)
    batch.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
    req_pool_indices_cpu = [req.req_pool_idx for req in reqs]
    seq_lens = [len(req.fill_ids) for req in reqs]
    with torch.device("cuda"):
        batch.req_pool_indices = torch.tensor(req_pool_indices_cpu)
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)

    return batch


class ReqRPC:
    def __init__(self, is_prefill, host="127.0.0.1", port=31006, ipc_name=None) -> None:
        context = zmq.Context(io_threads=2)
        self.socket = context.socket(zmq.PAIR)
        if ipc_name is not None:
            assert ipc_name.startswith("ipc://")
            addr = ipc_name
        else:
            addr = f"tcp://{host}:{port}"
        if is_prefill:
            self.socket.bind(addr)
        else:
            self.socket.connect(addr)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def send(self, obj):
        buf = pickle.dumps(obj)
        self.socket.send(buf)

    def recv(self, block=True):
        if block:
            buf = self.socket.recv()
            obj = pickle.loads(buf)
            return obj
        try:
            buf = self.socket.recv(flags=zmq.NOBLOCK)
            obj = pickle.loads(buf)
            return obj
        except zmq.Again:
            return None

    def send_batch_thread_v2(self, batch):
        dumped = schedule_batch_to_dict_v2(batch)
        self.socket.send_pyobj(dumped)

    def send_batch_v2(self, batch):
        # self.send_batch_thread_v2(batch)
        self.executor.submit(self.send_batch_thread_v2, batch)

    def recv_batch_v2(
        self, req_to_token_pool, token_to_kv_pool_allocator, tree_cache, model_config, do_serialize=False
    ):
        try:
            batch = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            if do_serialize:
                batch = dict_to_schedule_batch_v2(
                    batch, req_to_token_pool, token_to_kv_pool_allocator, tree_cache, model_config
                )
                # logger.info(f"RPC recv {len(batch.reqs)} reqs, {batch}")
            else:
                # logger.info(f"RPC recv {len(batch['reqs'])} reqs")
                pass
            return batch
        except zmq.Again:
            # logger.info(f"RPC recv None")
            return None

    def send_batch_thread(self, batch):
        dumped = dump_batch(batch)
        self.socket.send_pyobj(dumped)

    def send_batch(self, batch):
        self.executor.submit(self.send_batch_thread, batch)

    def recv_batch(self, vocab_size, req_to_token_pool, token_to_kv_pool, tree_cache, tpserver_debug_info):
        try:
            obj = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            batch = load_batch(
                obj, vocab_size, req_to_token_pool, token_to_kv_pool, tree_cache, tpserver_debug_info
            )
            # logger.info(f"RPC recv {len(batch.reqs)} reqs, {batch}")
            return batch
        except zmq.Again:
            return None
        except Exception as e:
            raise e

    def send_reqs_thread(self, reqs: List[Req]):
        def light_dump(req: Req):
            return {
                "rid": req.rid,
                "output_ids": req.output_ids,
                "req_pool_idx": req.req_pool_idx,
                "prefix_len": len(req.prefix_indices),
                "last_node_id": 0,  # req.last_node.uid,
            }

        req_dicts = [light_dump(req) for req in reqs]
        self.socket.send_pyobj(req_dicts)

    def send_to_cache(self, reqs: List[Req]):
        return
        self.executor.submit(self.send_reqs_thread, reqs)

    def recv_to_cache(self, *, loockup_tree_node_id):
        return []

        def light_load(dct):
            req = Req(dct["rid"], None, None, None)
            req.output_ids = dct["output_ids"]
            # req.req_pool_idx = dct["req_pool_idx"]
            # req.prefix_len = dct["prefix_len"]
            # req.last_node = TreeNode.id_to_node[req.last_node_id]
            return req

        try:
            obj = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            reqs = [light_load(dct) for dct in obj]
            return reqs
        except zmq.Again:
            return []
        except Exception as e:
            raise e
