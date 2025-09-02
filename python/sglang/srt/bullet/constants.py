
class ShmName:
    @staticmethod
    def req_pool(rank: int):
        return f"reqpool_{rank}"
    
    @staticmethod
    def kv_pool(rank: int):
        return f"kvpool_{rank}"


class DebugTy:
    PREFILL_BATCH_SIZE = "timestamp,prefill_ms,prefill_batch_size,[req_lens]"
    DEOCDE_BATCH_SIZE = "timestamp,decode_ms,decode_batch_size,[req_lens]"


class IPCGroup:
    DRIVER = "driver"  # The driver process to launch subprocesses
    MEMPOOL = "mempool"  # The KVCache memory pool
    REQPOOL = "reqpool"  # The ReqToToken pool
    WEIGHT = "weight"  # The model weight server
    GATEWAY = "gateway"  # The gateway server
    AUTOTEST = "autotest"  # The autotest coordinator
    PROFILER = "profiler"  # The latency profiler server
    NANOFLOWER = "nanoflower"  # The test bed for nanoflow

    @staticmethod
    def engine_id(host: str, port: int, tp_rank: int, pp_rank: int, is_prefill: bool):
        sufix = "prefill" if is_prefill else "decode"
        return f"engine_{host}:{port}@tp_{tp_rank}_pp_{pp_rank}_{sufix}"


class MsgTy:
    READY_TO_INIT_KV = 0
    GRANT_KV_INIT = 1
    KV_MEMPOOL_INITED = 2
    REQ_POOL_INITED = 3
    WEIGHT_INITED = 4
    CONTROLLER_INITED = 5
    ENGINE_INITED = 6
    DRIVER_INITED = 7
    STOP_DRIVER = 8
    SYNC_PROFILE_START = 9
    SYNC_PROFILE_END = 10
    RADIX_CACHE_INITED = 11
    SPAWN_KV = 12
