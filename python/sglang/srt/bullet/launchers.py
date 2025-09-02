import argparse
import dataclasses
from dataclasses import dataclass
import enum
import os
import logging
import shutil
import subprocess
import sys
import time
from typing import List

import torch
from sglang.srt.bullet.radix_cache_rpc import RadixCacheForZMQServer
from sglang.srt.bullet_utils import BASE
from sglang.srt.server_args import ServerArgs
from sglang.srt.bullet.ipc_chat import MsgRouter


logger = logging.getLogger(__name__)


def launch_radix_cache_server(args: ServerArgs):
    host = args.rpc_radix_host
    port = args.rpc_radix_port
    server = RadixCacheForZMQServer.launch(host, port)
    return server


def launch_msg_router(ipc_name: str):
    router = MsgRouter(ipc_name=ipc_name)
    router.launch_process(prefix=" MsgRouter")
    return router
