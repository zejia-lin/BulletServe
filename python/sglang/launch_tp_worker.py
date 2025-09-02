"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        _launch_subprocesses(server_args=server_args, port_args=server_args.port_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

