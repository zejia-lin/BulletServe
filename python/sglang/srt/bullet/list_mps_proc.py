#!/usr/bin/env python3
"""
列出连接到自定义 MPS 目录的全部进程
"""
import os
import subprocess
from typing import List, Tuple
import argparse


def read_proc(pid: int) -> Tuple[str, str]:
    """返回 (comm, cmdline)"""
    try:
        with open(f'/proc/{pid}/comm') as f:
            comm = f.read().strip()
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmd = f.read().replace(b'\0', b' ').decode(errors='replace').strip()
        
    except (FileNotFoundError, PermissionError):
        comm, cmd = '<unknown>', '<unknown>'
    return comm, cmd


def list_mps_clients(env=None) -> List[Tuple[int, str, str]]:
    # 2. 询问 MPS server
    try:
        servers = subprocess.check_output(
            ['nvidia-cuda-mps-control'], input='get_server_list',
            text=True, stderr=subprocess.STDOUT, env=os.environ if env is None else env
        ).strip().split()
    except subprocess.CalledProcessError:
        print('No MPS servers found.')
        return []
    if len(servers) == 0:
        print('No MPS servers found.')
        return []

    print("Found MPS servers: ", [int(s) for s in servers])
    clients: List[Tuple[int, str, str]] = []
    for s in servers:
        out = subprocess.check_output(
            ['nvidia-cuda-mps-control'], input=f'get_client_list {s}\n',
            text=True, stderr=subprocess.STDOUT, env=os.environ if env is None else env
        ).strip().split()
        for pid_str in out:
            pid = int(pid_str)
            comm, cmd = read_proc(pid)
            clients.append((pid, comm, cmd))
    return clients


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe-dir", type=str, required=False)
    parser.add_argument("--log-dir", type=str, required=False)

    args = parser.parse_args()

    if args.pipe_dir is not None:
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = args.pipe_dir
    if args.log_dir is not None:
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = args.log_dir

    clients = list_mps_clients()
    if not clients:
        print('No process connected to MPS.')
    else:
        for pid, comm, cmd in clients:
            print(f'{pid} | {comm} | {cmd}')
