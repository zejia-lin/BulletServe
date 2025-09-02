import asyncio
import logging
import os
import pickle
import signal
from dataclasses import dataclass
import collections.abc
import traceback
from typing import Any, Dict, List, Union

import torch.multiprocessing as torch_mp
import zmq
import zmq.asyncio

from sglang.srt.bullet.rpc_server import ServerException
from sglang.srt.bullet.loop_forever import LoopForeverAbstract

logger = logging.getLogger(__name__)


class MsgRouter(LoopForeverAbstract):

    REGISTER_REQ = "register"
    UNREGISTER_REQ = "unregister"

    def __init__(self, host: str = None, port: int = None, ipc_name: str = None):
        self.host = host
        self.port = port
        self.ipc_name = ipc_name
        if ipc_name is not None:
            assert ipc_name.startswith("ipc://")
            self.addr = ipc_name
        else:
            self.addr = f"tcp://{host}:{port}"
        self.clients: Dict[str, bytes] = {}

    def register_client(self, identity, name):
        self.clients[name] = identity
        logger.info(f"Register client '{name}' with identity {identity}")

    def unregister_client(self, name):
        if name in self.clients:
            logger.info(f"Unregister client '{name}' with identity {self.clients[name]}")
            self.clients.pop(name)
        else:
            logger.error(f"Client '{name}' not found in clients")

    async def _internal_loop(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.addr)
        logger.info(f"[pid={os.getpid()}] MsgRouter started at {self.addr}")
        while True:
            try:
                identity, buf = await self.socket.recv_multipart()
                target_names, datas = pickle.loads(buf)
                if datas[0] == self.REGISTER_REQ:
                    self.register_client(identity, datas[1])
                    self.socket.send_multipart([identity, pickle.dumps(True)])
                elif datas[0] == self.UNREGISTER_REQ:
                    self.unregister_client(datas[1])
                    self.socket.send_multipart([identity, pickle.dumps(True)])
                else:
                    for target_name in target_names:
                        target_id = self.clients[target_name]
                        buf = pickle.dumps(datas)
                        await self.socket.send_multipart([target_id, buf])
            except Exception as e:
                logger.critical(f"Catch exception: {e}, send to client '{identity.hex()}'", exc_info=True)
                tb = traceback.format_exc()
                await self.socket.send_multipart([identity, pickle.dumps(ServerException(e, tb))])


class MsgClient:
    def __init__(self, host: str, port: int, name: str, register_on_create=True, ipc_name:str = None):
        self.host = host
        self.port = port
        self.name = name
        self.ipc_name = ipc_name
        if ipc_name is not None:
            assert ipc_name.startswith("ipc://")
            self.addr = ipc_name
        else:
            self.addr = f"tcp://{host}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, f"{name}@{os.getpid()}".encode("utf-8"))
        self.connect_on_create = register_on_create
        if register_on_create:
            self.register()

    def register(self):
        self.socket.connect(self.addr)
        self.socket.send_multipart([pickle.dumps(([], [MsgRouter.REGISTER_REQ, self.name]))])
        if self.socket.recv():
            return
        raise RuntimeError("Failed to register")

    def unregister(self):
        self.socket.send(pickle.dumps(([], [MsgRouter.UNREGISTER_REQ, self.name])))
        if self.socket.recv():
            return
        raise RuntimeError("Failed to unregister")

    def send(self, target_names: Union[str, list], *args):
        if isinstance(target_names, str):
            target_names = [target_names]
        self.socket.send_multipart([pickle.dumps((target_names, args))])

    def recv(self, expected=None) -> List[Any]:
        msg = pickle.loads(self.socket.recv())
        if expected is not None:
            if isinstance(msg, ServerException):
                logger.critical(f"Server exception: {msg}")
            if not isinstance(msg, collections.abc.Iterable):
                msg = [msg]
            if isinstance(msg[0], list):
                assert msg[0][0] == expected, f"Unexpected message: {msg}, expect {expected}"
                return msg[0][1:]
            else:
                assert msg[0] == expected, f"Unexpected message: {msg}, expect {expected}"
        return msg

    def recv_many(self, expects: List):
        msg = pickle.loads(self.socket.recv())
        if isinstance(msg, ServerException):
            logger.critical(f"Server exception: {msg}")
        if isinstance(msg[0], list):
            assert msg[0][0] in expects, f"Unexpected message: {msg}, expect {expects}"
            return msg[0][1:]
        else:
            assert msg[0] in expects, f"Unexpected message: {msg}, expect {expects}"
        return msg

    def close(self):
        self.unregister()
        self.socket.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            pass

