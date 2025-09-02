import ast
import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
import inspect
import logging
import os
import pickle
import signal
import multiprocessing as mp
import time
import traceback
from typing import Any, Callable, Concatenate, ParamSpec, Tuple, Type, TypeVar
import setproctitle
import zmq
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import torch
import torch.multiprocessing as torch_mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor as _torch_rebuild_cuda_tensor
import zmq.asyncio

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import configure_logger, kill_itself_when_parent_died, kill_process_tree


T = TypeVar("T")
logger = logging.getLogger(__name__)


def share_tensor_to_dict(tensor: torch.Tensor):
    return torch_mp.reductions.reduce_tensor(tensor)


def build_tensor_from_dict(buf) -> torch.Tensor:
    return buf[0](*buf[1])


def get_decorators(cls, method_name):
    # Get the source code of the class
    source = inspect.getsource(cls)
    # Parse the source code into an AST
    tree = ast.parse(source)

    # Find the class definition in the AST
    class_def = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            class_def = node
            break

    if class_def is None:
        raise ValueError(f"Class {cls.__name__} not found in the source code.")

    # Find the method definition in the class
    method_def = None
    for node in class_def.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            method_def = node
            break

    if method_def is None:
        raise ValueError(f"Method {method_name} not found in class {cls.__name__}.")

    # Extract the decorators
    decorators = [ast.dump(decorator) for decorator in method_def.decorator_list]
    return decorators


class require_identity:
    """Decorator, the first argument of the decorated function will be client id."""

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


class dont_response:
    """Decorator, the server will not send response back to the client."""

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


@dataclass
class ServerException:
    exception: Exception
    traceback: str


class ZMQServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def bind(self, target_cls: Type[T], *args, **kwargs):
        self.target_cls = target_cls
        self.target_args = args
        self.target_kwargs = kwargs
        return self

    def _get_special_methods(self):
        reqid = set()
        dontresp = set()
        for name, method in self.instance.__class__.__dict__.items():
            if callable(method) and not name.startswith("_"):
                decorators = get_decorators(self.instance.__class__, name)
                for dec in decorators:
                    if require_identity.__name__ in dec:
                        reqid.add(name)
                    if dont_response.__name__ in dec:
                        dontresp.add(name)
        return reqid, dontresp

    async def _internal_loop(self):
        self.instance = self.target_cls(*self.target_args, **self.target_kwargs)
        self.methods_require_id, self.methods_dont_resp = self._get_special_methods()
        logger.info(f"Methods require identity: {self.methods_require_id}")
        logger.info(f"Methods don't response: {self.methods_dont_resp}")
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        setattr(self.instance, "socket", self.socket)
        logger.info(
            f"[pid={os.getpid()}] RPC server for {self.instance.__class__} started at tcp://{self.host}:{self.port}"
        )
        while True:
            try:
                logger.debug("Waiting for message...")
                identity, buf = await self.socket.recv_multipart()
                msg = pickle.loads(buf)
                logger.debug(f"Received '{msg[0]}' with args {msg[1]}, kwargs {msg[2]}")
                method = getattr(self.instance, msg[0])
                if msg[0] in self.methods_require_id:
                    asyncio.create_task(self._handle_async(method, identity, msg[1], msg[2]))
                else:
                    result = method(*msg[1], **msg[2])
                    if msg[0] not in self.methods_dont_resp:
                        buf = pickle.dumps(result)
                        await self.socket.send_multipart([identity, buf])
                        logger.debug(f"Send response {result} to client '{identity.hex()}'")
            except Exception as e:
                await self._handle_exception(identity, e)

    async def _handle_async(self, method, identity, args, kwargs):
        try:
            result = await method(identity, *args, **kwargs)
            buf = pickle.dumps(result)
            await self.socket.send_multipart([identity, buf])
        except Exception as e:
            await self._handle_exception(identity, e)
            
    async def _handle_exception(self, identity, exception):
        logger.critical(f"Catch exception: {exception}, send to client '{identity.hex()}'", exc_info=True)
        tb = traceback.format_exc()
        await self.socket.send_multipart([identity, pickle.dumps(ServerException(exception, tb))])
        logger.critical(f"Killing the server.")
        kill_process_tree(os.getpid())

    async def start_loop(self):
        self.loop = asyncio.get_running_loop()
        self.server_task = self.loop.create_task(self._internal_loop())

        def signal_handler() -> None:
            self.server_task.cancel()
            self.socket.close()

        self.loop.add_signal_handler(signal.SIGINT, signal_handler)
        self.loop.add_signal_handler(signal.SIGTERM, signal_handler)

        try:
            await self.server_task
        except asyncio.CancelledError:
            logger.critical("Server task was cancelled.")
        finally:
            self.loop.remove_signal_handler(signal.SIGINT)
            self.loop.remove_signal_handler(signal.SIGTERM)
            self.socket.close()

    def run(self, log_level="INFO", prefix=""):
        configure_logger(ServerArgs("", log_level=log_level.upper()), prefix=prefix)
        setproctitle.setproctitle(f"RPC::{prefix.strip()}::{self.port}")
        kill_itself_when_parent_died()
        asyncio.run(self.start_loop())

    def launch_process(self, log_level="INFO", prefix=""):
        ctx = torch_mp.get_context("spawn")
        self.proc = ctx.Process(target=self.run, args=(log_level, prefix))
        self.proc.start()
        return self.proc
    
    def __del__(self):
        try:
            self.socket.close()
        except Exception as e:
            pass
        logger.info(
            f"[pid={os.getpid()}] Close RPC server for {self.target_cls.__name__} at tcp://{self.host}:{self.port}"
        )


def ZMQClient(*, replace_by_default: bool):
    class Client:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.DEALER)
            # self.socket.setsockopt(zmq.IDENTITY, f"{os.getpid()}".encode("utf-8"))
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.executor = ThreadPoolExecutor(10)

        def __del__(self):
            self.executor.shutdown()

        def __init_subclass__(subclass) -> None:
            if not replace_by_default:
                return
            bases = subclass.__bases__
            if len(bases) != 2:
                raise ValueError("The client should inherit from exactly ZMQClient and the target class.")
            this_cls = bases[0]
            target_cls = bases[1]
            origin_init = subclass.__init__

            def init(self, host, port, *args, **kwargs):
                this_cls.__init__(self, host, port)
                origin_init(self, host, port, *args, **kwargs)
                for name, method in inspect.getmembers(subclass, predicate=inspect.isfunction):
                    if not name.startswith("_") and self._is_inherit_from_parent(name):
                        # do not override methods again
                        setattr(self, name, self._create_method(name))

            subclass.__init__ = init

        def _is_inherit_from_parent(self, method_name):
            cls = self.__class__
            method = getattr(cls, method_name)
            for base in inspect.getmro(cls)[2:]:
                if hasattr(base, method_name) and getattr(base, method_name) == method:
                    return True
            return False

        def _create_method(self, method_name: str):
            def method(*args):
                return self.send_and_recv(method_name, *args)

            return method

        def send_and_recv(self, method_name: str, *args, **kwargs):
            logger.debug(f"Sending '{method_name}' with args {args}, kwargs {kwargs}")
            buf = pickle.dumps((method_name, args, kwargs))
            self.socket.send(buf)
            logger.debug("Waiting for response...")
            buf = self.socket.recv()
            result = pickle.loads(buf)
            logger.debug(f"Get response: {result}")
            if isinstance(result, ServerException):
                logger.error(
                    f"Received exception: {result.exception}, the server log is:\n{result.traceback}"
                )
                logger.error("Re-raise the exception in client, please check the server log.")
                traceback.print_stack()
                raise Exception("Server exception")
            return result

        def only_recv(self):
            buf = self.socket.recv()
            result = pickle.loads(buf)
            if isinstance(result, ServerException):
                logger.error(
                    f"Received exception: {result.exception}, the server log is:\n{result.traceback}"
                )
                logger.error("Re-raise the exception in client, please check the server log.")
                traceback.print_stack()
                raise Exception("Server exception")
            return result

        def send_without_recv(self, method_name: str, *args, **kwargs):
            buf = pickle.dumps((method_name, args, kwargs))
            self.socket.send(buf)

    return Client


def create_zmq_simple_client(host: str, port: int):
    class SimpleClient(ZMQClient(replace_by_default=False)):
        def __init__(self, host: str, port: int):
            super().__init__(host, port)

    return SimpleClient(host, port)
