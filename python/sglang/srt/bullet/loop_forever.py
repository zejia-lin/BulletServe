import asyncio
import signal
import logging
import torch.multiprocessing as torch_mp
from sglang.srt.utils import configure_logger

logger = logging.getLogger(__name__)


class LoopForeverAbstract:
    async def _internal_loop(self):
        raise NotImplementedError

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
        configure_logger(log_level, prefix=prefix)
        asyncio.run(self.start_loop())

    def launch_process(self, log_level="INFO", prefix=""):
        ctx = torch_mp.get_context("spawn")
        self.proc = ctx.Process(target=self.run, args=(log_level, prefix))
        self.proc.start()
        return self.proc
