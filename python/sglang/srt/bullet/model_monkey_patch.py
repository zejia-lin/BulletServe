import inspect
from types import SimpleNamespace
from typing import Any, Callable, Protocol
import torch
import math
import logging

from sglang.srt.distributed.parallel_state import GroupCoordinator


logger = logging.getLogger(__name__)

class LayerwiseForwardModel(Protocol):
    """Type hint only"""

    cur_step: int
    layers_per_step: int
    pp_start_layer: int
    pp_end_layer: int
    pp_last_rank: bool
    pp_first_rank: bool
    num_steps: int
    forward: Callable[..., Any]

    class Model:
        start_layer: int
        end_layer: int
        pp_group: GroupCoordinator
        forward: Callable[..., Any]

    model: Model


def model_monkey_patch_layerwise_forward(self: LayerwiseForwardModel, layers_per_step: int) -> None:
    self.layers_per_step = layers_per_step
    self.pp_start_layer = self.model.start_layer
    self.pp_end_layer = self.model.end_layer
    self.pp_last_rank = self.model.pp_group.is_last_rank
    self.pp_first_rank = self.model.pp_group.is_first_rank
    self.cur_step = 0
    self.num_steps = math.ceil((self.pp_end_layer - self.pp_start_layer) / self.layers_per_step)

    def forward(self: LayerwiseForwardModel, *args, **kwargs):

        # For preprocessor and postprocessors
        if self.cur_step == 0 and self.pp_first_rank:
            self.model.pp_group.is_first_rank = True
        else:
            self.model.pp_group.is_first_rank = False

        if self.cur_step == self.num_steps - 1 and self.pp_last_rank:
            self.model.pp_group.is_last_rank = True
        else:
            self.model.pp_group.is_last_rank = False

        # For layers
        self.model.start_layer = int(self.pp_start_layer + self.cur_step * self.layers_per_step)
        self.model.end_layer = int(
            min(self.pp_start_layer + (self.cur_step + 1) * self.layers_per_step, self.pp_end_layer)
        )

        # Update step
        self.cur_step += 1
        if self.cur_step == self.num_steps:
            self.cur_step = 0

        return self.origin_forward(*args, **kwargs)

    # Make signature compatible
    support_pp = "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
    if support_pp:
        def layerwise_forward(
            self: LayerwiseForwardModel,
            *args,
            pp_proxy_tensors = None,
            **kwargs
        ):
            return forward(self, *args, pp_proxy_tensors=pp_proxy_tensors, **kwargs)
    else:
        def layerwise_forward(self: LayerwiseForwardModel, *args, **kwargs):
            return forward(self, *args, **kwargs)

    self.origin_forward = self.forward
    self.forward = layerwise_forward.__get__(self)

    return self.num_steps


if __name__ == "__main__":
    pass
