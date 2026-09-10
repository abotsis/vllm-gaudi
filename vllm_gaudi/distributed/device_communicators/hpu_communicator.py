# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator \
    import DeviceCommunicatorBase
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group, get_tp_group, get_ep_group

import os

import habana_frameworks.torch as htorch  # noqa: F401

from vllm_gaudi.v1.worker.hpu_dp_utils import get_hpu_dp_metadata

# See all_reduce(). Default OFF: the mark_step it guards was a stale bridge
# workaround that cost a graph boundary per collective.
_ALLREDUCE_MARKSTEP = os.environ.get("VLLM_HPU_ALLREDUCE_MARKSTEP", "0") == "1"


class HpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = "",
                 **kwargs):
        # Forward any extra keyword arguments (e.g. ``global_ranks``,
        # ``global_world_size``, ``use_all2all``) to the upstream base
        # constructor. Upstream repeatedly churns this signature, so we
        # capture the trailing params via ``**kwargs`` rather than pinning a
        # fixed list that breaks whenever vLLM adds or removes one.
        super().__init__(cpu_group, device, device_group, unique_name, **kwargs)

        self.dp_group: Optional[GroupCoordinator] = None
        self.dp_rank = 0
        self.dp_world_size = 1
        # assume EP is enabled along with DP
        if "ep" in unique_name:
            self.dp_group = get_dp_group()
            self.dp_rank = self.dp_group.rank_in_group
            self.dp_world_size = self.dp_group.world_size
            self.tp_group = get_tp_group()
        self.world_size = dist.get_world_size(group=self.cpu_group)
        self.rank = dist.get_rank(group=self.cpu_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # No mark_step here. The one that used to precede this call was a
        # workaround for an older-bridge bug under PT_HPU_ENABLE_LAZY_COLLECTIVES.
        # It is a hard graph cut, and a decode step issues ~90 all-reduces (2
        # per layer), so it cut the step into ~230 recipes with a boundary
        # after each collective that cost a mid-teens percent of the
        # single-stream decode step, before counting the recipe merges it
        # prevented. Re-tested on bridge 1.24 with a 2-rank chained-graph
        # bench: no mark_step is bit-identical to the workaround and 10%
        # faster between graphs, 32% faster with the collective fused inside
        # the graph. VLLM_HPU_ALLREDUCE_MARKSTEP=1 restores the old behaviour.
        if _ALLREDUCE_MARKSTEP:
            htorch.core.mark_step()
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size, ) + input_size[1:]
        output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
        # All-gather.
        htorch.core.mark_step()
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] + (world_size * input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError("extra_tensors is not supported for HPU")
        # Use dispatch_tensor in the plugin FusedMoEMethod for better performance
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False) -> torch.Tensor:
        if htorch.utils.internal.is_lazy():
            htorch.core.mark_step()
        assert self.dp_group is not None
        assert hidden_states.dim() == 2, "Input hidden states must be 2D"

        dp_metadata = get_hpu_dp_metadata()
        if dp_metadata is not None:
            local_hidden_states = dp_metadata.local_hidden_states
        else:
            local_num_tokens = hidden_states.size(0) // self.world_size if is_sequence_parallel else hidden_states.size(
                0) // self.dp_world_size
            local_hidden_states = torch.empty((local_num_tokens, hidden_states.size(-1)),
                                              device=hidden_states.device,
                                              dtype=hidden_states.dtype)

        torch.distributed.reduce_scatter_tensor(
            local_hidden_states,
            hidden_states,
            group=get_ep_group().device_group if is_sequence_parallel else self.dp_group.device_group)
        hidden_states = local_hidden_states
        return hidden_states
