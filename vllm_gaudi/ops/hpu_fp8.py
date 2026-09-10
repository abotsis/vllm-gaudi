from functools import partial
from typing import Optional

import torch
from vllm_gaudi import envs
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE

from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod as OrigFp8LinearMethod, Fp8MoEMethod,
                                                         Fp8Config)
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpFP8)
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.ops.hpu_fused_moe import (_normalize_moe_activation, model_has_quant_config, select_experts_from_routed)
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

from vllm_gaudi.extension.logger import logger as init_logger
from vllm.model_executor.kernels.linear import _POSSIBLE_FP8_BLOCK_KERNELS, _POSSIBLE_FP8_KERNELS
from vllm.platforms import PlatformEnum
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
)

logger = init_logger()


class HPUPerTensorTorchFP8ScaledMMLinearKernel(PerTensorTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUChannelWiseTorchFP8ScaledMMLinearKernel(ChannelWiseTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """HPU stub for block-scaled FP8 linear.

    The actual computation is handled by HPU-specific ops in
    Fp8LinearMethod.apply(), so this kernel only needs to satisfy
    the kernel selection interface.
    """

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None

    def apply_weights(self, layer, x, bias=None):
        raise NotImplementedError("HPU uses Fp8LinearMethod.apply() directly")

    def apply_block_scaled_mm(self, A, B, As, Bs):
        raise NotImplementedError("HPU uses Fp8LinearMethod.apply() directly")


if PlatformEnum.OOT not in _POSSIBLE_FP8_KERNELS:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [
        HPUPerTensorTorchFP8ScaledMMLinearKernel,
        HPUChannelWiseTorchFP8ScaledMMLinearKernel,
    ]

if PlatformEnum.OOT not in _POSSIBLE_FP8_BLOCK_KERNELS:
    _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [
        HPUFp8BlockScaledMMLinearKernel,
    ]


class Fp8LinearMethod(OrigFp8LinearMethod):

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quant_config = self.quant_config
        if self.block_quant:
            layer = hpu_ops.fp8_block_linear_postprocess_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
            return
        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = hpu_ops.scaled_fp8_quant(layer.weight, scale=None)
            weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.

            weight, weight_scale, input_scale = hpu_ops.process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = (Parameter(input_scale, requires_grad=False) if input_scale is not None else None)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )

        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)
        input_2d = x.view(-1, x.shape[-1])
        output = hpu_ops.apply_fp8_linear_hpu(input=input_2d,
                                              weight=layer.weight,
                                              weight_scale=weight_scale,
                                              input_scale=input_scale,
                                              bias=bias,
                                              trans_B=False)
        return output.view(*x.shape[:-1], -1)

    def dequant_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight


class HPUFp8MoEMethod(Fp8MoEMethod):

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)

        # Disable marlin
        self.use_marlin = False
        self.fp8_backend = False

        # disable DeepGemm support.
        self.allow_deep_gemm = False

        self.use_dispatch_fn = get_config().use_dispatch_fn
        # Snapshot the (static) quant-config flag while the vLLM config context
        # is set; the forward hot path reads this cached value instead.
        self.has_moe_quant_config = model_has_quant_config()

    @property
    def is_monolithic(self) -> bool:
        return True

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        kwargs['weight_loader'] = hpu_ops.synced_weight_loader(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import os  # PT_HPU_GPT_MOE_WT_INTERLEAVED guard below
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.moe_config.is_sequence_parallel)
        else:
            dispatch_fn = None

        if self.block_quant and not envs.VLLM_HPU_FORCE_CHANNEL_FP8:
            layer.moe_op = VllmMixtureOfExpertsOpFP8(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        else:
            layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        if self.block_quant:
            layer = hpu_ops.fp8_block_moe_prepare_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
        else:
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                    raise ValueError("QuantConfig has static quantization, but found "
                                     "activation scales are None.")
                layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)
            layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)

        # Enable the native clamped-SwiGLU fused path HERE, at load, not on the
        # first forward: under HPU graphs the first forward is the capture, and
        # aux tensors created then are not valid graph inputs.
        _lim = getattr(self, "swiglu_limit", None)
        _lim_src = "self"
        if _lim is None:
            _lim = getattr(layer.moe_config, "swiglu_limit", None)
            _lim_src = "moe_config"
        # Unconditional: a narrower diagnostic could not distinguish
        # "swiglu_limit is None" from "this method never ran".
        logger.warning(
            "[GLM] HPUFp8MoEMethod.process_weights_after_loading: lim=%s src=%s "
            "env_fused=%s ignore_limit=%s block_quant=%s moe_op=%s has_enable=%s "
            "wt_interleaved=%s",
            _lim, _lim_src, envs.VLLM_GLM_FUSED_CLAMP_MOE,
            envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT, self.block_quant,
            type(getattr(layer, "moe_op", None)).__name__,
            hasattr(getattr(layer, "moe_op", None), "enable_clamped_swiglu"),
            os.environ.get("PT_HPU_GPT_MOE_WT_INTERLEAVED"))
        # The fused kernel reads w12 in the layout PT_HPU_GPT_MOE_WT_INTERLEAVED
        # selects, and vLLM only ever packs concatenated [gate | up]. Arming
        # under the interleaved layout is silent corruption (cos +0.08 against
        # _silu_clamp_moe), so treat a wrong value as "do not arm" and fall back
        # to the unfused clamped path rather than trusting the launcher.
        _ilv = os.environ.get("PT_HPU_GPT_MOE_WT_INTERLEAVED", "0")
        if _lim is not None and _ilv != "0":
            logger.error(
                "[GLM] NOT arming fused clamped-SwiGLU: "
                "PT_HPU_GPT_MOE_WT_INTERLEAVED=%s selects the interleaved w12 "
                "layout, but vLLM packs w13 concatenated [gate | up]. The fused "
                "kernel would silently compute wrong expert outputs. Unset it "
                "(or set 0) to use the fused path.", _ilv)
        if (_lim is not None and _ilv == "0" and envs.VLLM_GLM_FUSED_CLAMP_MOE
                and not envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT
                and hasattr(layer.moe_op, "enable_clamped_swiglu")):
            _al = getattr(self, "swiglu_alpha", None) or getattr(layer.moe_config, "swiglu_alpha", 1.0) or 1.0
            layer.moe_op.enable_clamped_swiglu(alpha=float(_al), limit=float(_lim))
            logger.warning("[GLM] fused clamped-SwiGLU MoE armed at load (alpha=%s limit=%s)", _al, _lim)
        elif _lim is not None:
            # Loud on purpose: the fallback is ~2x slower and otherwise silent.
            logger.warning(
                "[GLM] fused clamped-SwiGLU MoE NOT armed (swiglu_limit=%s) -- "
                "fused=%s ignore_limit=%s op=%s has_enable=%s. Decode falls back "
                "to the unfused clamp path. (Measured on GLM-5.3: arming this "
                "changed decode throughput by ~nothing, 6.5 vs 6.97 tok/s -- the "
                "earlier '2x' claim came from a retracted baseline. It only "
                "arms on the PER-CHANNEL op, so VLLM_HPU_FORCE_CHANNEL_FP8=0 "
                "disables it by construction.)", _lim,
                envs.VLLM_GLM_FUSED_CLAMP_MOE, envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT,
                type(layer.moe_op).__name__, hasattr(layer.moe_op, "enable_clamped_swiglu"))

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        is_sequence_parallel = layer.moe_config.is_sequence_parallel
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # The HPU mixture_of_experts kernel (including the chunked
        # weighted_sum_reduction_bf16 reduction) compiles for int64 routing
        # tables and bf16 (x.dtype) router weights. The grouped-topk /
        # custom-routing helper returns int32 ids and float32 weights; the
        # regular-topk branch above already normalized them, but the grouped
        # path was previously left unconverted -> the bf16 reduction kernel
        # received float32 router_weights and failed to compile
        # (GLUE_INCOMPATIBLE_DATA_TYPE). Normalize for every routing path so the
        # kernel graph receives dtype-consistent inputs.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, is_sequence_parallel)
        elif is_sequence_parallel:
            # See HPUCompressedTensorsW8A8Fp8MoEMethod.apply_monolithic: at
            # dp_size == 1 with sequence-parallel MoE (TP>1 + EP),
            # MoERunner._maybe_combine reduce-scatters the expert output over the
            # EP group but no paired dispatch all-gather runs (dispatch_fn is
            # wired only for dp_size > 1). Restore symmetry by all-gathering the
            # inputs over the EP group so the combine leaves the token count
            # unchanged for the block's post-experts reshape.
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        activation = _normalize_moe_activation(layer.activation)
        _limit = getattr(self, "swiglu_limit", None)
        if _limit is None:
            _limit = getattr(layer.moe_config, "swiglu_limit", None)
        # Native clamped-SwiGLU fused path: keeps weights fp8 and fuses
        # routing+GEMMs+activation in one Synapse kernel (the path GLM-5.2 uses
        # to reach ~10 tok/s). Falls back to the unfused clamped MoE otherwise.
        if (activation == "silu" and _limit is not None and not envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT
                and envs.VLLM_GLM_FUSED_CLAMP_MOE and getattr(layer.moe_op, "swiglu_limit", None) is not None):
            output = layer.moe_op(x, topk_ids, topk_weights, permuted_weights=True)
            return output.view(*(output.size(0), *input_shape[1:]))

        if (activation == "silu" and _limit is not None
                and not envs.VLLM_HPU_MOE_IGNORE_SWIGLU_LIMIT):
            # silu + swiglu_limit (GLM-5.x): the Habana fused MoE op drops the
            # clamp (and crashes on some shapes); run the clamped-SwiGLU
            # unfused expert path instead.
            from vllm_gaudi.ops.hpu_fused_moe import _silu_clamp_moe
            output = _silu_clamp_moe(layer, x, topk_ids, topk_weights,
                                     limit=float(_limit))
            return output.view(*(output.size(0), *input_shape[1:]))

        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=activation,
        )
        return output.view(*(output.size(0), *input_shape[1:]))


fp8.Fp8LinearMethod = Fp8LinearMethod
fp8.Fp8MoEMethod = HPUFp8MoEMethod
