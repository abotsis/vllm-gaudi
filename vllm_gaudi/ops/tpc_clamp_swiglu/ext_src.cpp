// SPDX-License-Identifier: Apache-2.0
// PyTorch custom-op binding for the clamp_swiglu_fwd_bf16 TPC kernel.
// Compiled as a torch extension by loader.py; the GUID resolves through
// GC_KERNEL_PATH to libclamp_swiglu_kernels.so.

#include <torch/extension.h>
#include <ATen/Tensor.h>
#include "hpu_custom_op.h"

struct ClampSwigluNodeParams
{
    float limit;
};

static std::vector<int64_t> clamp_swiglu_out_shape(const at::Stack& stack)
{
    auto               h     = stack[0].toTensor();
    auto               sizes = h.sizes();
    std::vector<int64_t> out(sizes.begin(), sizes.end());
    out[out.size() - 1] /= 2;
    return out;
}

static std::shared_ptr<void> clamp_swiglu_fill_params(const at::Stack& stack, size_t& size)
{
    HPU_PARAMS_STUB(ClampSwigluNodeParams);
    params->limit = static_cast<float>(stack[1].toDouble());
    return params;
}

static at::Tensor clamp_swiglu_hpu(const at::Tensor& input, double limit)
{
    const auto& desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::clamp_swiglu");
    auto       res   = const_cast<habana::custom_op::HabanaCustomOpDescriptor&>(desc).execute(
        {input, c10::IValue(limit)});
    return res[0];
}

TORCH_LIBRARY(custom_op, m)
{
    m.def("clamp_swiglu(Tensor input, float limit) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m)
{
    m.impl("clamp_swiglu", &clamp_swiglu_hpu);
}

namespace
{
const bool reg_clamp_swiglu = []() -> bool
{
    std::vector<habana::custom_op::InputDesc> inputs = {
        {habana::custom_op::input_type::TENSOR, 0},
        {habana::custom_op::input_type::USER_PARAMS, 1}};
    std::vector<habana::custom_op::OutputDesc> outputs = {
        {0, c10::ScalarType::BFloat16, clamp_swiglu_out_shape}};
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::clamp_swiglu", "clamp_swiglu_fwd_bf16", inputs, outputs, clamp_swiglu_fill_params);

    return true;
}();
} // namespace

// cpp_extension.load() imports the module as a python extension — provide a
// (near-empty) pybind surface; the real op lives in torch.ops.custom_op.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("registered", []() { return true; });
}
