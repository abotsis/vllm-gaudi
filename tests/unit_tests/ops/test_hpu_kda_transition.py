# SPDX-License-Identifier: Apache-2.0
"""Fast chunk transitions against the independent token-recurrent oracle.

Inputs stay on CPU; no HPU execution is needed for this algebra regression.
"""

import pytest
import torch
import torch.nn.functional as F

import vllm_gaudi.ops.hpu_kda_pytorch as kda
from vllm_gaudi.ops.hpu_kda_eager import recurrent_kda_eager


@pytest.mark.parametrize("scan", ["seq", "parallel"])
@pytest.mark.parametrize("dim,chunk", [(32, 16), (128, 64)])
@pytest.mark.parametrize("case", ["nonzero_single_chunk", "zero_multiple_chunks", "nonaligned_continuation"])
def test_chunk_transition_vs_recurrent(monkeypatch, scan, dim, chunk, case):
    """Catch a transposed state map, including padding and state handoff.

    Single-chunk outputs do not depend on the outgoing transition, so final
    state must also be checked. Slow, channel-varying decay preserves enough
    incoming memory to expose the error instead of washing it out.
    """
    monkeypatch.setattr(kda, "_KDA_SCAN_PARALLEL", scan == "parallel")
    length = chunk if case == "nonzero_single_chunk" else 3 * chunk + 7
    generator = torch.Generator(device="cpu").manual_seed(17)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cpu")

    shape = (2, length, 2, dim)
    q, k, v = (randn(*shape) for _ in range(3))
    rates = (0.7 * randn(2, 1, 2, dim)).exp()
    g = -0.025 * rates * F.softplus(randn(*shape)) - 0.001
    beta = randn(2, length, 2).sigmoid()
    initial = None if case == "zero_multiple_chunks" else 0.3 * randn(2, 2, dim, dim)
    inputs = (q, k, v, g, beta)
    options = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    expected_out, expected_state = recurrent_kda_eager(*inputs, initial_state=initial, **options)

    def fast(values, state):
        return kda.hpu_chunk_kda(*values, initial_state=state, chunk_size=chunk, neumann_iters=16, **options)

    out, state = fast(inputs, initial)
    torch.testing.assert_close(out, expected_out, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(state, expected_state, atol=3e-6, rtol=3e-5)
    if case == "nonaligned_continuation":
        split = chunk + 3
        first_out, first_state = fast(tuple(x[:, :split] for x in inputs), initial)
        _, expected_first_state = recurrent_kda_eager(*(x[:, :split] for x in inputs), initial_state=initial, **options)
        torch.testing.assert_close(first_state, expected_first_state, atol=3e-6, rtol=3e-5)
        second_out, state = fast(tuple(x[:, split:] for x in inputs), first_state)
        torch.testing.assert_close(torch.cat((first_out, second_out), dim=1), expected_out, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(state, expected_state, atol=3e-6, rtol=3e-5)
