# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape KDA convolution with private speculative candidate rows."""

import torch
import torch.nn.functional as F

from vllm_gaudi.ops.causal_conv1d_pytorch import _apply_activation, _depthwise_conv1d_tpc


@torch._dynamo.disable
def hpu_kda_conv_update(
    x: torch.Tensor,
    pool: torch.Tensor,
    weight: torch.Tensor,
    load_indices: torch.Tensor,
    store_indices: torch.Tensor,
    accepted: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    bias: torch.Tensor | None = None,
    activation: str | None = 'silu',
) -> torch.Tensor:
    """Update chronological convolution histories from a rectangular token batch.

    ``x`` is [B*L,D], maps are [B,1+capacity], and pool is [slots,P,D].
    Column zero is canonical; column a>0 holds the state after candidate a.
    Accepted counts refer to the PREVIOUS real verification length (validated
    by the caller), independently of this call's static L. Zero selects canonical.
    Only the final W-1 history entries are used. Candidate rows have zero prefixes.

    The caller must append a dedicated garbage row, provide disjoint live write
    destinations (copy-on-write for shared canonicals), and keep all live indices
    below that row. Lengths must be in [0,L]; zero-length positive-index padding
    is inactive too. Tensor-value contracts are caller-validated without host sync.

    L=1 writes only the post-token canonical. L>1 writes the incoming canonical
    plus private post-token candidates. This does NOT publish newly accepted
    prefixes: initial MTP validation requires prefix caching OFF.
    """
    length = max_query_len
    if x.ndim != 2 or pool.ndim != 3 or weight.ndim != 2:
        raise ValueError('Expected x [B*L,D], pool [slots,P,D], and weight [D,W].')
    if load_indices.ndim != 2 or store_indices.shape != load_indices.shape:
        raise ValueError('Load and store maps must have identical [B,1+capacity] shapes.')
    batch, columns = load_indices.shape
    dim = x.shape[1]
    if length < 1 or x.shape[0] != batch * length:
        raise ValueError('KDA convolution requires a rectangular [B*max_query_len,D] input.')
    if columns < 2 or length > columns - 1:
        raise ValueError('max_query_len exceeds candidate capacity.')
    if accepted.shape != (batch, ) or query_start_loc.shape != (batch + 1, ):
        raise ValueError('Expected accepted [B] and query_start_loc [B+1].')
    hist_len = weight.shape[1] - 1
    row_width = pool.shape[1]
    if hist_len < 0 or row_width < hist_len or pool.shape[2] != dim or weight.shape[0] != dim:
        raise ValueError('Incompatible convolution history or channel dimensions.')
    if pool.shape[0] < 1 or (bias is not None and bias.shape != (dim, )):
        raise ValueError('Expected a dedicated garbage row and bias [D].')
    if activation not in (None, 'silu', 'swish'):
        raise ValueError(f'Unsupported activation {activation!r}.')

    garbage = pool.shape[0] - 1
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    live = (lengths > 0) & (load_indices[:, 0] >= 0) & (store_indices[:, 0] >= 0)
    selected_columns = accepted.long().clamp(0, columns - 1)
    selected = load_indices.gather(1, selected_columns[:, None]).squeeze(1)
    live = live & (selected >= 0)
    selected = selected * live.long() + garbage * (~live).long()
    incoming = pool.index_select(0, selected.long())
    incoming = torch.where(live[:, None, None], incoming, torch.zeros_like(incoming))
    history = incoming[:, row_width - hist_len:, :].transpose(1, 2)
    tokens = x.reshape(batch, length, dim)
    stream = torch.cat([history, tokens.to(pool.dtype).transpose(1, 2)], dim=2)
    out = _apply_activation(
        _depthwise_conv1d_tpc(stream, weight.to(pool.dtype), None if bias is None else bias.to(pool.dtype)),
        activation,
    )
    positions = torch.arange(length, device=x.device)
    offsets = positions[:, None] + 1 + torch.arange(hist_len, device=x.device)[None, :]
    histories = stream.index_select(2, offsets.reshape(-1)).reshape(batch, dim, length, hist_len)
    candidates = F.pad(histories.permute(0, 2, 3, 1), (0, 0, row_width - hist_len, 0))
    token_live = live[:, None] & (positions[None, :] < lengths[:, None])
    if length == 1:
        targets = store_indices[:, :1]
        rows = candidates
        valid = live[:, None]
    else:
        targets = store_indices[:, :length + 1]
        rows = torch.cat([incoming[:, None], candidates], dim=1)
        valid = torch.cat([live[:, None], token_live], dim=1)
    valid = valid & (targets >= 0)
    # Every duplicate garbage destination receives the identical zero payload.
    targets = targets * valid.long() + garbage * (~valid).long()
    rows = torch.where(valid[:, :, None, None], rows, torch.zeros_like(rows))
    pool.index_copy_(0, targets.reshape(-1).long(), rows.reshape(-1, row_width, dim))
    out = torch.where(token_live[:, :, None], out.transpose(1, 2), 0)
    return out.reshape(batch * length, dim).to(x.dtype)
