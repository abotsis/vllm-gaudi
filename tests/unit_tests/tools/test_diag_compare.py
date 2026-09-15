# SPDX-License-Identifier: Apache-2.0
"""CPU tests for tools/diag_compare.py: pair discovery, routing analysis, delta classes."""
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("diag_compare", ROOT / "tools/diag_compare.py")
assert spec is not None and spec.loader is not None
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def payload(ordinal, phase, bs, records, boundaries):
    return dict(ordinal=ordinal, phase=phase, input_shape=(bs, 1), records=records, boundaries=boundaries)


def record(rid, position, prompt=(1, 2, 3)):
    return dict(request_id=rid, output_position=position, prompt_token_ids=list(prompt))


def test_auto_pairs_reference_is_earliest_single_row_forward():
    data = {
        0: payload(0, "prefill", 1, [record("A", 0)], []),
        1: payload(1, "decode", 1, [record("A", 1)], []),
        3: payload(3, "prefill", 1, [record("B", 0)], []),
        5: payload(5, "decode", 1, [record("B", 1)], []),
        6: payload(6, "decode", 2, [record("B", 2), record("C", 1)], []),
        7: payload(7, "decode", 2, [record("C", 2, prompt=(9, 9))], []),  # different prompt: unpaired
    }
    pairs, kinds = dc.auto_pairs(data)
    assert pairs == ["0:0>3:0", "1:0>5:0", "1:0>6:1"]
    assert [k for k, _ in kinds] == ["control", "control", "fork(rows=2,bs=2)"]


def test_auto_pairs_under_padded_buckets_uses_record_count():
    # Every decode forward is padded to 8 rows; the lone request is still the reference.
    data = {
        1: payload(1, "decode", 8, [record("A", 1)], []),
        5: payload(5, "decode", 8, [record("B", 1)], []),
        6: payload(6, "decode", 8, [record("B", 2), record("C", 1)], []),
    }
    pairs, kinds = dc.auto_pairs(data)
    assert pairs == ["1:0>5:0", "1:0>6:1"]
    assert [k for k, _ in kinds] == ["control", "fork(rows=2,bs=8)"]


def test_route_margin_and_flip_detection():
    bias = torch.zeros(12)
    logits = torch.linspace(-3, 3, 12)
    ids, margin = dc.route(logits, bias, top_k=4)
    assert ids == [11, 10, 9, 8]
    expected = float(torch.sigmoid(logits[8]) - torch.sigmoid(logits[7]))
    assert abs(margin - expected) < 1e-7
    # a bias that promotes expert 0 past expert 8 flips one slot
    bias2 = bias.clone()
    bias2[0] = 10.0
    ids2, _ = dc.route(logits, bias2, top_k=4)
    assert ids2[0] == 0 and 8 not in ids2


def test_routing_rows_use_bias_and_row_selection():
    name = "model.layers.5.mlp.router_logits"
    bias = torch.zeros(1, 6)
    a_rows = torch.tensor([[0., 1., 2., 3., 4., 5.]])
    b_rows = torch.tensor([[9., 9., 9., 9., 9., 9.], [0., 1., 2., 3., 5., 4.]])  # row 1 swaps top-2 order only
    pa = dict(boundaries=[dict(name=name, rows=a_rows), dict(name="model.layers.5.mlp.router_bias", rows=bias)])
    pb = dict(boundaries=[dict(name=name, rows=b_rows), dict(name="model.layers.5.mlp.router_bias", rows=bias)])
    (row, ) = dc.routing_rows(pa, 0, pb, 1, top_k=2)
    assert row["boundary"] == "model.layers.5.mlp.routing_topk" and row["status"] == "ROUTING"
    assert row["changed"] == 0 and set(row["ids_a"]) == set(row["ids_b"]) == {4, 5}
    (row, ) = dc.routing_rows(pa, 0, pb, 1, top_k=1)
    assert row["changed"] == 1 and row["ids_a"] == [5] and row["ids_b"] == [4]
    assert dc.delta_class(row) == "ROUTE "
    assert dc.sort_key(row)[:2] == (5, dc.ORDER.index("mlp.routing_topk"))


def test_delta_class_thresholds_use_rel_l2():
    assert dc.delta_class(dict(status="MATCHED", rel_l2=0.0, max_rel=1e9)) == "ident "
    assert dc.delta_class(dict(status="MATCHED", rel_l2=4e-3, max_rel=1e9)) == "ulp   "
    assert dc.delta_class(dict(status="MATCHED", rel_l2=5e-2)) == "small "
    assert dc.delta_class(dict(status="MATCHED", rel_l2=2e-1)) == "BIG   "
    assert dc.delta_class(dict(status="MISSING")) == "MISSIN"
    assert dc.delta_class(dict(status="MATCHED", note="shape (1,) vs (2,)")) == "SHAPEx"
