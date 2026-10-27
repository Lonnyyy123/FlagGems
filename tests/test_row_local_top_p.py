# SPDX-License-Identifier: Apache-2.0
"""Nucleus membership, mutation, masked rows and graph-replay contracts."""

import math

import pytest
import torch
from flag_gems.fused.row_local_top_p import row_local_top_p_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")


def assert_nucleus(source, filtered, p):
    # CPU float64 softmax is independent of the Triton selection algorithm.
    x, y, levels = source.cpu().double(), filtered.cpu().double(), p.cpu().double()
    for original, actual, level in zip(x, y, levels):
        if level >= 1 or not torch.isfinite(original).any():
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
            continue
        keep = torch.isfinite(actual)
        assert keep.any()
        torch.testing.assert_close(actual[keep], original[keep], rtol=0, atol=0)
        assert torch.isneginf(actual[~keep]).all()
        probabilities = original.softmax(0)
        mass = probabilities[keep].sum().item()
        last = probabilities[keep].min().item()
        target = max(0, level.item())
        assert mass >= target - 2e-6
        assert mass - last <= target + 2e-6
        if (~keep).any():
            assert original[keep].min() >= original[~keep].max()


@pytest.mark.parametrize("vocab", [1, 17, 1024, 8193, 65536, 130560, 131072])
@pytest.mark.parametrize("scale", [1.0, 4.0, 30.0])
def test_random_nucleus(vocab, scale):
    torch.manual_seed(1027)
    x = torch.randn(8, vocab, device="cuda") * scale
    x[:, ::7] = -float("inf")
    x[:, -1] = 0
    p = torch.tensor([0, 0.1, 0.5, 0.9, 0.95, 0.999, 1, 1], device="cuda")
    original = x.clone()
    rng = torch.cuda.get_rng_state().clone()
    assert row_local_top_p_(x, p) is x
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert_nucleus(original, x, p)


@pytest.mark.parametrize("vocab", [128, 130560])
def test_ties_use_token_order(vocab):
    x = torch.zeros(4, vocab, device="cuda")
    p = torch.tensor([0, 0.5, 0.95, 1], device="cuda")
    row_local_top_p_(x, p)
    for row, level in zip(x.cpu(), p.cpu().tolist()):
        count = max(1, math.ceil(level * vocab))
        assert torch.isfinite(row[:count]).all()
        assert torch.isneginf(row[count:]).all()


def test_strided_rows_levels_and_all_masked():
    source = torch.randn(3, 129, device="cuda")
    source[1] = -float("inf")
    backing = torch.full((3, 140), 123.0, device="cuda")
    x = backing[:, :129]
    x.copy_(source)
    p = torch.tensor([0.2, -1, 0.95, -1, 1, -1], device="cuda")[::2]
    row_local_top_p_(x, p)
    assert_nucleus(source, x, p)
    assert (backing[:, 129:] == 123).all()


def test_graph_replay_reads_updated_levels_and_logits():
    x = torch.randn(4, 1024, device="cuda")
    p = torch.full((4,), 0.95, device="cuda")
    row_local_top_p_(x, p)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        row_local_top_p_(x, p)
    source = torch.randn_like(x) * 4
    x.copy_(source)
    p.copy_(torch.tensor([0.1, 0.5, 0.95, 1], device="cuda"))
    graph.replay()
    assert_nucleus(source, x, p)


def test_empty_and_reject_layout():
    x = torch.empty(0, 128, device="cuda")
    assert row_local_top_p_(x, torch.empty(0, device="cuda")) is x
    with pytest.raises(ValueError):
        row_local_top_p_(
            torch.empty(2, 128, device="cuda")[:, ::2], torch.ones(2, device="cuda")
        )
    with pytest.raises(ValueError):
        row_local_top_p_(
            torch.empty(1, 128, device="cuda").expand(8, -1),
            torch.ones(8, device="cuda"),
        )


def test_extreme_finite_logits():
    source = torch.tensor(
        [
            [3e38, -3e38, 0, -float("inf")],
            [-10000, -10001, -10002, -float("inf")],
            [0, -1e-38, 1e-38, -float("inf")],
        ],
        device="cuda",
    )
    p = torch.tensor([0.95, 0.5, 0.5], device="cuda")
    x = source.clone()
    row_local_top_p_(x, p)
    assert_nucleus(source, x, p)
