# SPDX-License-Identifier: Apache-2.0
"""Row-resident nucleus filtering, without changing the random sampler.

The input contains FP32 logits after temperature/penalty processing. Each CTA
keeps one vocabulary row resident while selecting its probability boundary.
Equal boundary logits retain the lowest token indices first, as in vLLM's
Triton implementation. No fixed candidate count or distribution approximation
is used. Non-finite rows other than ordinary -inf masks are left unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_local_top_p_kernel(
    LOGITS,
    P,
    ROW_STRIDE,
    P_STRIDE,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    p = tl.load(P + row * P_STRIDE)
    if p < 1.0:
        cols = tl.arange(0, BLOCK)
        valid = cols < VOCAB
        x = tl.load(LOGITS + row * ROW_STRIDE + cols, valid, -float("inf"))
        maximum = tl.max(x, 0)
        # Self-comparison detects NaNs without depending on vendor extras.
        bad = tl.sum((valid & ((x != x) | (x == float("inf")))).to(tl.int32), 0)  # noqa: PLR0124
        if (maximum > -float("inf")) & (bad == 0):
            weights = tl.exp(x - maximum)
            total = tl.sum(weights, 0)
            tail = (1.0 - tl.maximum(p, 0.0)) * total
            low = tl.min(tl.where(x > -float("inf"), x, float("inf")), 0)
            high = maximum
            # Move bounds to present values on every iteration. This gives
            # strict progress even when the floating-point midpoint rounds.
            while low < high:
                pivot = low * 0.5 + high * 0.5
                pivot = tl.where(pivot < high, pivot, low)
                lower = x <= pivot
                mass = tl.sum(tl.where(lower, weights, 0.0), 0)
                if mass <= tail:
                    low = tl.min(tl.where(x > pivot, x, float("inf")), 0)
                else:
                    high = tl.max(tl.where(lower, x, -float("inf")), 0)

            below = tl.sum(tl.where(x < low, weights, 0.0), 0)
            equal = valid & (x == low)
            count = tl.sum(equal.to(tl.int32), 0)
            boundary_weight = tl.exp(low - maximum)
            drop = tl.floor(tl.maximum(tail - below, 0.0) / boundary_weight).to(
                tl.int32
            )
            drop = tl.minimum(tl.maximum(drop, 0), count - 1)
            keep_count = count - drop
            ordinal = tl.cumsum(equal.to(tl.int32), 0)
            keep = (x > low) | (equal & (ordinal <= keep_count))
            result = tl.where(keep, x, -float("inf"))
            tl.store(LOGITS + row * ROW_STRIDE + cols, result, valid)


def row_local_top_p_(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply top-p in place to independent FP32 rows; return ``logits``.

    Top-k, softmax for sampling, RNG state, and output logprobs are the caller's
    responsibility. Last-dimension-contiguous rows may have padding between
    them. ``p`` may have a non-unit stride. At least one finite token survives
    per valid row. Values of p >= 1 bypass filtering.
    """
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("expected two-dimensional FP32 logits")
    if logits.stride(1) != 1 or not logits.is_cuda:
        raise ValueError("expected GPU logits with a contiguous vocabulary dimension")
    rows, vocab = logits.shape
    if not 0 < vocab <= 131072:
        raise ValueError("supported vocabulary size is 1 through 131072")
    if logits.stride(0) < vocab:
        raise ValueError("logits rows must not overlap")
    if p.shape != (rows,) or p.dtype != torch.float32 or p.device != logits.device:
        raise ValueError("expected one FP32 top-p value per row on the same device")
    if rows:
        block = triton.next_power_of_2(vocab)
        # Iluvatar uses 64-lane warps and permits at most 1024 CTA threads.
        warps = 16 if block >= 65536 else (8 if block >= 8192 else 4)
        _row_local_top_p_kernel[(rows,)](
            logits,
            p,
            logits.stride(0),
            p.stride(0),
            vocab,
            block,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return logits
