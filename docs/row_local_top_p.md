# Row-resident top-p filtering on BI-V150

The `v535` branch ports this operator onto the official `v5.3.5` tag
(`a7620cc`). The GPU tests and timings described below were measured on the
earlier BI-V150 image baseline (`4df52d9`), not on this migrated branch.
The kernel and test source are unchanged; GPU validation on `v535` is pending.

`flag_gems.row_local_top_p_(logits, p)` mutates independent FP32 logit rows.
It selects a nucleus; it does not generate random numbers or sample token IDs.

One CTA keeps the row's logits and exponential weights resident. Selection
bisects the logit range, moving each bound to an actual input value. This
avoids a full vocabulary sort, a global candidate buffer, and repeated global
loads during pivot search. Equal boundary logits retain the lowest token IDs
first. FP32 reductions introduce ordinary rounding at the probability boundary.

The vocabulary must have 1–131072 elements and unit inner stride. Rows may be
padded but must not overlap. Per-row FP32 `p` tensors may be strided. At least
one finite token survives when `p <= 0`; `p >= 1` bypasses filtering. Ordinary
`-inf` masks and all-masked rows are supported. NaN/+inf rows are preserved;
their sampling distribution is undefined. Top-k is outside this operator.

The launch is tuned for Iluvatar's 64-lane warps, with at most 16 warps/CTA.
BI-V150 is the validated device. Resource use and performance on other devices
require independent validation.

On a captured MiniCPM5-2B `[64, 130560]` logit tensor with `top_p=0.95`, the
original vLLM filter took 6.733 ms and this operator took 3.032 ms (2.22x).
Both CUDA Graph measurements include the same input reset; 12 samples were
collected. Retained counts matched PyTorch sorting in every row; membership
differences occurred only among equal boundary logits. All retained values
were unchanged. This is an operator result, not an end-to-end throughput claim.

Tests include an independent CPU FP64 nucleus oracle, uniform ties, extreme
finite values, masks, strided metadata, RNG-state preservation and graph replay.
The vLLM adapter intentionally retains its existing path for small batches.
