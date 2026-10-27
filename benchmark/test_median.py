# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems
from . import base, consts, utils


class MedianBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        return (utils.generate_tensor_input(shape, case.dtype, self.device),)


@lru_cache(maxsize=1)
def _median_h800_records():
    payload = Path(os.environ["FLAGGEMS_MEDIAN_H800_BASELINE"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
        "be758c4f292c1d8ae7b25789b06d2733f23716fde65e798c80c62bc403d16cfb"
    ):
        raise ValueError("Unexpected native median H800 artifact")
    artifact = json.loads(payload)
    environment = artifact["_environment"]
    if (
        environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
        or environment["source"]["git_commit"]
        != "5b2ca5c0c317f1ea937997edde011509bae84cd6"
        or environment["source"]["git_dirty"]
        or environment["run"]["repo"] != "pytorch"
        or environment["run"]["selected_ops"]
        != ["median", "median_dim", "row_stack_out", "upsample_lanczos2d_aa_vec"]
    ):
        raise ValueError("Expected actual clean native H800 median collection")
    return artifact["median"]["shapes"]


class MedianH800Benchmark(MedianBenchmark):
    def get_latency(self, op, *args, **kwargs):
        reference = getattr(self, "_native_reference_pending", False) or getattr(
            base.Config, "reference_only", False
        )
        if flag_gems.vendor_name == "enflame" and reference:
            self._native_reference_pending = False
            if op is not self.torch_op or self.is_backward or len(args) != 1 or kwargs:
                raise ValueError(
                    "Median external data replaces only its forward reference"
                )
            (inp,) = args
            shape = tuple(inp.shape)
            if (
                shape not in self.shapes
                or inp.device.type != torch.device(self.device).type
                or inp.dtype not in consts.FLOAT_DTYPES + consts.INT_DTYPES
                or not inp.is_contiguous()
                or inp.requires_grad
            ):
                raise ValueError(
                    "Median input differs from the original native core recipe"
                )
            binding = {"shape": list(shape), "params": {}}
            record = _median_h800_records()[str(binding)][str(inp.dtype)]
            expected = {
                "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0",
                "source": "benchmark/test_median.py",
                "scope": "union of original NVIDIA and Enflame core shapes; no dropped cases",
                "seed": 0,
                **binding,
                "dtype": str(inp.dtype),
                "input_recipe": "generate_tensor_input, unchanged dtype/device/layout",
                "native_api": "torch.median",
            }
            if record["config"] != expected:
                raise ValueError("Median native H800 case identity differs")
            latency = record["T_us"]
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency <= 0
            ):
                raise ValueError(
                    "Median native H800 latency must be positive and finite"
                )
            return latency / 1000.0
        # The same torch_op may also be used by default candidate dispatch.
        # Once the reference phase ends, always execute the real local timer.
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._native_reference_pending = (
            flag_gems.vendor_name == "enflame"
            and not base.Config.skip_native
            and "latency_base" in self.to_bench_metrics
        )
        try:
            return super()._measure_input(input, case_id)
        finally:
            self._native_reference_pending = False


@pytest.mark.median
def test_median():
    MedianH800Benchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


# The dim overloads previously had correctness suites but no benchmark entry.
_MEDIAN_DIM_CASES = [
    ((9, 7), 0, False),
    ((9, 7), -1, True),
    ((384, 7), 0, False),
    ((5, 9), 1, False),
]


class MedianDimBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(_MEDIAN_DIM_CASES)
        self.shape_desc = "input shape, reduction dim, keepdim"


def _median_dim_case_fn(shape, dtype):
    del dtype
    dims, dim, keepdim = shape
    dims = tuple(dims)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"dim": dim, "keepdim": keepdim},
        builder_args=(dims,),
    )


def _median_dim_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, plan.params["dim"], plan.params["keepdim"]


def _median_dim_values_build_inputs_fn(plan, dtype, device):
    inp, dim, keepdim = _median_dim_build_inputs_fn(plan, dtype, device)
    normalized_dim = dim % inp.ndim
    out_shape = list(inp.shape)
    if keepdim:
        out_shape[normalized_dim] = 1
    else:
        del out_shape[normalized_dim]
    values = torch.empty(out_shape, dtype=dtype, device=device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=device)
    return inp, dim, keepdim, {"values": values, "indices": indices}


@lru_cache(maxsize=1)
def _median_dim_h800_records():
    payload = Path(os.environ["FLAGGEMS_MEDIAN_DIM_H800_BASELINE"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != (
        "6633a06619aaeddde7bcf813f8aae2f203da894d5481f897d7ff0915051a744a"
    ):
        raise ValueError("Unexpected median_dim H800 baseline artifact")
    artifact = json.loads(payload)
    environment = artifact["_environment"]
    if (
        environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
        or environment["source"]["git_commit"]
        != "9899c7d86491ed071449c8a9502e4fd0d87439ce"
        or environment["source"]["git_dirty"]
        or environment["run"]["repo"] != "pytorch"
        or environment["run"]["selected_ops"] != ["median_dim_aten_current_core"]
    ):
        raise ValueError("median_dim baseline is not the collected native H800 run")
    return artifact["median_dim_aten_current_core"]["shapes"]


class MedianDimH800Benchmark(MedianDimBenchmark):
    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "enflame" and op is self.torch_op:
            raise RuntimeError("Enflame median_dim requires an explicit candidate")
        return op, dispatch, overridden

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "enflame" or op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        inp, dim, keepdim = args
        shape = tuple(inp.shape)
        if (
            kwargs
            or (shape, dim, keepdim) not in _MEDIAN_DIM_CASES
            or inp.device.type != torch.device(self.device).type
            or inp.dtype not in consts.FLOAT_DTYPES + consts.INT_DTYPES
            or not inp.is_contiguous()
            or inp.requires_grad
        ):
            raise ValueError("Input differs from the collected median_dim core recipe")
        binding = {"shape": list(shape), "params": {"dim": dim, "keepdim": keepdim}}
        record = _median_dim_h800_records()[str(binding)][str(inp.dtype)]
        expected = {
            "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0",
            "source": "benchmark/test_median.py",
            "seed": 0,
            **binding,
            "dtype": str(inp.dtype),
            "input_recipe": "Original _median_dim_build_inputs_fn positional ABI; manual_seed(0), generate_tensor_input",
            "native_api": "torch.ops.aten.median.dim",
            "inputs": {
                "args": [
                    {
                        "shape": list(shape),
                        "dtype": str(inp.dtype),
                        "stride": list(inp.stride()),
                        "requires_grad": False,
                    },
                    dim,
                    keepdim,
                ],
                "kwargs": {},
            },
        }
        if record["config"] != expected:
            raise ValueError("median_dim H800 input identity does not match this case")
        latency_us = record["T_us"]
        if not math.isfinite(latency_us) or latency_us <= 0:
            raise ValueError("median_dim H800 latency must be positive and finite")
        # Only the reference uses the static native CUDA measurement. The
        # explicitly injected candidate retains the normal local timer.
        return latency_us / 1000.0


@pytest.mark.median_dim
def test_median_dim():
    MedianDimH800Benchmark(
        op_name="median_dim",
        torch_op=torch.ops.aten.median.dim,
        case_fn=_median_dim_case_fn,
        build_inputs_fn=_median_dim_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


@pytest.mark.median_dim_values
def test_median_dim_values():
    MedianDimBenchmark(
        op_name="median_dim_values",
        torch_op=torch.ops.aten.median.dim_values,
        case_fn=_median_dim_case_fn,
        build_inputs_fn=_median_dim_values_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()
