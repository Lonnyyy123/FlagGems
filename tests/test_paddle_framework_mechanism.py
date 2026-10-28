"""框架核心机制专项测试（技术规范书功能项 1：Paddle 框架支持 Triton 核心机制）。

四组用例与技术要求的对应关系：
  M1 打通 Paddle 与 Triton 的互操作接口（C++/Python 跨语言调用的 ABI 接口）
  M2 Triton wrapper 通过飞桨运行时与设备接口解析张量属性
  M3 动态图支持 Triton 后端算子的动态分发
  M4 无缝支持自动微分等框架核心功能

执行方式：
  CUDA_VISIBLE_DEVICES=<卡号> TRITON_CACHE_DIR=/tmp/tc_mech \
      python -m pytest tests/test_paddle_framework_mechanism.py -q
"""

import logging
import math
import sys

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle
import pytest
import triton
import triton.language as tl

from flag_gems import runtime

# enable() 中 framework_name == "paddle" 分支注册的全部入口，形如 _C_ops.<name>
REGISTERED_OPS = (
    "softmax",
    "bmm",
    "sum",
    "mean",
    "triu",
    "addmm",
    "all",
    "amax",
    "argmax",
    "diag",
    "dot",
    "embedding",
    "ones",
    "zeros",
    "zeros_like",
    "topk",
    "max",
    "min",
    "mv",
    "index_select",
)

# 与 flag_gems.testing.RESOLUTION 一致的相对容差
RTOL = {"float32": 1.3e-6, "float16": 1e-3, "bfloat16": 0.016}
ATOL = 1e-4


def _assert_same(res, ref, dtype):
    """返回值的结构、shape、dtype 与数值都必须与 Paddle 原生入口一致。"""
    assert type(res) is type(ref), f"返回类型不一致: {type(res)} vs {type(ref)}"
    if isinstance(ref, (tuple, list)):
        assert len(res) == len(ref)
        for r, f in zip(res, ref):
            _assert_same(r, f, dtype)
        return
    assert tuple(res.shape) == tuple(ref.shape)
    assert res.dtype == ref.dtype
    if res.dtype in (paddle.bool, paddle.int32, paddle.int64):
        assert bool((res == ref).all())
    else:
        assert bool(
            paddle.allclose(
                paddle.cast(res, "float32"),
                paddle.cast(ref, "float32"),
                rtol=RTOL[dtype],
                atol=ATOL,
            )
        )


def _make_inputs(dtype):
    paddle.seed(1234)
    return {
        "x": paddle.randn([8, 16], dtype=dtype),
        "a": paddle.randn([4, 8, 16], dtype=dtype),
        "b": paddle.randn([4, 16, 8], dtype=dtype),
        "m1": paddle.randn([8, 16], dtype=dtype),
        "m2": paddle.randn([16, 8], dtype=dtype),
        "bias": paddle.randn([8, 8], dtype=dtype),
        "v": paddle.randn([16], dtype=dtype),
        "v2": paddle.randn([16], dtype=dtype),
        "xb": paddle.cast(paddle.randint(0, 2, [8, 16]), "bool"),
        "idx": paddle.randint(0, 8, [4, 3]),
        "w": paddle.randn([8, 16], dtype=dtype),
        "ii": paddle.to_tensor([0, 2, 4], dtype="int64"),
    }


# 全部按位置参数调用 Paddle 公共 API，调用约定与被替换的 C++ 入口 _C_ops.<op> 保持一致
API_CALLS = {
    "softmax": lambda g, d: paddle.nn.functional.softmax(g["x"], -1),
    "bmm": lambda g, d: paddle.bmm(g["a"], g["b"]),
    "sum": lambda g, d: paddle.sum(g["x"], 1),
    "mean": lambda g, d: paddle.mean(g["x"], 1),
    "triu": lambda g, d: paddle.triu(g["x"], 1),
    "addmm": lambda g, d: paddle.addmm(g["bias"], g["m1"], g["m2"], 0.7, 1.3),
    "all": lambda g, d: paddle.all(g["xb"]),
    "amax": lambda g, d: paddle.amax(g["x"], 1),
    "argmax": lambda g, d: paddle.argmax(g["x"], 1),
    "diag": lambda g, d: paddle.diag(g["v"]),
    "dot": lambda g, d: paddle.dot(g["v"], g["v2"]),
    "embedding": lambda g, d: paddle.nn.functional.embedding(g["idx"], g["w"]),
    "ones": lambda g, d: paddle.ones([4, 5], d),
    "zeros": lambda g, d: paddle.zeros([4, 5], d),
    "zeros_like": lambda g, d: paddle.zeros_like(g["x"]),
    "topk": lambda g, d: paddle.topk(g["x"], 3),
    "max": lambda g, d: paddle.max(g["x"]),
    "min": lambda g, d: paddle.min(g["x"]),
    "mv": lambda g, d: paddle.mv(g["x"], g["v"]),
    "index_select": lambda g, d: paddle.index_select(g["x"], g["ii"], 0),
}


# ---------------------------------------------------------------------------
# M1  跨语言调用的 ABI 接口
# ---------------------------------------------------------------------------


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_native_entry_is_cpp_extension_symbol(op_name):
    """被接管的 _C_ops.<op> 入口本身是 C++ 扩展 libpaddle.so 导出的符号。"""
    entry = getattr(paddle._C_ops, op_name)
    assert type(entry).__name__ in ("builtin_function_or_method", "PyCapsule")
    assert "libpaddle" in entry.__module__
    assert paddle.base.libpaddle.__file__.endswith(".so")


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_abi_call_convention_compatible(op_name):
    """以 C++ 入口的位置参数约定调用，FlagGems 实现的入参与返回结构必须一致。"""
    dtype = "float32"
    g = _make_inputs(dtype)
    call = API_CALLS[op_name]
    ref = call(g, dtype)
    with flag_gems.use_gems():
        res = call(g, dtype)
    _assert_same(res, ref, dtype)


# 按飞桨公共 API 的关键字约定调用（关键字名须与被替换的 C++ 入口一致）
KEYWORD_CALLS = {
    "softmax": lambda g, d: paddle.nn.functional.softmax(g["x"], axis=-1),
    "sum": lambda g, d: paddle.sum(g["x"], axis=1, keepdim=False),
    "mean": lambda g, d: paddle.mean(g["x"], axis=1, keepdim=False),
    "triu": lambda g, d: paddle.triu(g["x"], diagonal=1),
    "addmm": lambda g, d: paddle.addmm(
        g["bias"], g["m1"], g["m2"], beta=0.7, alpha=1.3
    ),
    "all": lambda g, d: paddle.all(g["xb"], axis=1, keepdim=False),
    "amax": lambda g, d: paddle.amax(g["x"], axis=1, keepdim=False),
    "argmax": lambda g, d: paddle.argmax(g["x"], axis=1, keepdim=False),
    "diag": lambda g, d: paddle.diag(g["v"], offset=0),
    "embedding": lambda g, d: paddle.nn.functional.embedding(
        g["idx"], g["w"], padding_idx=None, sparse=False
    ),
    "ones": lambda g, d: paddle.ones(shape=[4, 5], dtype=d),
    "zeros": lambda g, d: paddle.zeros(shape=[4, 5], dtype=d),
    "zeros_like": lambda g, d: paddle.zeros_like(g["x"], dtype=d),
    "topk": lambda g, d: paddle.topk(g["x"], k=3, axis=-1, largest=True, sorted=True),
    "max": lambda g, d: paddle.max(g["x"], axis=1, keepdim=False),
    "min": lambda g, d: paddle.min(g["x"], axis=1, keepdim=False),
    "index_select": lambda g, d: paddle.index_select(g["x"], index=g["ii"], axis=0),
}


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", sorted(KEYWORD_CALLS))
def test_abi_keyword_convention_compatible(op_name):
    """以飞桨公共 API 的关键字约定调用，形参名与 C++ 入口一致。"""
    dtype = "float32"
    g = _make_inputs(dtype)
    call = KEYWORD_CALLS[op_name]
    ref = call(g, dtype)
    with flag_gems.use_gems():
        res = call(g, dtype)
    _assert_same(res, ref, dtype)


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", ["amax", "argmax"])
def test_abi_accepts_torch_compat_dim_keyword(op_name):
    """torch 兼容层以 dim= 调用时同样兼容，与 GN-0001 的调用路径一致。"""
    x = paddle.randn([8, 16], dtype="float32")
    ref = getattr(paddle, op_name)(x, 1)
    with flag_gems.use_gems():
        res = getattr(paddle, op_name)(x, dim=1)
    _assert_same(res, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
def test_triton_launcher_is_compiled_c_extension():
    """Triton 侧的 kernel 装载入口是编译产出的 C 扩展（cuda_utils*.so）。"""
    x = paddle.randn([64, 64], dtype="float32")
    with flag_gems.use_gems():
        paddle.nn.functional.softmax(x, -1)
    paddle.device.synchronize()
    utils = triton.runtime.driver.active.utils
    assert type(utils.load_binary).__name__ == "builtin_function_or_method"
    mod = sys.modules["cuda_utils"]
    assert mod.__file__.endswith(".so")
    assert type(mod.__loader__).__name__ == "ExtensionFileLoader"


@triton.jit
def _gather_strided_kernel(out_ptr, in_ptr, s0, s1, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    val = tl.load(in_ptr + row * s0 + cols * s1, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, val, mask=mask)


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
def test_raw_pointer_crosses_language_boundary():
    """飞桨张量直接交给 Triton kernel：裸地址与 stride 经 C 扩展透传后数据正确。"""
    x = paddle.randn([7, 9], dtype="float32")
    xt = x.t()  # 非连续视图
    out = paddle.empty([9, 7], dtype="float32")
    s0, s1 = xt.stride()
    _gather_strided_kernel[(9,)](out, xt, s0, s1, 7, BLOCK=8)
    paddle.device.synchronize()
    assert isinstance(xt.data_ptr(), int) and xt.data_ptr() > 0
    assert bool((out == xt.contiguous()).all())


# ---------------------------------------------------------------------------
# M2  经飞桨运行时与设备接口解析张量属性
# ---------------------------------------------------------------------------


def _layout(name, dtype="float32"):
    """返回 (基张量, 待解析张量)。"""
    base = paddle.randn([4, 6], dtype=dtype)
    if name == "contiguous":
        return base, base
    if name == "transposed":
        return base, base.t()
    if name == "sliced":
        return base, base[1:3, 2:5]
    if name == "scalar":
        t = paddle.randn([], dtype=dtype)
        return t, t
    if name == "empty":
        t = paddle.empty([0, 3], dtype=dtype)
        return t, t
    t = paddle.randn([2, 3, 4, 5, 6], dtype=dtype)
    return t, t


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
@pytest.mark.parametrize(
    "layout", ["contiguous", "transposed", "sliced", "scalar", "empty", "rank5"]
)
def test_runtime_resolves_tensor_attributes(layout):
    """shape / stride / dtype / element_size / place / data_ptr 由飞桨运行时解析且自洽。"""
    base, t = _layout(layout)
    numel = math.prod(t.shape)
    assert tuple(t.shape) == t.numpy().shape
    assert len(t.stride()) == t.dim() == len(t.shape)
    assert t.dtype == paddle.float32
    assert t.element_size() == 4
    assert t.place.is_gpu_place()
    assert t.place.gpu_device_id() == (
        paddle.framework._current_expected_place().gpu_device_id()
    )
    if numel > 0:
        assert isinstance(t.data_ptr(), int) and t.data_ptr() > 0
    if layout == "transposed":
        assert not t.is_contiguous()
        assert tuple(t.shape) == (6, 4)
        assert tuple(t.stride()) == (1, 6)  # 转置后内层 stride 为基张量的行距
    if layout == "sliced":
        assert t.data_ptr() == base.data_ptr() + (1 * 6 + 2) * t.element_size()
        assert tuple(t.shape) == (2, 3)
    if layout == "scalar":
        assert t.dim() == 0 and tuple(t.stride()) == ()


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
def test_view_offset_pointer_arithmetic(dtype):
    """视图的首地址偏移 = 元素偏移 × element_size，按 dtype 逐一核对。"""
    base = paddle.zeros([4, 6], dtype=dtype)
    view = base[2:, 3:]
    assert view.data_ptr() - base.data_ptr() == (2 * 6 + 3) * base.element_size()
    assert tuple(view.shape) == (2, 3)
    assert tuple(view.stride()) == (6, 1)


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
def test_device_guard_uses_paddle_place_interface():
    """device_guard 经 _current_expected_place 判定设备，同设备走零开销路径。"""
    x = paddle.randn([8], dtype="float32")
    cur = paddle.framework._current_expected_place()
    assert runtime.device_guard(x) is runtime._null_guard
    assert runtime.device_guard(x.place) is runtime._null_guard
    with runtime.device_guard(x):
        assert paddle.framework._current_expected_place() == cur


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
@pytest.mark.parametrize("op", ["softmax", "sum", "triu", "index_select"])
def test_noncontiguous_input_matches_native(op):
    """非连续输入下结果仍与原生一致，说明 stride 解析正确驱动了 kernel。"""
    x = paddle.randn([16, 24], dtype="float32")
    xt = x.t()
    calls = {
        "softmax": lambda t: paddle.nn.functional.softmax(t, -1),
        "sum": lambda t: paddle.sum(t, 1),
        "triu": lambda t: paddle.triu(t, 1),
        "index_select": lambda t: paddle.index_select(
            t, paddle.to_tensor([0, 3, 5], dtype="int64"), 0
        ),
    }
    ref = calls[op](xt)
    with flag_gems.use_gems():
        res = calls[op](xt)
    _assert_same(res, ref, "float32")


# ---------------------------------------------------------------------------
# M3  动态图下 Triton 算子的动态分发
# ---------------------------------------------------------------------------


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_entry_swapped_and_restored(op_name):
    """进入作用域后入口指向 FlagGems 实现，退出后还原为同一个 C++ 入口对象。"""
    original = getattr(paddle._C_ops, op_name)
    with flag_gems.use_gems():
        patched = getattr(paddle._C_ops, op_name)
        assert patched is not original
        assert patched is getattr(flag_gems, op_name)
    assert getattr(paddle._C_ops, op_name) is original


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_registered_op_list_matches_config():
    """实际注册的入口集合与 enable() 的 paddle 分支配置一致，不多不少。"""
    with flag_gems.use_gems():
        registered = sorted(flag_gems.all_ops())
    assert registered == sorted(f"_C_ops.{name}" for name in REGISTERED_OPS)


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_triton_implementation_is_actually_invoked(caplog):
    """同一调用语句：作用域外无 FlagGems 记录，作用域内进入 Triton 实现。"""
    x = paddle.randn([32, 64], dtype="float32")
    with caplog.at_level(logging.DEBUG, logger="flag_gems"):
        paddle.nn.functional.softmax(x, -1)
        native = [r for r in caplog.records if r.name.startswith("flag_gems")]
        with flag_gems.use_gems():
            paddle.nn.functional.softmax(x, -1)
        gems = [r for r in caplog.records if r.name.startswith("flag_gems")]
    assert native == []
    assert any("GEMS SOFTMAX" in r.getMessage() for r in gems)


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("excluded", ["triu_paddle", "sum_paddle"])
def test_excluded_op_falls_back_to_native(excluded):
    """exclude 指定的算子保持原生入口，其余算子照常分发。"""
    name = excluded.replace("_paddle", "")
    original = getattr(paddle._C_ops, name)
    with flag_gems.use_gems(exclude=[excluded]):
        assert getattr(paddle._C_ops, name) is original
        assert paddle._C_ops.softmax is flag_gems.softmax
    assert getattr(paddle._C_ops, name) is original


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_repeated_scopes_restore_entry_each_time():
    """连续三次进出作用域，入口每次都能复位且数值稳定。"""
    original = paddle._C_ops.softmax
    x = paddle.randn([8, 16], dtype="float32")
    ref = paddle.nn.functional.softmax(x, -1)
    for _ in range(3):
        with flag_gems.use_gems():
            res = paddle.nn.functional.softmax(x, -1)
        assert paddle._C_ops.softmax is original
        _assert_same(res, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("shape", [(1, 2), (8, 16), (64, 128), (200, 3)])
def test_dynamic_shape_dispatch(shape):
    """动态图下同一入口连续承接不同 shape，无需重新注册。"""
    x = paddle.randn(list(shape), dtype="float32")
    ref = paddle.nn.functional.softmax(x, -1)
    with flag_gems.use_gems():
        res = paddle.nn.functional.softmax(x, -1)
    _assert_same(res, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_data_dependent_control_flow_dispatch():
    """分支条件依赖张量取值（动态图特性），两条分支都落到 Triton 实现。"""
    x = paddle.randn([8, 16], dtype="float32")

    def run(t):
        if float(paddle.sum(t)) > 0:
            return paddle.nn.functional.softmax(t, -1)
        return paddle.triu(t, 1)

    pos, neg = paddle.abs(x), -paddle.abs(x)
    ref_pos, ref_neg = run(pos), run(neg)
    with flag_gems.use_gems():
        res_pos, res_neg = run(pos), run(neg)
    _assert_same(res_pos, ref_pos, "float32")
    _assert_same(res_neg, ref_neg, "float32")


# ---------------------------------------------------------------------------
# M4  自动微分等框架核心功能
# ---------------------------------------------------------------------------


def _leaf(shape, dtype):
    t = paddle.cast(paddle.randn(shape, dtype="float32"), dtype)
    t.stop_gradient = False
    return t


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_backward_path_goes_through_pylayer():
    """带反向的算子以 paddle.autograd.PyLayer 接入动态图自动微分。"""
    for name in ("softmax", "embedding"):
        entry = getattr(flag_gems, name)
        owner = getattr(entry, "__self__", None)
        assert owner is not None, f"{name} 未以 PyLayer.apply 形式注册"
        assert issubclass(owner, paddle.autograd.PyLayer)


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_softmax_backward_matches_native(dtype):
    """softmax 反向梯度与 Paddle 原生一致，且输出参与建图。"""
    x = _leaf([32, 64], dtype)
    y = paddle.nn.functional.softmax(x, -1)
    y.backward(paddle.ones_like(y))
    ref = x.grad.clone()
    x.clear_grad()
    with flag_gems.use_gems():
        y2 = paddle.nn.functional.softmax(x, -1)
        assert not y2.stop_gradient
        y2.backward(paddle.ones_like(y2))
    _assert_same(x.grad, ref, dtype)
    x.clear_grad()


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_embedding_weight_grad_matches_native(caplog):
    """embedding 权重梯度与原生一致，且反向确实走 Triton 反向 kernel。"""
    w = _leaf([64, 32], "float32")
    idx = paddle.randint(0, 64, [8, 4])
    out = paddle.nn.functional.embedding(idx, w)
    out.backward(paddle.ones_like(out))
    ref = w.grad.clone()
    w.clear_grad()
    with caplog.at_level(logging.DEBUG, logger="flag_gems"):
        with flag_gems.use_gems():
            out2 = paddle.nn.functional.embedding(idx, w)
            out2.backward(paddle.ones_like(out2))
    messages = [r.getMessage() for r in caplog.records if r.name.startswith("flag_gems")]
    assert any("GEMS EMBEDDING FORWARD" in m for m in messages)
    assert any("GEMS EMBEDDING BACKWARD" in m for m in messages)
    _assert_same(w.grad, ref, "float32")
    w.clear_grad()


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_chained_triton_ops_backward():
    """梯度跨 embedding → softmax → sum 三个 Triton 算子回传。"""
    w = _leaf([64, 32], "float32")
    idx = paddle.randint(0, 64, [8, 4])

    def forward():
        return paddle.sum(paddle.nn.functional.softmax(
            paddle.nn.functional.embedding(idx, w), -1))

    loss = forward()
    loss.backward()
    ref = w.grad.clone()
    w.clear_grad()
    with flag_gems.use_gems():
        loss2 = forward()
        loss2.backward()
    _assert_same(w.grad, ref, "float32")
    _assert_same(loss2, loss, "float32")
    w.clear_grad()


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_no_grad_scope_builds_no_graph():
    """no_grad 作用域内不建图，数值仍与原生一致。"""
    x = _leaf([8, 16], "float32")
    with flag_gems.use_gems(), paddle.no_grad():
        y = paddle.nn.functional.softmax(x, -1)
    assert y.stop_gradient
    ref = paddle.nn.functional.softmax(x, -1)
    _assert_same(y, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_stop_gradient_input_produces_no_grad():
    """输入 stop_gradient=True 时不产生梯度，与原生语义一致。"""
    x = paddle.randn([8, 16], dtype="float32")
    assert x.stop_gradient
    with flag_gems.use_gems():
        y = paddle.nn.functional.softmax(x, -1)
    assert y.stop_gradient
    assert x.grad is None
