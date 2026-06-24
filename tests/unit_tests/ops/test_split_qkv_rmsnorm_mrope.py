# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for the Ascend split-qkv-rmsnorm-mrope fused op.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest
import torch


SOURCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "ascend"
    / "impl"
    / "split_qkv_rmsnorm_mrope.py"
)


def _load_module_with_stubs(monkeypatch: pytest.MonkeyPatch):
    """Load the target module without requiring vLLM or Ascend packages."""

    vllm_mod = types.ModuleType("vllm")
    triton_utils_mod = types.ModuleType("vllm.triton_utils")
    utils_mod = types.ModuleType("vllm.utils")
    torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")

    class _FakeTriton:
        @staticmethod
        def jit(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    triton_utils_mod.tl = types.SimpleNamespace(constexpr=object())
    triton_utils_mod.triton = _FakeTriton()
    torch_utils_mod.direct_register_custom_op = lambda **kwargs: None

    vllm_mod.triton_utils = triton_utils_mod
    vllm_mod.utils = utils_mod
    utils_mod.torch_utils = torch_utils_mod

    vllm_ascend_mod = types.ModuleType("vllm_ascend")
    ops_mod = types.ModuleType("vllm_ascend.ops")
    triton_mod = types.ModuleType("vllm_ascend.ops.triton")
    ascend_triton_utils_mod = types.ModuleType(
        "vllm_ascend.ops.triton.triton_utils"
    )

    ascend_triton_utils_mod.extract_slice = lambda x, offsets, sizes, strides: x
    ascend_triton_utils_mod.insert_slice = (
        lambda target, source, offsets, sizes, strides: target
    )
    ascend_triton_utils_mod.get_vectorcore_num = lambda: 1

    vllm_ascend_mod.ops = ops_mod
    ops_mod.triton = triton_mod
    triton_mod.triton_utils = ascend_triton_utils_mod

    for name, module in {
        "vllm": vllm_mod,
        "vllm.triton_utils": triton_utils_mod,
        "vllm.utils": utils_mod,
        "vllm.utils.torch_utils": torch_utils_mod,
        "vllm_ascend": vllm_ascend_mod,
        "vllm_ascend.ops": ops_mod,
        "vllm_ascend.ops.triton": triton_mod,
        "vllm_ascend.ops.triton.triton_utils": ascend_triton_utils_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_split_qkv_rmsnorm_mrope_test_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(module_name, SOURCE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeKernel:
    def __init__(self):
        self.grid = None
        self.args = None

    def __getitem__(self, grid):
        self.grid = grid

        def launcher(*args):
            self.args = args

        return launcher


def _reference_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
):
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0
    num_tokens = qkv.shape[0]
    rope_dim = head_size if rope_dim is None else rope_dim
    half_rope_dim = rope_dim // 2

    if has_gate:
        q_gate = qkv[:, : q_size + gate_size].float().reshape(
            num_tokens, num_q_heads, head_size * 2
        )
        q = q_gate[:, :, :head_size]
        gate = q_gate[:, :, head_size:].reshape(num_tokens, q_size)
    else:
        q = qkv[:, :q_size].float().reshape(num_tokens, num_q_heads, head_size)
        gate = torch.empty(num_tokens, 0, device=qkv.device, dtype=qkv.dtype)

    k_start = q_size + gate_size
    v_start = k_start + kv_size
    k = qkv[:, k_start:v_start].float().reshape(
        num_tokens, num_kv_heads, head_size
    )
    v = qkv[:, v_start : v_start + kv_size]

    def rms_norm(x, weight, bias):
        variance = (x * x).sum(dim=-1, keepdim=True) / head_size
        y = x * torch.rsqrt(variance + eps)
        y = y * weight.float().view(1, 1, head_size)
        if bias is not None:
            y = y + bias.float().view(1, 1, head_size)
        return y

    q = rms_norm(q, q_weight, q_bias)
    k = rms_norm(k, k_weight, k_bias)

    cos_sin = cos_sin.reshape(3, num_tokens, rope_dim)
    offsets = torch.arange(half_rope_dim, device=qkv.device)
    if is_interleaved:
        h_mask = ((offsets % 3) == 1) & (offsets <= 3 * mrope_section[1])
        w_mask = ((offsets % 3) == 2) & (offsets <= 3 * mrope_section[2])
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = offsets < mrope_section[0]
        h_mask = (mrope_section[0] - 1 < offsets) & (
            offsets < mrope_section[0] + mrope_section[1]
        )
        w_mask = (mrope_section[0] + mrope_section[1] - 1 < offsets) & (
            offsets < sum(mrope_section)
        )

    def mrope(x):
        cos_half = torch.zeros(num_tokens, half_rope_dim, device=qkv.device)
        sin_half = torch.zeros(num_tokens, half_rope_dim, device=qkv.device)
        for axis, mask in enumerate((t_mask, h_mask, w_mask)):
            cos_half[:, mask] = cos_sin[axis, :, :half_rope_dim][:, mask].float()
            sin_half[:, mask] = cos_sin[axis, :, half_rope_dim:rope_dim][
                :, mask
            ].float()
        cos = cos_half.repeat(1, 2).unsqueeze(1)
        sin = sin_half.repeat(1, 2).unsqueeze(1)

        x_rope = x[:, :, :rope_dim]
        x1 = x_rope[:, :, :half_rope_dim]
        x2 = x_rope[:, :, half_rope_dim:rope_dim]
        rotated = torch.cat((-x2, x1), dim=-1)
        roped = rotated * sin + x_rope * cos
        if rope_dim == head_size:
            return roped
        return torch.cat((roped, x[:, :, rope_dim:]), dim=-1)

    q = mrope(q).reshape(num_tokens, q_size).to(qkv.dtype)
    k = mrope(k).reshape(num_tokens, kv_size).to(qkv.dtype)
    return q, k, v, gate.to(qkv.dtype)


@pytest.mark.parametrize("has_gate", [False, True])
def test_fake_impl_returns_expected_shapes(monkeypatch, has_gate):
    module = _load_module_with_stubs(monkeypatch)

    qkv = torch.empty(5, 48)
    outputs = module.triton_split_qkv_rmsnorm_mrope_fake(
        qkv=qkv,
        q_weight=torch.empty(8),
        k_weight=torch.empty(8),
        cos_sin=torch.empty(3, 5, 8),
        num_q_heads=4,
        num_kv_heads=1,
        head_size=8,
        eps=1e-6,
        mrope_section=[1, 2, 1],
        is_interleaved=False,
        has_gate=has_gate,
    )

    q, k, v, gate = outputs
    assert q.shape == (5, 32)
    assert k.shape == (5, 8)
    assert v.shape == (5, 8)
    assert gate.shape == (5, 32 if has_gate else 0)


def test_wrapper_allocates_outputs_and_launches_kernel(monkeypatch):
    module = _load_module_with_stubs(monkeypatch)
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(module, "get_vectorcore_num", lambda: 4)
    monkeypatch.setattr(module, "split_qkv_rmsnorm_mrope_kernel", fake_kernel)

    qkv = torch.empty(6, 80)
    q_bias = torch.empty(8)
    k_bias = torch.empty(8)
    q, k, v, gate = module.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=torch.empty(8),
        k_weight=torch.empty(8),
        cos_sin=torch.empty(3, 6, 4),
        num_q_heads=4,
        num_kv_heads=1,
        head_size=8,
        eps=1e-6,
        mrope_section=[1, 1, 0],
        is_interleaved=True,
        rope_dim=4,
        q_bias=q_bias,
        k_bias=k_bias,
        has_gate=True,
    )

    assert fake_kernel.grid == (4,)
    assert q.shape == (6, 32)
    assert k.shape == (6, 8)
    assert v.shape == (6, 8)
    assert gate.shape == (6, 32)

    args = fake_kernel.args
    assert args is not None
    assert args[10:14] == (6, 2, 2, 1)
    assert args[14:19] == (4, 1, 8, 32, 8)
    assert args[20:23] == (1, 1, 0)
    assert args[23] is True
    assert args[24] is True
    assert args[25:29] == (4, 2, True, 32)


@pytest.mark.parametrize(
    "has_gate,num_tokens,num_q_heads,num_kv_heads,head_size,mrope_section",
    [
        (False, 1, 1, 1, 4, [1, 1, 2]),
        (True, 3, 4, 2, 8, [1, 0, 1]),
        (False, 10, 8, 1, 16, [2, 3, 4]),
    ],
)
def test_fake_impl_edge_cases(
    monkeypatch, has_gate, num_tokens, num_q_heads, num_kv_heads, head_size, mrope_section
):
    """Test fake implementation with various tensor shapes and edge cases."""
    module = _load_module_with_stubs(monkeypatch)

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0
    total_size = q_size + gate_size + 2 * kv_size

    qkv = torch.empty(num_tokens, total_size)
    outputs = module.triton_split_qkv_rmsnorm_mrope_fake(
        qkv=qkv,
        q_weight=torch.empty(head_size),
        k_weight=torch.empty(head_size),
        cos_sin=torch.empty(3, num_tokens, head_size),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=1e-6,
        mrope_section=mrope_section,
        is_interleaved=False,
        has_gate=has_gate,
    )

    q, k, v, gate = outputs
    assert q.shape == (num_tokens, q_size)
    assert k.shape == (num_tokens, kv_size)
    assert v.shape == (num_tokens, kv_size)
    assert gate.shape == (num_tokens, gate_size)
    assert q.device.type == "cpu"
    assert q.dtype == qkv.dtype


def test_fake_impl_device_propagation(monkeypatch):
    """Verify fake impl respects the input device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    module = _load_module_with_stubs(monkeypatch)

    qkv = torch.empty(2, 16, device="cuda")
    outputs = module.triton_split_qkv_rmsnorm_mrope_fake(
        qkv=qkv,
        q_weight=torch.empty(4, device="cuda"),
        k_weight=torch.empty(4, device="cuda"),
        cos_sin=torch.empty(3, 2, 4, device="cuda"),
        num_q_heads=2,
        num_kv_heads=1,
        head_size=4,
        eps=1e-6,
        mrope_section=[1, 1, 2],
        is_interleaved=False,
        has_gate=False,
    )

    for t in outputs:
        assert t.device.type == "cuda"


@pytest.mark.parametrize(
    "core_num,num_tokens,expected_grid,expected_args_prefix",
    [
        # num_tokens < core_num: total_core = num_tokens, block_dim = num_tokens
        (8, 3, (3,), (3, 3, 1, 0)),
        # num_tokens == core_num: no tail cores, but num_tokens_each_tail_core is still computed
        (4, 4, (4,), (4, 4, 1, 1)),
        # num_tokens % core_num == 0, num_tokens > core_num: no front_core_num adjustment
        (4, 8, (4,), (8, 4, 2, 2)),
        # num_tokens % core_num != 0, num_tokens > core_num: has tail cores
        (4, 6, (4,), (6, 2, 2, 1)),
        # large skip: num_tokens >> core_num
        (4, 18, (4,), (18, 2, 5, 4)),
    ],
)
def test_wrapper_grid_configurations(
    monkeypatch, core_num, num_tokens, expected_grid, expected_args_prefix,
):
    """Test wrapper with different core_num and num_tokens combinations."""
    module = _load_module_with_stubs(monkeypatch)
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(module, "get_vectorcore_num", lambda: core_num)
    monkeypatch.setattr(module, "split_qkv_rmsnorm_mrope_kernel", fake_kernel)

    q_size = 16
    kv_size = 4
    qkv = torch.empty(num_tokens, q_size + 2 * kv_size)
    cos_sin = torch.empty(3, num_tokens, 8)

    module.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=torch.empty(8),
        k_weight=torch.empty(8),
        cos_sin=cos_sin,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=8,
        eps=1e-6,
        mrope_section=[1, 1, 2],
        is_interleaved=False,
    )

    assert fake_kernel.grid == expected_grid, f"grid={fake_kernel.grid}, expected={expected_grid}"
    args = fake_kernel.args
    assert args is not None
    assert args[10:14] == expected_args_prefix, (
        f"args[10:14]={args[10:14]}, expected={expected_args_prefix}"
    )


@pytest.mark.parametrize(
    "has_gate,has_bias,is_interleaved,mrope_section,rope_dim,head_size",
    [
        (False, False, False, [1, 1, 2], None, 8),
        (True, False, False, [2, 1, 1], None, 8),
        (False, True, False, [0, 0, 4], 4, 8),
        (True, True, True, [1, 0, 1], 4, 8),
        (False, False, True, [1, 1, 0], None, 8),
        (True, True, True, [2, 2, 2], 6, 8),
    ],
)
def test_wrapper_various_configurations(
    monkeypatch,
    has_gate,
    has_bias,
    is_interleaved,
    mrope_section,
    rope_dim,
    head_size,
):
    """Test wrapper with various configurations of bias, gate, interleaved, partial rope."""
    module = _load_module_with_stubs(monkeypatch)
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(module, "get_vectorcore_num", lambda: 4)
    monkeypatch.setattr(module, "split_qkv_rmsnorm_mrope_kernel", fake_kernel)

    num_tokens = 5
    num_q_heads = 2
    num_kv_heads = 1
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0
    total_size = q_size + gate_size + 2 * kv_size
    actual_rope_dim = head_size if rope_dim is None else rope_dim

    qkv = torch.empty(num_tokens, total_size)
    q_bias = torch.empty(head_size) if has_bias else None
    k_bias = torch.empty(head_size) if has_bias else None
    cos_sin = torch.empty(3, num_tokens, actual_rope_dim)

    q, k, v, gate = module.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=torch.empty(head_size),
        k_weight=torch.empty(head_size),
        cos_sin=cos_sin,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=1e-6,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
        q_bias=q_bias,
        k_bias=k_bias,
        has_gate=has_gate,
    )

    assert q.shape == (num_tokens, q_size)
    assert k.shape == (num_tokens, kv_size)
    assert v.shape == (num_tokens, kv_size)
    assert gate.shape == (num_tokens, gate_size)

    args = fake_kernel.args
    assert args is not None
    # Verify has_bias flag
    assert args[23] is has_bias
    # Verify is_interleaved flag
    assert args[24] is is_interleaved
    # Verify rope_dim
    assert args[25] == actual_rope_dim
    # Verify half_rope_dim
    assert args[26] == actual_rope_dim // 2
    # Verify IS_PARTIAL_ROPE
    assert args[27] is (actual_rope_dim != head_size)
    # Verify gate_size
    assert args[28] == gate_size


def test_wrapper_no_gate_no_bias_defaults(monkeypatch):
    """Test wrapper with all optional parameters at defaults (no bias, no gate)."""
    module = _load_module_with_stubs(monkeypatch)
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(module, "get_vectorcore_num", lambda: 2)
    monkeypatch.setattr(module, "split_qkv_rmsnorm_mrope_kernel", fake_kernel)

    qkv = torch.empty(3, 24)
    cos_sin = torch.empty(3, 3, 8)

    q, k, v, gate = module.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=torch.empty(8),
        k_weight=torch.empty(8),
        cos_sin=cos_sin,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=8,
        eps=1e-6,
        mrope_section=[1, 2, 1],
        is_interleaved=False,
    )

    assert q.shape == (3, 16)
    assert k.shape == (3, 8)
    assert v.shape == (3, 8)
    assert gate.shape == (3, 0)

    args = fake_kernel.args
    assert args is not None
    assert args[23] is False  # has_bias = False
    assert args[28] == 0       # gate_size = 0


def test_wrapper_single_token_single_head(monkeypatch):
    """Test wrapper with minimal configuration: 1 token, 1 head."""
    module = _load_module_with_stubs(monkeypatch)
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(module, "get_vectorcore_num", lambda: 4)
    monkeypatch.setattr(module, "split_qkv_rmsnorm_mrope_kernel", fake_kernel)

    qkv = torch.empty(1, 16)
    cos_sin = torch.empty(3, 1, 8)

    q, k, v, gate = module.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=torch.empty(8),
        k_weight=torch.empty(8),
        cos_sin=cos_sin,
        num_q_heads=1,
        num_kv_heads=1,
        head_size=8,
        eps=1e-6,
        mrope_section=[1, 1, 2],
        is_interleaved=False,
    )

    assert q.shape == (1, 8)
    assert k.shape == (1, 8)
    assert v.shape == (1, 8)
    assert gate.shape == (1, 0)
    # With 1 token and 4 cores: num_tokens < core_num, so total_core = 1
    assert fake_kernel.grid == (1,)


def test_module_exposes_public_api(monkeypatch):
    """Verify the loaded module exposes the expected public functions."""
    module = _load_module_with_stubs(monkeypatch)

    assert hasattr(module, "triton_split_qkv_rmsnorm_mrope")
    assert hasattr(module, "triton_split_qkv_rmsnorm_mrope_fake")
    assert hasattr(module, "split_qkv_rmsnorm_mrope_kernel")
    assert callable(module.triton_split_qkv_rmsnorm_mrope)
    assert callable(module.triton_split_qkv_rmsnorm_mrope_fake)

    # Verify the functions have the expected signatures
    import inspect

    sig = inspect.signature(module.triton_split_qkv_rmsnorm_mrope)
    param_names = list(sig.parameters.keys())
    assert "qkv" in param_names
    assert "q_weight" in param_names
    assert "k_weight" in param_names
    assert "cos_sin" in param_names
    assert "num_q_heads" in param_names
    assert "num_kv_heads" in param_names
    assert "has_gate" in param_names
    assert "mrope_section" in param_names
    assert "is_interleaved" in param_names


@pytest.mark.gpu
@pytest.mark.parametrize(
    "has_gate,has_bias,rope_dim,is_interleaved,mrope_section",
    [
        (False, False, None, False, [1, 1, 2]),
        (True, True, 4, True, [1, 1, 0]),
    ],
)
def test_triton_kernel_matches_reference_on_ascend(
    has_gate, has_bias, rope_dim, is_interleaved, mrope_section
):
    torch_npu = pytest.importorskip("torch_npu")
    if not getattr(torch, "npu", None) or not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")

    pytest.importorskip("vllm_ascend")
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.split_qkv_rmsnorm_mrope"
    )

    try:
        triton_utils = importlib.import_module("vllm_ascend.ops.triton.triton_utils")
        if hasattr(triton_utils, "init_device_properties_triton"):
            triton_utils.init_device_properties_triton()
    except Exception as exc:
        pytest.skip(f"Could not initialize Ascend Triton properties: {exc}")

    torch.manual_seed(0)
    device = torch.device("npu:0")
    dtype = torch.float16
    num_tokens = 7
    num_q_heads = 2
    num_kv_heads = 1
    head_size = 8
    actual_rope_dim = head_size if rope_dim is None else rope_dim
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0

    qkv = torch.randn(
        num_tokens,
        q_size + gate_size + 2 * kv_size,
        device=device,
        dtype=dtype,
    )
    q_weight = torch.randn(head_size, device=device, dtype=dtype)
    k_weight = torch.randn(head_size, device=device, dtype=dtype)
    q_bias = torch.randn(head_size, device=device, dtype=dtype) if has_bias else None
    k_bias = torch.randn(head_size, device=device, dtype=dtype) if has_bias else None
    cos_sin = torch.randn(3, num_tokens, actual_rope_dim, device=device, dtype=dtype)

    expected = _reference_split_qkv_rmsnorm_mrope(
        qkv=qkv.cpu(),
        q_weight=q_weight.cpu(),
        k_weight=k_weight.cpu(),
        cos_sin=cos_sin.cpu(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=1e-6,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
        q_bias=q_bias.cpu() if q_bias is not None else None,
        k_bias=k_bias.cpu() if k_bias is not None else None,
        has_gate=has_gate,
    )

    actual = module.triton_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=1e-6,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
        q_bias=q_bias,
        k_bias=k_bias,
        has_gate=has_gate,
    )
    torch_npu.npu.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.cpu(),
            expected_tensor,
            rtol=2e-2,
            atol=2e-2,
        )
