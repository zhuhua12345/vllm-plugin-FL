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
from unittest.mock import Mock

import pytest
import torch


OP_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "ascend"
    / "impl"
    / "split_qkv_rmsnorm_mrope.py"
)


def _load_op_module(monkeypatch: pytest.MonkeyPatch):
    """Import the op module with small stubs for optional vLLM dependencies."""

    vllm = types.ModuleType("vllm")
    vllm_triton_utils = types.ModuleType("vllm.triton_utils")
    vllm_utils = types.ModuleType("vllm.utils")
    vllm_torch_utils = types.ModuleType("vllm.utils.torch_utils")

    class _FakeTriton:
        @staticmethod
        def jit(*args, **kwargs):
            def decorate(fn):
                return fn

            return decorate

    vllm_triton_utils.tl = types.SimpleNamespace(constexpr=object())
    vllm_triton_utils.triton = _FakeTriton()
    vllm_torch_utils.direct_register_custom_op = Mock()

    vllm.triton_utils = vllm_triton_utils
    vllm.utils = vllm_utils
    vllm_utils.torch_utils = vllm_torch_utils

    vllm_ascend = types.ModuleType("vllm_ascend")
    vllm_ascend_ops = types.ModuleType("vllm_ascend.ops")
    vllm_ascend_triton = types.ModuleType("vllm_ascend.ops.triton")
    ascend_triton_utils = types.ModuleType("vllm_ascend.ops.triton.triton_utils")
    ascend_triton_utils.extract_slice = Mock()
    ascend_triton_utils.insert_slice = Mock()
    ascend_triton_utils.get_vectorcore_num = Mock(return_value=1)

    modules = {
        "vllm": vllm,
        "vllm.triton_utils": vllm_triton_utils,
        "vllm.utils": vllm_utils,
        "vllm.utils.torch_utils": vllm_torch_utils,
        "vllm_ascend": vllm_ascend,
        "vllm_ascend.ops": vllm_ascend_ops,
        "vllm_ascend.ops.triton": vllm_ascend_triton,
        "vllm_ascend.ops.triton.triton_utils": ascend_triton_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = f"_split_qkv_rmsnorm_mrope_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, OP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingKernel:
    """Stand-in for a Triton kernel launched as ``kernel[(grid,)](*args)``."""

    def __init__(self):
        self.grid = None
        self.args = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args):
            self.args = args

        return launch


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
        q_gate = qkv[:, : q_size + gate_size].float()
        q_gate = q_gate.reshape(num_tokens, num_q_heads, head_size * 2)
        q = q_gate[:, :, :head_size]
        gate = q_gate[:, :, head_size:].reshape(num_tokens, q_size)
    else:
        q = qkv[:, :q_size].float().reshape(num_tokens, num_q_heads, head_size)
        gate = torch.empty(num_tokens, 0, dtype=qkv.dtype)

    k_start = q_size + gate_size
    v_start = k_start + kv_size
    k = qkv[:, k_start:v_start].float().reshape(
        num_tokens, num_kv_heads, head_size
    )
    v = qkv[:, v_start : v_start + kv_size]

    def rms_norm(x, weight, bias):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(variance + eps)
        y = y * weight.float().view(1, 1, head_size)
        if bias is not None:
            y = y + bias.float().view(1, 1, head_size)
        return y

    q = rms_norm(q, q_weight, q_bias)
    k = rms_norm(k, k_weight, k_bias)

    cos_sin = cos_sin.reshape(3, num_tokens, rope_dim)
    offsets = torch.arange(half_rope_dim)
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

    def apply_mrope(x):
        cos_half = torch.zeros(num_tokens, half_rope_dim)
        sin_half = torch.zeros(num_tokens, half_rope_dim)
        for axis, mask in enumerate((t_mask, h_mask, w_mask)):
            cos_half[:, mask] = cos_sin[axis, :, :half_rope_dim][:, mask].float()
            sin_half[:, mask] = cos_sin[axis, :, half_rope_dim:rope_dim][
                :, mask
            ].float()

        cos = cos_half.repeat(1, 2).unsqueeze(1)
        sin = sin_half.repeat(1, 2).unsqueeze(1)
        rope_part = x[:, :, :rope_dim]
        x1 = rope_part[:, :, :half_rope_dim]
        x2 = rope_part[:, :, half_rope_dim:rope_dim]
        rotated = torch.cat((-x2, x1), dim=-1)
        roped = rope_part * cos + rotated * sin
        if rope_dim == head_size:
            return roped
        return torch.cat((roped, x[:, :, rope_dim:]), dim=-1)

    q = apply_mrope(q).reshape(num_tokens, q_size).to(qkv.dtype)
    k = apply_mrope(k).reshape(num_tokens, kv_size).to(qkv.dtype)
    return q, k, v, gate.to(qkv.dtype)


class TestSplitQKVRMSNormMRopeFake:
    """Shape tests for the fake implementation used by graph tracing."""

    @pytest.fixture
    def op_module(self, monkeypatch):
        return _load_op_module(monkeypatch)

    @pytest.mark.parametrize("has_gate", [False, True])
    def test_fake_impl_output_shapes(self, op_module, has_gate):
        qkv = torch.empty(5, 80)

        q, k, v, gate = op_module.triton_split_qkv_rmsnorm_mrope_fake(
            qkv=qkv,
            q_weight=torch.empty(8),
            k_weight=torch.empty(8),
            cos_sin=torch.empty(3, 5, 8),
            num_q_heads=4,
            num_kv_heads=1,
            head_size=8,
            eps=1e-6,
            mrope_section=[1, 1, 2],
            is_interleaved=False,
            has_gate=has_gate,
        )

        assert q.shape == (5, 32)
        assert k.shape == (5, 8)
        assert v.shape == (5, 8)
        assert gate.shape == (5, 32 if has_gate else 0)
        assert q.dtype == qkv.dtype


class TestSplitQKVRMSNormMRopeWrapper:
    """Tests for Python wrapper shape allocation and kernel launch args."""

    @pytest.fixture
    def op_module(self, monkeypatch):
        module = _load_op_module(monkeypatch)
        monkeypatch.setattr(module, "get_vectorcore_num", Mock(return_value=4))
        monkeypatch.setattr(
            module, "split_qkv_rmsnorm_mrope_kernel", _RecordingKernel()
        )
        return module

    def test_wrapper_allocates_outputs(self, op_module):
        qkv = torch.empty(6, 80)

        q, k, v, gate = op_module.triton_split_qkv_rmsnorm_mrope(
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
            q_bias=torch.empty(8),
            k_bias=torch.empty(8),
            has_gate=True,
        )

        assert q.shape == (6, 32)
        assert k.shape == (6, 8)
        assert v.shape == (6, 8)
        assert gate.shape == (6, 32)

    def test_wrapper_launches_kernel_with_expected_args(self, op_module):
        kernel = op_module.split_qkv_rmsnorm_mrope_kernel
        qkv = torch.empty(6, 80)

        op_module.triton_split_qkv_rmsnorm_mrope(
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
            q_bias=torch.empty(8),
            k_bias=torch.empty(8),
            has_gate=True,
        )

        assert kernel.grid == (4,)
        assert kernel.args is not None
        assert kernel.args[10:14] == (6, 2, 2, 1)
        assert kernel.args[14:19] == (4, 1, 8, 32, 8)
        assert kernel.args[20:23] == (1, 1, 0)
        assert kernel.args[23] is True
        assert kernel.args[24] is True
        assert kernel.args[25:29] == (4, 2, True, 32)

    @pytest.mark.parametrize(
        "num_tokens,expected_grid,expected_split",
        [
            (3, (3,), (3, 3, 1, 0)),
            (4, (4,), (4, 4, 1, 1)),
            (6, (4,), (6, 2, 2, 1)),
            (8, (4,), (8, 4, 2, 2)),
        ],
    )
    def test_wrapper_token_split(self, op_module, num_tokens, expected_grid, expected_split):
        kernel = op_module.split_qkv_rmsnorm_mrope_kernel

        op_module.triton_split_qkv_rmsnorm_mrope(
            qkv=torch.empty(num_tokens, 32),
            q_weight=torch.empty(8),
            k_weight=torch.empty(8),
            cos_sin=torch.empty(3, num_tokens, 8),
            num_q_heads=2,
            num_kv_heads=1,
            head_size=8,
            eps=1e-6,
            mrope_section=[1, 1, 2],
            is_interleaved=False,
        )

        assert kernel.grid == expected_grid
        assert kernel.args[10:14] == expected_split


def _load_module_for_npu(monkeypatch):
    """Load the target module for NPU testing.

    Two strategies (tried in order):
    1. Normal ``import`` — works on FlagOS where vllm / vllm-ascend are
       installed as real packages.
    2. Direct file load with stubs — works on this dev machine where
       ``vllm_fl`` conflicts with ``vllm_ascend``.
    """
    from pathlib import Path

    _target = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl" / "dispatch" / "backends" / "vendor"
        / "ascend" / "impl" / "split_qkv_rmsnorm_mrope.py"
    )

    # ---- Strategy 1: normal import (FlagOS) ----
    try:
        return importlib.import_module(
            "vllm_fl.dispatch.backends.vendor.ascend.impl.split_qkv_rmsnorm_mrope"
        )
    except Exception:
        pass

    # ---- Strategy 2: direct file load with stubs (dev machine) ----
    import triton

    vllm_mod = types.ModuleType("vllm")
    tu_mod = types.ModuleType("vllm.triton_utils")
    u_mod = types.ModuleType("vllm.utils")
    tou_mod = types.ModuleType("vllm.utils.torch_utils")

    tu_mod.tl = triton.language
    tu_mod.triton = triton
    tu_mod.HAS_TRITON = True
    tou_mod.direct_register_custom_op = lambda **kw: None
    vllm_mod.triton_utils = tu_mod
    vllm_mod.utils = u_mod
    u_mod.torch_utils = tou_mod

    for name, mod in {
        "vllm": vllm_mod,
        "vllm.triton_utils": tu_mod,
        "vllm.utils": u_mod,
        "vllm.utils.torch_utils": tou_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Load real vllm_ascend triton_utils — try installed package first
    try:
        import vllm_ascend.ops.triton.triton_utils as ascend_tu  # type: ignore[import-unused]
    except ImportError:
        _fallback = Path("/data2/wfp/FlagOS/vllm-ascend/vllm_ascend/ops/triton/triton_utils.py")
        if _fallback.exists():
            tu_spec = importlib.util.spec_from_file_location(
                "vllm_ascend.ops.triton.triton_utils", str(_fallback)
            )
            ascend_tu = importlib.util.module_from_spec(tu_spec)
        else:
            raise

    for pkg in ["vllm_ascend", "vllm_ascend.ops", "vllm_ascend.ops.triton"]:
        if pkg not in sys.modules:
            monkeypatch.setitem(sys.modules, pkg, types.ModuleType(pkg))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.triton.triton_utils", ascend_tu
    )
    if hasattr(ascend_tu, "init_device_properties_triton"):
        try:
            ascend_tu.init_device_properties_triton()
        except Exception:
            pass

    spec = importlib.util.spec_from_file_location(
        "_split_qkv_npu_" + uuid.uuid4().hex, str(_target)
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.gpu
class TestSplitQKVRMSNormMRopeAscend:
    """Numerical test for the real Ascend Triton kernel."""

    def test_kernel_matches_reference(self, monkeypatch):
        torch_npu = pytest.importorskip("torch_npu")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            pytest.skip("Ascend NPU is not available")

        module = _load_module_for_npu(monkeypatch)

        torch.manual_seed(0)
        device = torch.device("npu:0")
        dtype = torch.float16
        num_tokens = 7
        num_q_heads = 2
        num_kv_heads = 1
        head_size = 8
        rope_dim = 4
        mrope_section = [1, 1, 0]
        q_size = num_q_heads * head_size
        kv_size = num_kv_heads * head_size
        gate_size = q_size

        qkv = torch.randn(
            num_tokens,
            q_size + gate_size + 2 * kv_size,
            device=device,
            dtype=dtype,
        )
        q_weight = torch.randn(head_size, device=device, dtype=dtype)
        k_weight = torch.randn(head_size, device=device, dtype=dtype)
        q_bias = torch.randn(head_size, device=device, dtype=dtype)
        k_bias = torch.randn(head_size, device=device, dtype=dtype)
        cos_sin = torch.randn(3, num_tokens, rope_dim, device=device, dtype=dtype)

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
            is_interleaved=True,
            rope_dim=rope_dim,
            q_bias=q_bias.cpu(),
            k_bias=k_bias.cpu(),
            has_gate=True,
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
            is_interleaved=True,
            rope_dim=rope_dim,
            q_bias=q_bias,
            k_bias=k_bias,
            has_gate=True,
        )
        torch_npu.npu.synchronize()

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(
                actual_tensor.cpu(),
                expected_tensor,
                rtol=2e-2,
                atol=2e-2,
            )
