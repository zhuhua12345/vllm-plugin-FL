#!/usr/bin/env python
"""Standalone NPU test for split_qkv_rmsnorm_mrope.
Bypasses the slow vllm_fl import chain by loading the target module
directly via importlib.util.
"""
import importlib.util
import os
import sys
import types
import uuid
from pathlib import Path

# Triton kernels need this to access globals like extract_slice
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = (
    PROJECT_ROOT / "vllm_fl" / "dispatch" / "backends" / "vendor"
    / "ascend" / "impl" / "split_qkv_rmsnorm_mrope.py"
)


def _load_target_module():
    vllm_mod = types.ModuleType("vllm")
    tu_mod = types.ModuleType("vllm.triton_utils")
    u_mod = types.ModuleType("vllm.utils")
    tou_mod = types.ModuleType("vllm.utils.torch_utils")

    # Use the REAL triton module for JIT compilation on NPU
    import triton
    tu_mod.tl = triton.language
    tu_mod.triton = triton
    tu_mod.HAS_TRITON = True
    tou_mod.direct_register_custom_op = lambda **kw: None
    vllm_mod.triton_utils = tu_mod
    vllm_mod.utils = u_mod
    u_mod.torch_utils = tou_mod
    for n, m in {"vllm": vllm_mod, "vllm.triton_utils": tu_mod,
                 "vllm.utils": u_mod, "vllm.utils.torch_utils": tou_mod}.items():
        sys.modules[n] = m

    # Stub vllm_ascend modules so the target module can import
    # extract_slice/insert_slice/get_vectorcore_num without triggering
    # the full vllm_ascend.ops import chain (which needs real vllm).
    ascend_tu = types.ModuleType("vllm_ascend.ops.triton.triton_utils")
    ascend_tu.extract_slice = lambda x, *a, **kw: x
    ascend_tu.insert_slice = lambda t, s, *a, **kw: t
    ascend_tu.get_vectorcore_num = lambda: 8
    for n in ["vllm_ascend", "vllm_ascend.ops", "vllm_ascend.ops.triton"]:
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)
    sys.modules["vllm_ascend.ops.triton.triton_utils"] = ascend_tu

    name = "_test_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, SOURCE_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ref(qkv, qw, kw, cs, nqh, nkvh, hs, eps, ms, ii,
         rd=None, qb=None, kb=None, hg=False):
    qs = nqh * hs
    kvs = nkvh * hs
    gs = qs if hg else 0
    nt = qkv.shape[0]
    rd = hs if rd is None else rd
    hrd = rd // 2

    if hg:
        qg = qkv[:, :qs+gs].float().reshape(nt, nqh, hs*2)
        q = qg[:, :, :hs]
        gate = qg[:, :, hs:].reshape(nt, qs)
    else:
        q = qkv[:, :qs].float().reshape(nt, nqh, hs)
        gate = torch.empty(nt, 0, device=qkv.device, dtype=qkv.dtype)

    ks = qs + gs
    vs = ks + kvs
    k = qkv[:, ks:vs].float().reshape(nt, nkvh, hs)
    v = qkv[:, vs:vs+kvs]

    def rn(x, w, b):
        var = (x*x).sum(-1, keepdim=True) / hs
        y = x * torch.rsqrt(var + eps) * w.float().view(1, 1, hs)
        if b is not None:
            y = y + b.float().view(1, 1, hs)
        return y

    q = rn(q, qw, qb)
    k = rn(k, kw, kb)
    cs = cs.reshape(3, nt, rd)
    off = torch.arange(hrd, device=qkv.device)
    if ii:
        hm = ((off % 3) == 1) & (off <= 3 * ms[1])
        wm = ((off % 3) == 2) & (off <= 3 * ms[2])
        tm = ~(hm | wm)
    else:
        tm = off < ms[0]
        hm = (ms[0]-1 < off) & (off < ms[0]+ms[1])
        wm = (ms[0]+ms[1]-1 < off) & (off < sum(ms))

    def mr(x):
        ch = torch.zeros(nt, hrd, device=qkv.device)
        sh = torch.zeros(nt, hrd, device=qkv.device)
        for ax, mk in enumerate((tm, hm, wm)):
            ch[:, mk] = cs[ax, :, :hrd][:, mk].float()
            sh[:, mk] = cs[ax, :, hrd:rd][:, mk].float()
        c = ch.repeat(1, 2).unsqueeze(1)
        s = sh.repeat(1, 2).unsqueeze(1)
        xr = x[:, :, :rd]
        x1 = xr[:, :, :hrd]
        x2 = xr[:, :, hrd:rd]
        roped = torch.cat((-x2, x1), -1) * s + xr * c
        return roped if rd == hs else torch.cat((roped, x[:, :, rd:]), -1)

    q = mr(q).reshape(nt, qs).to(qkv.dtype)
    k = mr(k).reshape(nt, kvs).to(qkv.dtype)
    return q, k, v, gate.to(qkv.dtype)


def main():
    print("=" * 60, flush=True)
    print("NPU Test: split_qkv_rmsnorm_mrope", flush=True)
    print("=" * 60, flush=True)

    if not getattr(torch, "npu", None) or not torch.npu.is_available():
        print("SKIP: NPU not available", flush=True)
        return

    dc = torch.npu.device_count()
    # Try devices in order: 1, 4, 0, 2, 3, 5, 6, 7
    preferred = [int(x) for x in os.environ.get("FL_TEST_DEVICE_ID", "1,4").split(",")]
    tried = []
    dev = None
    for did in preferred + [i for i in range(dc) if i not in preferred]:
        if did >= dc:
            continue
        try:
            test_dev = torch.device(f"npu:{did}")
            # Quick validation: create a small tensor
            _ = torch.zeros(1, device=test_dev)
            dev = test_dev
            print(f"NPU count={dc}, using npu:{did}", flush=True)
            break
        except Exception as e:
            tried.append(did)
            print(f"npu:{did} unavailable: {e}", flush=True)
    if dev is None:
        print("FAIL: No usable NPU device found", flush=True)
        return

    print("Loading module...", flush=True)
    mod = _load_target_module()
    print("OK.", flush=True)

    try:
        tu = __import__("vllm_ascend.ops.triton.triton_utils", fromlist=[""])
        if hasattr(tu, "init_device_properties_triton"):
            tu.init_device_properties_triton()
    except Exception as e:
        print(f"Note: Triton init: {e}", flush=True)

    cases = [
        (False, False, None, False, [1, 1, 2], "case1: no_gate, no_bias, full rope"),
        (True, True, 4, True, [1, 1, 0], "case2: gate+bias, partial rope, interleaved"),
    ]
    ok = True
    for hg, hb, rd, ii, ms, desc in cases:
        print(f"\n--- {desc} ---", flush=True)
        torch.manual_seed(0)
        nt, nqh, nkvh, hs = 7, 2, 1, 8
        ard = hs if rd is None else rd
        qs = nqh * hs
        kvs = nkvh * hs
        gs = qs if hg else 0

        qkv = torch.randn(nt, qs+gs+2*kvs, device=dev, dtype=torch.float16)
        qw = torch.randn(hs, device=dev, dtype=torch.float16)
        kw = torch.randn(hs, device=dev, dtype=torch.float16)
        qb = torch.randn(hs, device=dev, dtype=torch.float16) if hb else None
        kb = torch.randn(hs, device=dev, dtype=torch.float16) if hb else None
        cs = torch.randn(3, nt, ard, device=dev, dtype=torch.float16)

        exp = _ref(qkv.cpu(), qw.cpu(), kw.cpu(), cs.cpu(), nqh, nkvh, hs,
                   1e-6, ms, ii, rd, qb.cpu() if qb is not None else None,
                   kb.cpu() if kb is not None else None, hg)

        print("  Running kernel...", flush=True)
        try:
            act = mod.triton_split_qkv_rmsnorm_mrope(
                qkv, qw, kw, cs, nqh, nkvh, hs, 1e-6, ms, ii,
                rope_dim=rd, q_bias=qb, k_bias=kb, has_gate=hg)
            torch.npu.synchronize()
            print("  Done.", flush=True)
        except Exception as e:
            print(f"  FAIL: {e}", flush=True)
            import traceback
            traceback.print_exc()
            ok = False
            continue

        for name, a, e in zip(["q","k","v","gate"], act, exp):
            try:
                torch.testing.assert_close(a.cpu(), e, rtol=2e-2, atol=2e-2)
                print(f"  {name}: PASS  shape={a.shape}", flush=True)
            except AssertionError as ex:
                print(f"  {name}: FAIL  max_err={(a.cpu()-e).abs().max().item():.6f}", flush=True)
                ok = False

    print("\n" + "=" * 60, flush=True)
    print("ALL TESTS PASSED ✅" if ok else "SOME TESTS FAILED ❌", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
