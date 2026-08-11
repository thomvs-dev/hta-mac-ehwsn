"""Exact runtime fingerprints for same-build preflight/training contracts."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata

import numpy as np
import torch


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_runtime_fingerprint() -> dict:
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    torch_config = torch.__config__.show()
    payload = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_basename": platform.python_implementation().lower(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": (
            torch.backends.cudnn.version() if cuda_available else None
        ),
        "torch_config_sha256": hashlib.sha256(torch_config.encode()).hexdigest(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": devices,
        "numpy_version": np.__version__,
        "scipy_version": _package_version("scipy"),
        "torch_geometric_version": _package_version("torch-geometric"),
        "gymnasium_version": _package_version("gymnasium"),
    }
    return payload


def fingerprint_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_runtime_contract(*, require_cuda: bool) -> dict:
    fingerprint = capture_runtime_fingerprint()
    if require_cuda and not fingerprint["cuda_available"]:
        raise RuntimeError("runtime contract requires CUDA")
    return {
        "schema_version": 1,
        "status": "frozen_same_runtime_contract",
        "match_mode": "exact_fingerprint",
        "require_cuda": bool(require_cuda),
        "fingerprint": fingerprint,
        "fingerprint_sha256": fingerprint_sha256(fingerprint),
        "cross_platform_tolerance_validated": False,
    }


def validate_runtime_contract(contract: dict, current: dict | None = None) -> dict:
    if contract.get("status") != "frozen_same_runtime_contract":
        raise ValueError("runtime contract is not frozen")
    if contract.get("match_mode") != "exact_fingerprint":
        raise ValueError("only exact same-runtime matching is supported")
    if contract.get("cross_platform_tolerance_validated") is not False:
        raise ValueError("this contract must not claim cross-platform validation")
    expected = contract["fingerprint"]
    expected_sha = fingerprint_sha256(expected)
    if contract.get("fingerprint_sha256") != expected_sha:
        raise ValueError("runtime contract fingerprint hash mismatch")
    observed = capture_runtime_fingerprint() if current is None else current
    observed_sha = fingerprint_sha256(observed)
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in sorted(set(expected) | set(observed))
        if expected.get(key) != observed.get(key)
    }
    if contract.get("require_cuda") and not observed.get("cuda_available"):
        mismatches["cuda_required"] = {"expected": True, "observed": False}
    return {
        "pass": not mismatches and observed_sha == expected_sha,
        "expected_fingerprint_sha256": expected_sha,
        "observed_fingerprint_sha256": observed_sha,
        "mismatches": mismatches,
        "match_mode": "exact_fingerprint",
    }
