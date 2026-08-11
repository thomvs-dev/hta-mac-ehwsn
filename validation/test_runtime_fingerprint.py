import copy

import pytest

from core.runtime_fingerprint import (
    fingerprint_sha256,
    validate_runtime_contract,
)


def contract(fingerprint):
    return {
        "status": "frozen_same_runtime_contract",
        "match_mode": "exact_fingerprint",
        "require_cuda": True,
        "fingerprint": fingerprint,
        "fingerprint_sha256": fingerprint_sha256(fingerprint),
        "cross_platform_tolerance_validated": False,
    }


def test_exact_runtime_contract_passes_only_identical_fingerprint():
    fingerprint = {"schema_version": 1, "cuda_available": True, "torch_version": "x"}
    result = validate_runtime_contract(contract(fingerprint), current=copy.deepcopy(fingerprint))
    assert result["pass"] is True
    assert result["mismatches"] == {}


def test_exact_runtime_contract_reports_build_drift():
    expected = {"schema_version": 1, "cuda_available": True, "torch_version": "x"}
    observed = {"schema_version": 1, "cuda_available": True, "torch_version": "y"}
    result = validate_runtime_contract(contract(expected), current=observed)
    assert result["pass"] is False
    assert result["mismatches"]["torch_version"] == {"expected": "x", "observed": "y"}


def test_contract_rejects_post_hoc_cross_platform_claim():
    fingerprint = {"schema_version": 1, "cuda_available": True}
    payload = contract(fingerprint)
    payload["cross_platform_tolerance_validated"] = True
    with pytest.raises(ValueError, match="must not claim"):
        validate_runtime_contract(payload, current=fingerprint)
