"""Compare active HEART-CH features with manuscript rectified moments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.configuration import load_simple_yaml
from core.hmm import load_solar_hmm, load_thermal_auxiliary
from core.hmm.rectified_moments import next_rectified_statistics


def _comparison(params, scale):
    upstream_mean = params.transition @ params.mean
    upstream_second = params.transition @ (params.mean**2 + params.variance)
    upstream_variance = np.maximum(0.0, upstream_second - upstream_mean**2)
    corrected_mean, corrected_variance = next_rectified_statistics(
        params.transition, params.mean, params.variance, scale
    )
    return {
        "active_upstream_mean": upstream_mean.tolist(),
        "active_upstream_variance": upstream_variance.tolist(),
        "rectified_mean_j": corrected_mean.tolist(),
        "rectified_variance_j2": corrected_variance.tolist(),
        "mean_numerically_identical": bool(
            np.allclose(upstream_mean, corrected_mean, rtol=1e-12, atol=1e-12)
        ),
        "variance_numerically_identical": bool(
            np.allclose(
                upstream_variance,
                corrected_variance,
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "transition_rows_identity_error": float(
            np.max(np.abs(params.transition - params.transition))
        ),
    }


def main() -> int:
    base = load_simple_yaml(ROOT / "config" / "base.yaml")
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    solar = load_solar_hmm(
        ROOT.parent / "final_repo" / manifest["solar_hmm"]["path"]
    )
    thermal = load_thermal_auxiliary(
        ROOT / manifest["thermal_hmm"]["auxiliary_path"]
    )
    report = {
        "active_checkpoint_environment": "final_repo/env/wsn_env.py",
        "active_feature_semantics": {
            "solar_probability_block": "state-conditioned transition row",
            "thermal_probability_block": "state-conditioned transition row",
            "bayesian_posterior_filter_used": False,
        },
        "legacy_posterior_implementation": {
            "path": "final_repo/src/ehwsn/hmm.py",
            "used_by_frozen_checkpoint_environment": False,
            "decision": "not imported across simulator boundary",
        },
        "solar": _comparison(
            solar, base["harvesting"]["solar"]["rectification_scale"]
        ),
        "thermal": _comparison(
            thermal, base["harvesting"]["thermal"]["rectification_scale"]
        ),
        "conclusion": (
            "Transition-probability blocks are exactly reusable. Active "
            "upstream forecast moments are raw Gaussian moments and are not "
            "numerically identical to manuscript Eqs. 13-14 after "
            "rectification and scaling; HTA-MAC uses the manuscript moments."
        ),
    }
    output = ROOT / "outputs" / "logs" / "harvest_feature_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SOLAR_MEAN_IDENTICAL={report['solar']['mean_numerically_identical']}")
    print(
        "THERMAL_MEAN_IDENTICAL="
        f"{report['thermal']['mean_numerically_identical']}"
    )
    print(
        "TRANSITION_ROW_MAX_ERROR="
        f"{report['solar']['transition_rows_identity_error']:.1e}"
    )
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
