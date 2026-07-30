"""Load HEART-CH's fixed four-state thermal auxiliary model.

This adapter deliberately calls the model an auxiliary, not a trained HMM.
The current upstream repository has no thermal training trace or fitted export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .solar_hmm import HMMParameters


def load_thermal_auxiliary(path: str | Path) -> HMMParameters:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        required = ("thermal_A", "thermal_mu", "thermal_sigma2", "thermal_pi0")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"thermal auxiliary artifact missing keys: {missing}")
        params = HMMParameters(
            transition=np.asarray(data["thermal_A"], dtype=np.float64),
            mean=np.asarray(data["thermal_mu"], dtype=np.float64).reshape(-1),
            variance=np.asarray(data["thermal_sigma2"], dtype=np.float64).reshape(-1),
            initial=np.asarray(data["thermal_pi0"], dtype=np.float64).reshape(-1),
            source=path,
            provenance="synthetic_auxiliary_from_heart_ch_defaults",
        )
    params.validate(expected_states=4)
    return params

