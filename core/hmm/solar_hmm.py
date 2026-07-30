"""Load and validate the frozen eight-state Stage 1 solar HMM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class HMMParameters:
    transition: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    initial: np.ndarray
    source: Path
    provenance: str

    def validate(self, expected_states: int) -> None:
        if self.transition.shape != (expected_states, expected_states):
            raise ValueError(f"invalid transition shape {self.transition.shape}")
        for name, value in (
            ("mean", self.mean),
            ("variance", self.variance),
            ("initial", self.initial),
        ):
            if value.shape != (expected_states,):
                raise ValueError(f"invalid {name} shape {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.transition < 0.0):
            raise ValueError("transition matrix contains negative probabilities")
        if not np.allclose(self.transition.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("transition rows do not sum to one")
        if np.any(self.variance < 0.0):
            raise ValueError("variance contains negative values")
        if np.any(self.initial < 0.0) or not np.isclose(self.initial.sum(), 1.0):
            raise ValueError("initial distribution is invalid")


def load_solar_hmm(path: str | Path) -> HMMParameters:
    path = Path(path).resolve()
    data = loadmat(path)
    required = ("hmm_A", "hmm_mu", "hmm_sigma2", "hmm_pi0")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"solar HMM artifact missing keys: {missing}")
    params = HMMParameters(
        transition=np.asarray(data["hmm_A"], dtype=np.float64),
        mean=np.asarray(data["hmm_mu"], dtype=np.float64).reshape(-1),
        variance=np.asarray(data["hmm_sigma2"], dtype=np.float64).reshape(-1),
        initial=np.asarray(data["hmm_pi0"], dtype=np.float64).reshape(-1),
        source=path,
        provenance="trained_stage1_baum_welch",
    )
    params.validate(expected_states=8)
    return params

