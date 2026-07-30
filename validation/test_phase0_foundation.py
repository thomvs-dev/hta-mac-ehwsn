from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core.configuration import load_simple_yaml
from core.energy import RadioModel
from core.hmm import load_solar_hmm, load_thermal_auxiliary


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "final_repo"


def test_base_config_and_radio_model() -> None:
    config = load_simple_yaml(ROOT / "config" / "base.yaml")
    radio = config["radio"]
    model = RadioModel(**radio)
    assert math.isclose(
        model.d0_m,
        math.sqrt(
            radio["eps_fs_j_per_bit_m2"] / radio["eps_mp_j_per_bit_m4"]
        ),
        abs_tol=1e-12,
    )
    bits = config["network"]["packet_bits"]
    assert math.isclose(model.rx(bits), bits * radio["e_elec_j_per_bit"])
    assert model.tx(bits, 50.0) > model.rx(bits)


def test_frozen_hmm_artifacts() -> None:
    solar = load_solar_hmm(UPSTREAM / "outputs" / "stage1_params.mat")
    thermal = load_thermal_auxiliary(
        ROOT / "core" / "hmm" / "thermal_auxiliary_params.npz"
    )
    assert solar.provenance == "trained_stage1_baum_welch"
    assert thermal.provenance == "synthetic_auxiliary_from_heart_ch_defaults"
    assert np.allclose(solar.transition.sum(axis=1), 1.0)
    assert np.allclose(thermal.transition.sum(axis=1), 1.0)
    assert solar.mean.shape == (8,)
    assert thermal.mean.shape == (4,)

