import json
from pathlib import Path

from experiments.run_step4_final_confirmation import load_contract


ROOT = Path(__file__).resolve().parents[1]


def test_final_confirmation_contract_is_frozen_and_disjoint():
    path = ROOT / "config" / "step4_final_confirmation_v1.json"
    contract = load_contract(path)
    assert contract["confirmation_seeds"] == list(range(3900, 3920))
    assert set(contract["confirmation_seeds"]).isdisjoint(contract["prior_development_seeds"])
    assert contract["confirmation_seeds_opened"] is False
    assert contract["decision_rules"]["no_retuning_after_open"] is True
    assert contract["decision_rules"]["failed_claims_are_reported_not_repaired"] is True


def test_final_confirmation_freezes_inference_unit_and_baselines():
    contract = json.loads((ROOT / "config" / "step4_final_confirmation_v1.json").read_text())
    assert contract["inference"]["unit"] == "seed_mean_across_target_ranks"
    assert contract["policies"] == ["hta_mac", "energy_proportional", "online_primal_dual"]
    assert contract["horizon"] == 3000
    assert contract["predeclared_claims"]["reference_noninferiority_vs_primal_dual"] == {
        "delivery_margin_absolute": 0.01,
        "rmst_margin_rounds": 6.0,
        "packets_per_j_margin_fraction": 0.05,
    }
