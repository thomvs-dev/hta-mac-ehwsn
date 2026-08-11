from experiments.audit_phase3_fnd_node_identity import summarize


def _record(seed, policy, nodes):
    return {
        "seed": seed,
        "policy": policy,
        "newly_dead_nodes": [{"node_id": node} for node in nodes],
    }


def test_summary_distinguishes_same_node_from_policy_specific_deaths():
    records = [
        _record(1, "hta_mac", [4]),
        _record(1, "energy_proportional", [4]),
        _record(2, "hta_mac", [7]),
        _record(2, "energy_proportional", [9]),
    ]
    result = summarize(records, ["hta_mac", "energy_proportional"])
    assert result["paired_trial_count"] == 2
    assert result["paired_same_node_set_count"] == 1
    assert result["paired_any_node_overlap_count"] == 1
    assert result["same_node_set_fraction"] == 0.5
    assert result["by_policy"]["hta_mac"]["unique_fnd_node_ids"] == [4, 7]


def test_summary_handles_simultaneous_first_deaths():
    records = [
        _record(1, "hta_mac", [2, 3]),
        _record(1, "energy_proportional", [3, 2]),
    ]
    result = summarize(records, ["hta_mac", "energy_proportional"])
    assert result["paired_same_node_set_count"] == 1
    assert result["paired_any_node_overlap_count"] == 1
