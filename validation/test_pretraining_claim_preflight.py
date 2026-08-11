from validation.pretraining_claim_preflight import evaluate_episode_gates


def row(*, deaths, t_fnd, steps, delivered, demand):
    return {
        "steps": steps,
        "t_fnd": t_fnd,
        "raw_terms": {"deaths": deaths},
        "qos_constraint": {
            "cumulative_counts": {"delivered": delivered, "demand": demand},
            "metric_contract": {
                "ratio_scope": "episode_cumulative_target_backlog_service",
                "demand_field": "target_packets_offered",
                "fairness_metric_name": "target_cluster_service_fairness",
            },
        },
    }


def test_all_episode_gates_pass_for_representative_claim_probe():
    gates = evaluate_episode_gates(
        [row(deaths=-1, t_fnd=1100, steps=1200, delivered=800, demand=1000)],
        reward_term="deaths",
        headline_event_field="t_fnd",
        minimum_nonzero_term_records=1,
        minimum_event_records=1,
        minimum_training_horizon=1100,
    )
    assert all(gate["pass"] for gate in gates.values())


def test_gate_fails_for_inert_reward_invalid_accounting_and_short_horizon():
    gates = evaluate_episode_gates(
        [row(deaths=0, t_fnd=None, steps=300, delivered=2806, demand=1192)],
        reward_term="deaths",
        headline_event_field="t_fnd",
        minimum_nonzero_term_records=1,
        minimum_event_records=1,
        minimum_training_horizon=1080,
    )
    assert gates["claim_reward_term_activated"]["pass"] is False
    assert gates["accounting_invariant_all_records"]["pass"] is False
    assert gates["headline_event_observed"]["pass"] is False
    assert gates["training_horizon_covers_claim_event"]["pass"] is False
