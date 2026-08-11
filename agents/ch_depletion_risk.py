"""Observable scheduled-CH depletion risk used only by Step 3."""

from __future__ import annotations

import math


def validate_ch_risk_config(payload: dict) -> dict:
    if payload.get("status") != "frozen_step3_development_risk":
        raise ValueError("CH-risk config is not frozen Step 3 development evidence")
    if payload.get("learned_intervention") != "mac_allocation_only":
        raise ValueError("Step 3 must not modify CH selection")
    if payload.get("ch_schedule_modified") is not False:
        raise ValueError("Step 3 CH schedule must remain frozen")
    if payload.get("apply_to") != "raw_learning_reward_before_frozen_c51_scale":
        raise ValueError("unsupported CH-risk application point")
    numeric = (
        "reserve_threshold_fraction",
        "forecast_reference_j",
        "distance_reference_m",
        "forecast_relief_weight",
        "distance_exposure_weight",
        "scale",
        "weight",
    )
    for name in numeric:
        value = float(payload[name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite CH-risk field {name}")
    if not 0.0 < float(payload["reserve_threshold_fraction"]) <= 1.0:
        raise ValueError("reserve threshold must be in (0,1]")
    if float(payload["forecast_reference_j"]) <= 0.0:
        raise ValueError("forecast reference must be positive")
    if float(payload["distance_reference_m"]) <= 0.0:
        raise ValueError("distance reference must be positive")
    if not 0.0 <= float(payload["forecast_relief_weight"]) <= 1.0:
        raise ValueError("forecast relief weight must be in [0,1]")
    if float(payload["distance_exposure_weight"]) < 0.0:
        raise ValueError("distance exposure weight must be nonnegative")
    if float(payload["scale"]) <= 0.0 or float(payload["weight"]) <= 0.0:
        raise ValueError("risk scale and weight must be positive")
    if payload.get("uses_realized_future_harvest") is not False:
        raise ValueError("risk signal must not leak realized future harvest")
    return payload


def ch_depletion_risk(
    config: dict,
    *,
    reserve_fraction: float,
    forecast_harvest_j: float,
    distance_to_bs_m: float,
    intended_delivered_packets: int,
    frame_slot_budget: int,
) -> dict:
    validate_ch_risk_config(config)
    threshold = float(config["reserve_threshold_fraction"])
    reserve_deficit = min(1.0, max(0.0, (threshold - reserve_fraction) / threshold))
    harvest_relief = min(
        1.0,
        max(0.0, forecast_harvest_j / float(config["forecast_reference_j"])),
    )
    distance_exposure = min(
        1.0,
        max(0.0, distance_to_bs_m / float(config["distance_reference_m"])),
    )
    service_load = min(
        1.0,
        max(0.0, intended_delivered_packets / max(1, int(frame_slot_budget))),
    )
    multiplier = (
        1.0 + float(config["distance_exposure_weight"]) * distance_exposure
    ) * (1.0 - float(config["forecast_relief_weight"]) * harvest_relief)
    score = reserve_deficit * service_load * multiplier
    return {
        "raw_penalty": -float(score),
        "risk_score": float(score),
        "reserve_fraction": float(reserve_fraction),
        "reserve_deficit": float(reserve_deficit),
        "forecast_harvest_j": float(forecast_harvest_j),
        "harvest_relief": float(harvest_relief),
        "distance_to_bs_m": float(distance_to_bs_m),
        "distance_exposure": float(distance_exposure),
        "intended_delivered_packets": int(intended_delivered_packets),
        "service_load": float(service_load),
        "scheduled_ch_role": True,
    }
