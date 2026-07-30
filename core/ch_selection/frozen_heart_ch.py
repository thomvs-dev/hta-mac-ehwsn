"""Evaluation-only adapter for the frozen HEART-CH checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from core.reproducibility import sha256_file


class FrozenHeartCH:
    def __init__(
        self,
        upstream: str | Path,
        checkpoint_relative_path: str,
        expected_sha256: str,
    ) -> None:
        self.upstream = Path(upstream).resolve()
        self.checkpoint_path = self.upstream / checkpoint_relative_path
        actual_hash = sha256_file(self.checkpoint_path)
        if actual_hash != expected_sha256:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: {actual_hash} != {expected_sha256}"
            )

        sys.path.insert(0, str(self.upstream))
        import config as cfg
        from train import build_agent

        self.agent, _ = build_agent(device="cpu", mode=cfg.AGENT_MODE)
        self.checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.agent.online_net.load_state_dict(
            self.checkpoint["online_state_dict"], strict=True
        )
        self.agent.online_net.eval()
        for parameter in self.agent.online_net.parameters():
            parameter.requires_grad_(False)

    @property
    def metadata(self) -> dict:
        excluded = {
            "online_state_dict",
            "target_state_dict",
            "optimizer_state_dict",
        }
        return {
            key: value
            for key, value in self.checkpoint.items()
            if key not in excluded
        }

    def select(self, state, edge_index, edge_weight, alive_mask):
        with torch.inference_mode():
            action, embedding = self.agent.select_action(
                state, edge_index, edge_weight, alive_mask
            )
        return action, embedding

