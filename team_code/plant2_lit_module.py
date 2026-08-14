"""
PyTorch Lightning wrapper for PlanT2.
Adapted from https://github.com/autonomousvision/plant2
"""

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn

from plant2_model import HFLM


class DictAsMember(dict):
    """Allow attribute-style access to nested dicts (for OmegaConf compatibility)."""
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")
        if isinstance(value, dict) and not isinstance(value, DictAsMember):
            value = DictAsMember(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _patch_hf_checkpoint(cfg):
    """Replace HF hub path with local path if available."""
    hf_ckpt = cfg.get("model", {}).get("network", {}).get("hf_checkpoint", "prajjwal1/bert-medium")
    project_root = Path(__file__).parent.parent
    local_hf = project_root / "checkpoints" / "hf_models" / hf_ckpt
    if local_hf.exists():
        cfg["model"]["network"]["hf_checkpoint"] = str(local_hf)
    return cfg


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(cfg, dict):
            cfg = DictAsMember(cfg)
        self.cfg = cfg

        _patch_hf_checkpoint(self.cfg)
        self.model = HFLM(self.cfg.model.network, self.cfg)

    def forward(self, batch):
        return self.model(batch)