"""Model registry: bViT-style recurrent ViT, VCNet and the CIFAR ResNet baseline, behind the same step
interface (``embed`` / ``forward_step`` / ``readout`` / ``features`` / ``cfg.t_max``; ResNet: one step)."""

from dataclasses import asdict

from .bvit import PRESETS as BVIT_PRESETS
from .bvit import BViTConfig, RecurrentViT
from .bvit import build_model as build_bvit
from .resnet import PRESETS as RESNET_PRESETS
from .resnet import ResNet, ResNetConfig, build_resnet
from .vcnet import PRESETS as VCNET_PRESETS
from .vcnet import VCNet, VCNetConfig, build_vcnet

PRESETS = {**BVIT_PRESETS, **VCNET_PRESETS, **RESNET_PRESETS}


def build_model(name: str = "bvit_s", **overrides):
    if name in VCNET_PRESETS:
        return build_vcnet(name, **overrides)
    if name in RESNET_PRESETS:
        return build_resnet(name, **overrides)
    return build_bvit(name, **overrides)


def model_from_config(cfg: dict):
    """Rebuild a model from the ``model_cfg`` stored in a checkpoint (no ``arch`` key = bViT)."""
    if cfg.get("arch") == "vcnet":
        return VCNet(VCNetConfig(**cfg))
    if cfg.get("arch") == "resnet":
        return ResNet(ResNetConfig(**cfg))
    return RecurrentViT(BViTConfig(**cfg))


def load_checkpoint_model(ck: dict):
    model = model_from_config(ck["model_cfg"])
    model.load_state_dict(ck["model"])
    return model


def config_dict(model) -> dict:
    return asdict(model.cfg)


__all__ = ["PRESETS", "BViTConfig", "RecurrentViT", "ResNet", "ResNetConfig", "VCNet", "VCNetConfig", "build_model",
           "config_dict", "load_checkpoint_model", "model_from_config"]
