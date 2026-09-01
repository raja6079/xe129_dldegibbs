"""Configuration system for NAFNet model sizes.

Provides:
  * ``NAFNetConfig`` — a typed dataclass describing one NAFNet size.
  * ``NAFNET_CONFIGS`` — built-in presets (tiny / small / base) matching
    the values in ``configs/nafnet.yaml``.
  * ``load_nafnet_config`` — resolve a preset name OR a path to a YAML
    file containing ``presets:`` entries.
  * ``build_nafnet_from_config`` — instantiate a ``NAFNetDeGibbs`` from
    a config plus pipeline-level flags (in_ch, out_ch, comp2mag,
    noise_map_conditioning).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union

import yaml


@dataclass
class NAFNetConfig:
    """Shape + block hyperparameters for one NAFNet variant.

    Attributes:
        width: base channel width at the top of the U-Net.
        enc_blks: NAFBlock counts per encoder level (top -> bottom).
        middle_blks: NAFBlock count at the bottleneck.
        dec_blks: NAFBlock counts per decoder level (bottom -> top).
        dw_expand: depthwise branch expansion factor inside each NAFBlock.
        ffn_expand: channel-MLP branch expansion factor inside each NAFBlock.
        drop_out_rate: Dropout2d rate inside each NAFBlock (0 disables).
    """

    width: int
    enc_blks: List[int]
    middle_blks: int
    dec_blks: List[int]
    dw_expand: int = 2
    ffn_expand: int = 2
    drop_out_rate: float = 0.0

    def __post_init__(self) -> None:
        if len(self.enc_blks) != len(self.dec_blks):
            raise ValueError(
                f"enc_blks (len={len(self.enc_blks)}) and "
                f"dec_blks (len={len(self.dec_blks)}) must have equal length"
            )
        if self.width <= 0 or self.width % 2:
            raise ValueError(f"width must be a positive even integer (got {self.width})")
        if any(n <= 0 for n in self.enc_blks + self.dec_blks + [self.middle_blks]):
            raise ValueError("block counts must be >= 1 at every level")

    @property
    def padder_size(self) -> int:
        """Spatial multiple H and W must be divisible by."""
        return 2 ** len(self.enc_blks)

    def to_dict(self) -> dict:
        return asdict(self)


# Built-in presets — mirrors configs/nafnet.yaml.
NAFNET_PRESETS: dict = {
    "tiny": NAFNetConfig(
        width=16, enc_blks=[1, 1, 2, 2], middle_blks=4, dec_blks=[1, 1, 1, 1]
    ),
    "small": NAFNetConfig(
        width=32, enc_blks=[2, 2, 4, 8], middle_blks=12, dec_blks=[2, 2, 2, 2]
    ),
    "base": NAFNetConfig(
        width=64, enc_blks=[2, 2, 4, 8], middle_blks=12, dec_blks=[2, 2, 2, 2]
    ),
}

# Backwards-compatible dict view (matches the constant previously defined
# inline in nafnet.py).
NAFNET_CONFIGS: dict = {
    name: dict(
        width=cfg.width,
        enc_blks=list(cfg.enc_blks),
        middle_blks=cfg.middle_blks,
        dec_blks=list(cfg.dec_blks),
    )
    for name, cfg in NAFNET_PRESETS.items()
}


def load_nafnet_config(
    name_or_path: str,
    preset: Optional[str] = None,
) -> NAFNetConfig:
    """Resolve a NAFNet config from a preset name or a YAML file.

    Args:
        name_or_path: one of ``NAFNET_PRESETS`` keys (e.g. ``"tiny"``),
            OR a path to a YAML file with a top-level ``presets:`` mapping
            following the format of ``configs/nafnet.yaml``.
        preset: when ``name_or_path`` points to a YAML file, the preset
            key to select from it. Defaults to the value of
            ``$NAFNET_PRESET`` if set, otherwise ``"small"``.

    Returns:
        NAFNetConfig
    """
    if name_or_path in NAFNET_PRESETS:
        return NAFNET_PRESETS[name_or_path]

    if not os.path.isfile(name_or_path):
        raise FileNotFoundError(
            f"'{name_or_path}' is neither a built-in preset "
            f"({list(NAFNET_PRESETS)}) nor an existing YAML file"
        )

    with open(name_or_path, "r") as f:
        doc = yaml.safe_load(f)

    presets = doc.get("presets", {})
    block_defaults = doc.get("block_defaults", {})

    key = preset or os.environ.get("NAFNET_PRESET", "small")
    if key not in presets:
        raise KeyError(
            f"preset '{key}' not found in {name_or_path}. "
            f"Available: {list(presets)}"
        )

    entry = dict(presets[key])
    for k, v in block_defaults.items():
        entry.setdefault(k, v)
    return NAFNetConfig(**entry)


def build_nafnet_from_config(
    config: Union[str, NAFNetConfig],
    in_ch: int = 2,
    out_ch: int = 2,
    comp2mag: bool = False,
    noise_map_conditioning: bool = False,
    preset: Optional[str] = None,
    arch_variant: str = "v1",
):
    """Instantiate a ``NAFNetDeGibbs`` from a config (or preset name).

    Imported lazily so this module can be used as a pure config layer
    (e.g. inside argparse handling) without pulling in torch.
    """
    if isinstance(config, str):
        config = load_nafnet_config(config, preset=preset)

    # Lazy import to avoid a torch dependency at config-parse time.
    from nafnet import NAFNetDeGibbs

    return NAFNetDeGibbs(
        in_ch=in_ch,
        out_ch=out_ch,
        width=config.width,
        enc_blks=config.enc_blks,
        middle_blks=config.middle_blks,
        dec_blks=config.dec_blks,
        comp2mag=comp2mag,
        noise_map_conditioning=noise_map_conditioning,
        arch_variant=arch_variant,
    )


if __name__ == "__main__":
    # Smoke checks for the config layer (no torch needed for these).
    for name in ("tiny", "small", "base"):
        cfg = load_nafnet_config(name)
        print(f"{name:>5s}: width={cfg.width:3d}  "
              f"enc={cfg.enc_blks} mid={cfg.middle_blks} dec={cfg.dec_blks}  "
              f"padder_size={cfg.padder_size}")

    yaml_path = os.path.join(os.path.dirname(__file__), "configs", "nafnet.yaml")
    cfg = load_nafnet_config(yaml_path, preset="tiny")
    print(f"YAML tiny: {cfg.to_dict()}")
    print("nafnet_config OK")
