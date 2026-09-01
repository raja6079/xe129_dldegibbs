"""Model factory for the DLDeGibbs pipeline.

Returns either the original Muckley wavelet U-Net (``UNetWavelet``) or
the new NAFNet backbone wrapped as ``NAFNetDeGibbs``. New training and
inference code should obtain models exclusively through ``build_model``;
this keeps the rest of the codebase agnostic to which backbone is in use.

Usage::

    from model_factory import build_model

    net = build_model(
        model="nafnet",
        nafnet_config="small",
        in_ch=2, out_ch=2,
        comp2mag=True,
        noise_map_conditioning=False,
    )
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


SUPPORTED_MODELS = ("wavelet", "nafnet")


def build_model(
    model: str = "wavelet",
    *,
    # --- common pipeline flags (Muckley contract) -------------------------
    in_ch: int = 2,
    out_ch: int = 2,
    comp2mag: bool = False,
    # --- wavelet-specific (ignored when model='nafnet') -------------------
    nlayers: int = 10,
    nchans: int = 64,
    leaky: bool = False,
    # --- nafnet-specific (ignored when model='wavelet') -------------------
    nafnet_config: str = "small",
    noise_map_conditioning: bool = False,
    arch_variant: str = "v1",
) -> nn.Module:
    """Build a DeGibbs backbone module.

    Args:
        model: ``"wavelet"`` (original Muckley U-Net) or ``"nafnet"``.
        in_ch: input channels (2 for complex, 1 for magnitude).
        out_ch: output channels.
        comp2mag: post-process complex output to magnitude.
        nlayers, nchans, leaky: wavelet-only hyperparameters.
        nafnet_config: NAFNet preset name (``"tiny"`` / ``"small"`` /
            ``"base"``) or path to a YAML file with ``presets:``.
        noise_map_conditioning: NAFNet only — accept a per-pixel noise
            variance map as an extra input channel (non-Cartesian).

    Returns:
        nn.Module. Both backbones honour the Muckley contract:
        ``out = net(x) + x`` is applied internally (the wavelet model
        uses ``resid=True``; ``NAFNetDeGibbs`` applies the residual in
        its ``forward``).
    """
    model = model.lower()
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"unknown model '{model}'. Supported: {SUPPORTED_MODELS}"
        )

    if model == "wavelet":
        # Lazy import so a missing wavelet dep doesn't block NAFNet users.
        from mrinet.unet_wavelet import UNetWavelet

        return UNetWavelet(
            ndims=2,
            nlayers=nlayers,
            in_ch=in_ch,
            out_ch=out_ch,
            top_filtnum=nchans,
            resid=True,
            wave_concat=True,
            comp2mag=comp2mag,
            leaky=leaky,
        )

    # model == 'nafnet'
    from nafnet_config import build_nafnet_from_config

    return build_nafnet_from_config(
        config=nafnet_config,
        in_ch=in_ch,
        out_ch=out_ch,
        comp2mag=comp2mag,
        noise_map_conditioning=noise_map_conditioning,
        arch_variant=arch_variant,
    )


def model_accepts_noise_map(net: nn.Module) -> bool:
    """Return True if ``net.forward`` takes an optional ``noise_map`` arg.

    Lets training/inference code decide whether to pass the noise map
    without coupling to a specific class.
    """
    return getattr(net, "noise_map_conditioning", False) is True


if __name__ == "__main__":
    import torch

    # NAFNet path
    net = build_model(model="nafnet", nafnet_config="tiny", in_ch=2, out_ch=2)
    x = torch.randn(1, 2, 256, 256)
    y = net(x)
    assert y.shape == x.shape, f"nafnet shape mismatch: {y.shape}"
    print(f"nafnet tiny OK: params={sum(p.numel() for p in net.parameters()):,}")

    # NAFNet with noise-map conditioning
    net2 = build_model(
        model="nafnet", nafnet_config="tiny",
        in_ch=2, out_ch=2, noise_map_conditioning=True,
    )
    nm = torch.ones(1, 1, 256, 256)
    y2 = net2(x, noise_map=nm)
    assert y2.shape == x.shape
    print(f"nafnet tiny + noise_map OK: accepts_noise_map="
          f"{model_accepts_noise_map(net2)}")

    print("model_factory OK")
