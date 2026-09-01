"""NAFNet backbone for the DLDeGibbs pipeline.

Implements the Simple Baselines for Image Restoration architecture
(Chen et al., ECCV 2022) and wraps it for the Muckley DLDeGibbs
training/inference contract (2-channel complex input, global residual
skip, optional magnitude post-processing, optional noise-map
conditioning for non-Cartesian trajectories).

Tensor shape convention throughout: [B, C, H, W].
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Multiplicative gating activation from NAFNet.

    Input:  [B, C,   H, W]   (C must be even)
    Output: [B, C/2, H, W]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA) from NAFNet.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.conv(self.pool(x))
        return x * scale


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension for [B, C, H, W] tensors.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()

        dw_ch = channels * dw_expand
        ffn_ch = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_ch, dw_ch, kernel_size=3, padding=1, groups=dw_ch, bias=True
        )
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_ch // 2)
        self.conv2 = nn.Conv2d(dw_ch // 2, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_ch * 2, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_ch, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.drop1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        y = self.drop1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        y = self.drop2(y)
        x = x + y * self.gamma
        return x


class NAFNet(nn.Module):
    """NAFNet U-Net backbone.

    Input:  [B, in_ch,  H, W]
    Output: [B, out_ch, H, W]
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 2,
        width: int = 16,
        enc_blks: Sequence[int] = (1, 1, 2, 2),
        middle_blks: int = 4,
        dec_blks: Sequence[int] = (1, 1, 1, 1),
        arch_variant: str = "v1",
    ) -> None:
        super().__init__()

        if len(enc_blks) != len(dec_blks):
            raise ValueError("enc_blks and dec_blks must have the same length")
        if arch_variant not in ("v1", "v2"):
            raise ValueError(
                f"arch_variant must be 'v1' or 'v2' (got {arch_variant!r})"
            )
        self.arch_variant = arch_variant

        self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_ch, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        # v2 uses concat skips; a 1x1 conv reduces 2*ch back to ch before decoder blocks.
        self.skip_fuse = nn.ModuleList() if arch_variant == "v2" else None

        ch = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            if arch_variant == "v2":
                # PixelUnshuffle is invertible -> preserves all frequencies.
                self.downs.append(
                    nn.Sequential(
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(ch * 4, ch * 2, kernel_size=1, bias=True),
                    )
                )
            else:
                self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blks)])

        for n in dec_blks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            ch //= 2
            if arch_variant == "v2":
                self.skip_fuse.append(
                    nn.Conv2d(ch * 2, ch, kernel_size=1, bias=True)
                )
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.padder_size = 2 ** len(self.encoders)

    def _check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = self._check_image_size(x)

        x = self.intro(x)

        skips: List[torch.Tensor] = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for i, (up, dec, skip) in enumerate(
            zip(self.ups, self.decoders, reversed(skips))
        ):
            x = up(x)
            if self.arch_variant == "v2":
                # Concat skip preserves encoder HF; 1x1 conv fuses back to ch.
                x = torch.cat([x, skip], dim=1)
                x = self.skip_fuse[i](x)
            else:
                x = x + skip
            x = dec(x)

        x = self.ending(x)
        return x[..., :h, :w]


NAFNET_CONFIGS = {
    "tiny":  dict(width=16, enc_blks=[1, 1, 2, 2], middle_blks=4,  dec_blks=[1, 1, 1, 1]),
    "small": dict(width=32, enc_blks=[2, 2, 4, 8], middle_blks=12, dec_blks=[2, 2, 2, 2]),
    "base":  dict(width=64, enc_blks=[2, 2, 4, 8], middle_blks=12, dec_blks=[2, 2, 2, 2]),
}


class NAFNetDeGibbs(nn.Module):
    """NAFNet wrapped for the Muckley DLDeGibbs contract.

    Forward input:
        x:         [B, in_ch, H, W]
        noise_map: [B, 1,     H, W] (optional)

    Forward output:
        [B, out_ch, H, W]            if comp2mag is False
        [B, 1,      H, W]            if comp2mag is True
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 2,
        width: int = 16,
        enc_blks: Sequence[int] = (1, 1, 2, 2),
        middle_blks: int = 4,
        dec_blks: Sequence[int] = (1, 1, 1, 1),
        comp2mag: bool = False,
        noise_map_conditioning: bool = False,
        arch_variant: str = "v1",
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.comp2mag = comp2mag
        self.noise_map_conditioning = noise_map_conditioning
        self.arch_variant = arch_variant

        net_in_ch = in_ch + 1 if noise_map_conditioning else in_ch
        self.nafnet = NAFNet(
            in_ch=net_in_ch,
            out_ch=out_ch,
            width=width,
            enc_blks=enc_blks,
            middle_blks=middle_blks,
            dec_blks=dec_blks,
            arch_variant=arch_variant,
        )

    @classmethod
    def from_config(
        cls,
        config: str,
        in_ch: int = 2,
        out_ch: int = 2,
        comp2mag: bool = False,
        noise_map_conditioning: bool = False,
        arch_variant: str = "v1",
    ) -> "NAFNetDeGibbs":
        if config not in NAFNET_CONFIGS:
            raise KeyError(
                f"Unknown NAFNet config '{config}'. Available: {list(NAFNET_CONFIGS)}"
            )
        cfg = NAFNET_CONFIGS[config]
        return cls(
            in_ch=in_ch,
            out_ch=out_ch,
            comp2mag=comp2mag,
            noise_map_conditioning=noise_map_conditioning,
            arch_variant=arch_variant,
            **cfg,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        if self.noise_map_conditioning:
            if noise_map is None:
                noise_map = torch.ones_like(x[:, 0:1])
            inp = torch.cat([x, noise_map], dim=1)
        else:
            inp = x

        out = self.nafnet(inp)
        out = out + residual  # Muckley global residual — do not remove

        if self.comp2mag:
            if out.shape[1] < 2:
                raise RuntimeError(
                    "comp2mag requires at least 2 output channels (real, imag)"
                )
            out = torch.sqrt(out[:, 0:1] ** 2 + out[:, 1:2] ** 2 + 1e-12)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)

    model = NAFNetDeGibbs.from_config("tiny", in_ch=2, out_ch=2)
    model.eval()

    x = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        y = model(x)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet-tiny parameter count: {n_params:,}")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")

    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    print("NAFNet-tiny OK")
