""" A collection of neural network utilities for MRI in PyTorch.

Functions include:
    fftshift - torch.fft.fftshift (built-in)
    ifftshift - torch.fft.ifftshift (built-in)
    back_fft - applies fftshift(ifft(ifftshift(input))) in PyTorch
    dwt2d - applies 2D discrete wavelet transform
    idwt2d - applies inverse 2D discrete wavelet transform
"""

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class back_fft(nn.Module):
    """Applies fftshift(ifftn(ifftshift(input))) using native complex tensors.

    Input tensor shape: [B, 2, ...] where channel 0 = real, channel 1 = imag.
    Output tensor shape: same as input.
    """

    def __init__(self, ndims, startdim=2):
        super().__init__()
        self.ndims = ndims
        self.startdim = startdim
        # For ndims=4 (B,C,H,W): after removing channel dim, spatial are (1,2)
        self.fft_dims = tuple(range(startdim - 1, ndims - 2 + startdim - 1))

    def forward(self, x):
        # Convert [B, 2, ...] real tensor to [B, ...] complex tensor
        x_complex = torch.complex(x[:, 0], x[:, 1])
        x_complex = torch.fft.ifftshift(x_complex, dim=self.fft_dims)
        x_complex = torch.fft.ifftn(x_complex, dim=self.fft_dims)
        x_complex = torch.fft.fftshift(x_complex, dim=self.fft_dims)
        # Convert back to [B, 2, ...] real tensor
        return torch.stack([x_complex.real, x_complex.imag], dim=1)


class dwt2d(nn.Module):
    def __init__(self, channels=1, wtype='haar', levels=1):
        super().__init__()
        if wtype == 'haar':
            wt = pywt.Wavelet(wtype)
            dec_hi = torch.tensor(wt.dec_hi[::-1], dtype=torch.float64)
            dec_lo = torch.tensor(wt.dec_lo[::-1], dtype=torch.float64)
        else:
            print('filter type not found!')

        self.register_buffer(
            'filters',
            torch.stack(
                [
                    dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                    dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                    dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                    dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)
                ] * channels
            ).unsqueeze(1)
        )

        self.channels = channels
        self.levels = levels

    def forward(self, x, levels=-1):
        if levels == -1:
            levels = self.levels

        h = x.size(2)
        w = x.size(3)
        x = torch.nn.functional.conv2d(
            x,
            self.filters,
            stride=2,
            groups=self.channels)

        if levels > 1:
            x[:, :1] = self.forward(x, levels-1)

        return x


class idwt2d(nn.Module):
    def __init__(self, channels=1, wtype='haar', levels=1, maxgroup=False):
        super().__init__()
        if wtype == 'haar':
            wt = pywt.Wavelet(wtype)
            rec_hi = torch.tensor(wt.rec_hi, dtype=torch.float64)
            rec_lo = torch.tensor(wt.rec_lo, dtype=torch.float64)
        else:
            print('filter type not found!')

        self.register_buffer(
            'inv_filters',
            torch.stack(
                [
                    rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                    rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                    rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                    rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)
                ] * channels
            ).unsqueeze(1)
        )

        self.maxgroup = maxgroup
        self.channels = channels
        self.levels = levels

    def forward(self, x, levels=-1):
        if levels == -1:
            levels = self.levels

        h = x.size(2)
        w = x.size(3)

        if levels > 1:
            x[:, :1] = self.forward(x, levels-1)

        if self.maxgroup:
            out_chans = self.inv_filters.shape[0]
        else:
            out_chans = self.channels

        x = torch.nn.functional.conv_transpose2d(
            x,
            self.inv_filters,
            stride=2,
            groups=out_chans)

        return x
