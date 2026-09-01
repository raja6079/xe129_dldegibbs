import numpy as np
import torch


class ComplexToTensor:
    """Convert a set of complex ndarrays to a tensor.

    Args:
        complex_flag (boolean, default=True): If true, expects complex inputs.
        dtype (torch.dtype): Output tensor dtype.
        pf_flag (boolean, default=False): Whether to concatenate a standard
            partial Fourier reconstruction as an extra channel dimension.
    """

    def __init__(self, complex_flag=True, dtype=torch.float32, pf_flag=False):
        self.complex_flag = complex_flag
        self.dtype = dtype
        self.pf_flag = pf_flag

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        # Handle dat array
        if np.iscomplexobj(dat):
            # Complex data: split into real/imag channels
            if dat.ndim == 2:
                dat = np.stack([np.real(dat), np.imag(dat)], axis=0)
            else:
                dat = np.concatenate([np.real(dat), np.imag(dat)], axis=0)
        else:
            # Real data: ensure channel dimension exists
            if dat.ndim == 2:
                dat = np.expand_dims(dat, 0)

        # Handle target array
        if np.iscomplexobj(target):
            if target.ndim == 2:
                target = np.stack([np.real(target), np.imag(target)], axis=0)
            else:
                target = np.concatenate([np.real(target), np.imag(target)], axis=0)
        else:
            # Real data: ensure channel dimension exists
            if target.ndim == 2:
                target = np.expand_dims(target, 0)

        # Handle partial Fourier reconstructions if present
        if 'margosian_image' in sample:
            marg = sample['margosian_image']
            if marg.ndim == 2:
                marg = np.expand_dims(marg, 0)
            dat = np.concatenate([dat, marg], axis=0)
            
        if 'pocs_image' in sample:
            pocs = sample['pocs_image']
            if np.iscomplexobj(pocs):
                if pocs.ndim == 2:
                    pocs = np.stack([np.real(pocs), np.imag(pocs)], axis=0)
                else:
                    pocs = np.concatenate([np.real(pocs), np.imag(pocs)], axis=0)
            dat = np.concatenate([dat, pocs], axis=0)

        return {
            'target': torch.tensor(target.copy(), dtype=self.dtype),
            'dat': torch.tensor(dat.copy(), dtype=self.dtype),
        }
