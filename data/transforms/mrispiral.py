import numpy as np


class MriSpiral:
    """Simulate stack-of-spirals (s3d) MRI artifacts for 129Xe imaging.

    This transform converts a clean complex image into a spiral-reconstruction
    estimate by synthesizing the three artifact components characteristic of
    stack-of-spirals acquisitions:

    1. Colored noise from density compensation after non-uniform gridding. The
       noise spectral density (NSD) follows a radial profile parameterized by a
       DCF saturation rate with an optional Gaussian boundary overshoot.
    2. A deterministic complex modulation transfer function (MTF) combining T2*
       magnitude decay and B0 off-resonance phase, both proportional to the
       readout time t ~ r**2 of a center-out spiral.
    3. Through-plane (kz) Cartesian sampling, which modulates the projected
       noise amplitude in each axial slice via a scalar coupling factor.

    The output is ``dat = apply_psf(clean) + colored_noise`` while ``target``
    (the clean image) is left untouched. All k-space operations use centered
    FFT conventions consistent with the rest of the transform package.

    Args:
        dat_op (boolean, default=True): Whether to apply the artifact to the
            'dat' array.
        targ_op (boolean, default=False): Whether to apply the artifact to the
            'target' array (rarely desired; the target is normally clean).
        noise_range (tuple, default=(0.001, 0.05)): Min/max noise standard
            deviation as a fraction of the peak signal. ~3x wider than the
            Cartesian range to reflect lower spiral SNR.
        nsd_version (str, default='v1'): NSD coloring model. 'v1' is the
            parametric DCF saturation + Gaussian edge model; 'v2' is the
            strictly-monotone power-law/sigmoid blend; 'random' picks one per
            sample.
        alpha_range (tuple, default=(0.3, 8.0)): DCF saturation rate (v1).
        edge_amp_range (tuple, default=(0.0, 10.0)): DCF boundary overshoot
            amplitude (v1).
        edge_sigma_range (tuple, default=(0.005, 0.10)): Edge spike width (v1).
        gamma_range (tuple, default=(0.3, 1.0)): Power-law curvature (v2).
        r0_range (tuple, default=(0.70, 1.0)): Sigmoid center (v2).
        sigma_range (tuple, default=(0.03, 0.15)): Sigmoid width (v2).
        eta_range (tuple, default=(0.0, 1.0)): Power-law/sigmoid blend (v2).
        skip_psf_prob (double, default=0.4): Probability of skipping the PSF
            entirely (T2*-corrected or very short readout scans).
        rate_mean (double, default=0.6): Mean of the exponential distribution
            for T2* magnitude decay rate.
        rate_max (double, default=1.5): Maximum T2* decay rate (clipped).
        offres_prob (double, default=0.3): Probability of non-zero off-resonance
            phase (otherwise well-shimmed, beta=0).
        beta_range (tuple, default=(0.1, 1.2)): Total off-resonance phase at the
            k-space edge, in radians.
        psf_blend_range (tuple, default=(0.2, 0.5)): Blend strength for PSF
            application when enabled. Final image is
            ``(1 - s) * image + s * blurred`` with ``s`` sampled uniformly.
        kz_full_prob (double, default=0.40): Probability of full kz sampling.
        kz_arc_prob (double, default=0.35): Probability of a kz ARC profile
            (amplified noise outside calibration). The remaining probability
            mass is assigned to kz partial Fourier.
        cal_width_range (tuple, default=(0.15, 0.45)): kz ARC calibration width.
        noise_amp_range (tuple, default=(1.5, 4.0)): kz ARC noise amplification.
        pf_fract_range (tuple, default=(0.10, 0.90)): kz partial Fourier
            sampled fraction.
        n_kz (int, default=64): Number of kz phase-encode points to model.
    """

    def __init__(self, dat_op=True, targ_op=False,
                 noise_range=(0.001, 0.05),
                 nsd_version='v1',
                 alpha_range=(0.3, 8.0),
                 edge_amp_range=(0.0, 10.0),
                 edge_sigma_range=(0.005, 0.10),
                 gamma_range=(0.3, 1.0),
                 r0_range=(0.70, 1.0),
                 sigma_range=(0.03, 0.15),
                 eta_range=(0.0, 1.0),
                 skip_psf_prob=0.4,
                 rate_mean=0.6,
                 rate_max=1.5,
                 offres_prob=0.3,
                 beta_range=(0.1, 1.2),
                 psf_blend_range=(0.2, 0.5),
                 kz_full_prob=0.40,
                 kz_arc_prob=0.35,
                 cal_width_range=(0.15, 0.45),
                 noise_amp_range=(1.5, 4.0),
                 pf_fract_range=(0.10, 0.90),
                 n_kz=64):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.noise_range = noise_range
        self.nsd_version = nsd_version
        self.alpha_range = alpha_range
        self.edge_amp_range = edge_amp_range
        self.edge_sigma_range = edge_sigma_range
        self.gamma_range = gamma_range
        self.r0_range = r0_range
        self.sigma_range = sigma_range
        self.eta_range = eta_range
        self.skip_psf_prob = skip_psf_prob
        self.rate_mean = rate_mean
        self.rate_max = rate_max
        self.offres_prob = offres_prob
        self.beta_range = beta_range
        self.psf_blend_range = psf_blend_range
        self.kz_full_prob = kz_full_prob
        self.kz_arc_prob = kz_arc_prob
        self.cal_width_range = cal_width_range
        self.noise_amp_range = noise_amp_range
        self.pf_fract_range = pf_fract_range
        self.n_kz = n_kz

        # Cache the radial grid / support mask keyed by spatial shape.
        self._grid_cache = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _fft2(x, axes):
        return np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes),
            axes=axes)

    @staticmethod
    def _ifft2(x, axes):
        return np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes),
            axes=axes)

    def _radial_grid(self, shape):
        """Return (r, cs_mask) for a 2D spatial shape, with caching."""
        if shape not in self._grid_cache:
            ny, nx = shape
            ky = (np.arange(ny) - ny // 2) / (ny / 2.0)
            kx = (np.arange(nx) - nx // 2) / (nx / 2.0)
            kyy, kxx = np.meshgrid(ky, kx, indexing='ij')
            r = np.sqrt(kyy ** 2 + kxx ** 2)
            cs_mask = (r <= 1.0).astype(np.float64)
            self._grid_cache[shape] = (r, cs_mask)
        return self._grid_cache[shape]

    # ------------------------------------------------------------- components
    def _build_nsd(self, r, cs_mask):
        """Radial noise spectral density profile (normalized noise power)."""
        version = self.nsd_version
        if version == 'random':
            version = 'v1' if np.random.uniform() < 0.5 else 'v2'

        if version == 'v2':
            gamma = np.random.uniform(*self.gamma_range)
            r0 = np.random.uniform(*self.r0_range)
            sigma = np.random.uniform(*self.sigma_range)
            eta = np.random.uniform(*self.eta_range)
            nsd = (1.0 - eta) * (r ** gamma) + \
                eta * 0.5 * (1.0 + np.tanh((r - r0) / sigma))
        else:
            alpha = np.random.uniform(*self.alpha_range)
            base = np.sqrt(np.tanh(alpha * r) + 1e-6)
            edge_amp = np.random.uniform(*self.edge_amp_range)
            edge_sigma = np.random.uniform(*self.edge_sigma_range)
            edge = edge_amp * np.exp(-(r - 1.0) ** 2 /
                                     (2.0 * edge_sigma ** 2))
            nsd = base + edge

        nsd = nsd * cs_mask
        power = np.sqrt(np.mean(nsd ** 2))
        nsd = nsd / np.maximum(power, 1e-12)
        return nsd

    def _build_mtf(self, r):
        """Complex T2* + off-resonance MTF with unit DC gain (or None)."""
        if np.random.uniform() < self.skip_psf_prob:
            return None

        r_squared = r ** 2

        # T2* magnitude decay: exponential draw clipped to a maximum.
        rate = np.random.exponential(self.rate_mean)
        rate = np.minimum(rate, self.rate_max)

        # Off-resonance phase (well-shimmed cases have beta = 0).
        if np.random.uniform() < self.offres_prob:
            beta = np.random.uniform(*self.beta_range)
        else:
            beta = 0.0
        sign = np.random.choice([-1.0, 1.0])
        phase = sign * beta * r_squared

        mtf = np.exp(-rate * r_squared) * np.exp(1j * phase)
        cy, cx = r.shape[0] // 2, r.shape[1] // 2
        mtf = mtf / mtf[cy, cx]
        return mtf

    def _kz_factor(self):
        """Scalar through-plane noise coupling factor."""
        n_kz = self.n_kz
        kz = (np.arange(n_kz) - n_kz // 2) / (n_kz / 2.0)

        u = np.random.uniform()
        if u < self.kz_full_prob:
            kz_profile = np.ones(n_kz)
        elif u < self.kz_full_prob + self.kz_arc_prob:
            cal_width = np.random.uniform(*self.cal_width_range)
            noise_amp = np.random.uniform(*self.noise_amp_range)
            kz_profile = np.where(np.abs(kz) <= cal_width, 1.0, noise_amp)
        else:
            pf_fract = np.random.uniform(*self.pf_fract_range)
            kz_profile = np.where(kz < pf_fract, 1.0, 0.3)

        return np.sqrt(np.mean(kz_profile ** 2))

    def _corrupt(self, image, peak_signal):
        """Apply the full spiral artifact model to a 2D complex image."""
        shape = image.shape
        axes = (0, 1)
        r, cs_mask = self._radial_grid(shape)

        # 1. Colored noise from density compensation.
        nsd = self._build_nsd(r, cs_mask)
        noise_std = peak_signal * np.random.uniform(*self.noise_range)
        white = noise_std * (np.random.normal(size=shape) +
                             1j * np.random.normal(size=shape))
        colored_noise = self._ifft2(self._fft2(white, axes) * nsd, axes)

        # 3. Through-plane (kz) coupling scales the projected noise amplitude.
        # colored_noise = colored_noise * self._kz_factor()

        # 2. Deterministic complex PSF (T2* + off-resonance).
        mtf = self._build_mtf(r)
        if mtf is not None:
            blurred = self._ifft2(self._fft2(image, axes) * mtf, axes)
            s_blend = np.random.uniform(*self.psf_blend_range)
            blurred = (1.0 - s_blend) * image + s_blend * blurred
        else:
            blurred = image

        return blurred + colored_noise

    # ----------------------------------------------------------------- call
    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with complex 'target' and 'dat' numpy
                arrays. 'dat' is coil-first (C, H, W); 'target' is (H, W).
        Returns:
            sample (dict): with 'dat' (and optionally 'target') replaced by the
                spiral-artifact estimate.
        """
        # Peak signal is measured on the clean image (image domain), not on any
        # cached k-space 'siglevel', which lives on a different scale.
        peak_signal = max(float(np.max(np.absolute(sample['dat']))), 1e-12)

        if self.dat_op:
            dat = sample['dat']
            if dat.ndim == 2:
                sample['dat'] = self._corrupt(dat, peak_signal)
            else:
                out = np.empty_like(dat, dtype=np.complex128)
                for c in range(dat.shape[0]):
                    out[c] = self._corrupt(dat[c], peak_signal)
                sample['dat'] = out

        if self.targ_op:
            target = sample['target']
            if target.ndim == 2:
                sample['target'] = self._corrupt(target, peak_signal)
            else:
                out = np.empty_like(target, dtype=np.complex128)
                for c in range(target.shape[0]):
                    out[c] = self._corrupt(target[c], peak_signal)
                sample['target'] = out

        return sample

    def __repr__(self):
        return self.__class__.__name__
