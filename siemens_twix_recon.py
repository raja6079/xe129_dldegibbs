"""Siemens twix (.dat) -> k-space -> image front-end for 129Xe Cartesian GRE.

This is the Siemens counterpart of the GE pfile recon scripts in this folder.
It reads a Siemens raw ``.dat`` file with ``twixtools``, grids the acquired
k-space onto the full (possibly partial-Fourier) lattice using the per-readout
loop counters, removes readout oversampling, does a plain zero-filled inverse
FFT, and dumps every intermediate to a ``.mat`` for MATLAB analysis.

IMPORTANT (state of this script):
- Partial Fourier is handled by iterative **POCS** on every partial Cartesian
  axis: phase (Lin), partition (Par), and the **readout asymmetric echo** (Col).
  The readout is first gridded onto the full symmetric lattice with the echo
  (center_col) placed at the array centre, then POCS fills the truncated side;
  without this the off-centre / truncated readout produces horizontal-line
  ringing. Both the zero-filled and POCS complex images are dumped.
- Single receive channel is assumed (verified for HV03). Multi-channel data
  would need coil combination added.
- No DL model is applied here; downstream DL de-Gibbs runs on image_pocs_complex.

Geometry is discovered from the data, not hard-coded:
- readout oversampling factor  = SamplesInScan / (2 * CenterCol)  (=2 for HV03)
- full phase grid              = sKSpace.lPhaseEncodingLines
- full partition grid          = sKSpace.lPartitions (3D) or 1 (2D)
- PF gridding shift            = (full//2) - Center<Lin|Par>

Example:
    python siemens_twix_recon.py \
        --dat /home/raja/xe129_dldegibbs/3T_vent_data_Manchester/RAW/HV03/meas_MID00041_FID204963_gre_Xe_3D.dat \
        --outdir /home/raja/xe129_dldegibbs/3T_vent_data_Manchester/RECON/HV03
"""

import argparse
import os

import numpy as np
import scipy.io as sio
import twixtools


def _centered_ifft(kspace: np.ndarray, axes) -> np.ndarray:
    """Centered N-D inverse FFT (image = ifft of k-space, DC in the middle)."""
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes),
        axes=axes,
    )


def _centered_fft(image: np.ndarray, axes) -> np.ndarray:
    """Centered N-D forward FFT (image -> k-space, DC in the middle)."""
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes),
        axes=axes,
    )


def _get_yaps(twix_meas):
    """Return the MeasYaps protocol dict (ascconv) for a measurement."""
    return twix_meas["hdr"]["MeasYaps"]


def read_kspace(dat_path: str, kspace_scale: float = 1e10):
    """Read a Siemens .dat and return gridded k-space + geometry metadata.

    twixtools returns raw ADC data on a tiny native scale (|k| ~ 1e-6 here),
    which is awkward for viewing/precision. ``kspace_scale`` is a fixed constant
    (data-independent, so consistent across subjects) applied to all k-space;
    it does not affect the DL models, which normalize each slice by its own max.

    Returns a dict with:
        kspace_grid : complex64 [Col, Lin_full, Par_full, Rep]  (zero-filled)
        kspace_raw  : complex64 [Col, Lin_acq,  Par_acq,  Rep]  (as acquired)
        meta        : dict of geometry/header parameters
    """
    twix = twixtools.read_twix(dat_path, parse_pmu=False, verbose=False)[-1]
    yaps = _get_yaps(twix)
    ksp_hdr = yaps["sKSpace"]
    slice_hdr = yaps["sSliceArray"]["asSlice"][0]

    mdbs = [m for m in twix["mdb"] if m.is_image_scan()]
    if not mdbs:
        raise RuntimeError("No image-scan readouts found in file.")

    n_col = int(mdbs[0].mdh.SamplesInScan)
    center_col = int(mdbs[0].mdh.CenterCol)
    center_lin = int(mdbs[0].mdh.CenterLin)
    center_par = int(mdbs[0].mdh.CenterPar)

    n_cha = np.asarray(mdbs[0].data).shape[0]
    if n_cha != 1:
        raise NotImplementedError(
            f"{n_cha} receive channels found; only single-channel is wired up. "
            "Add coil combination before proceeding."
        )

    dimension = int(ksp_hdr["ucDimension"])          # 2 => 2D, 4 => 3D
    is_3d = dimension == 4

    # Slice-plane orientation from the slice normal (dominant PCS axis):
    # Sag=sagittal, Cor=coronal, Tra=axial. Drives the display reorientation
    # in main() so anatomy comes out the right way up.
    s_normal = slice_hdr.get("sNormal", {}) or {}
    normal_axis = max(
        (("Sag", "dSag"), ("Cor", "dCor"), ("Tra", "dTra")),
        key=lambda ax: abs(s_normal.get(ax[1], 0.0)),
    )[0] if s_normal else "Tra"
    n_lin_full = int(ksp_hdr["lPhaseEncodingLines"])
    n_par_full = int(ksp_hdr["lPartitions"]) if is_3d else 1

    lin_idx = np.array([m.cLin for m in mdbs])
    par_idx = np.array([m.cPar for m in mdbs])
    rep_idx = np.array([m.cRep for m in mdbs])
    n_rep = int(rep_idx.max()) + 1

    n_lin_acq = int(lin_idx.max()) + 1
    n_par_acq = int(par_idx.max()) + 1

    base_res = int(ksp_hdr["lBaseResolution"])

    # Readout: with an asymmetric echo the acquired DC (center_col) is NOT at the
    # array centre (e.g. RADicA: center_col=40 of 104). Grid the readout onto the
    # full 2x-oversampled lattice (2*base_res) with DC placed at the centre, so the
    # iFFT is properly centred; the truncated low-readout side is zero-filled.
    # Symmetric echoes (center_col = base_res, n_col = 2*base_res) get col_shift=0.
    n_col_full = 2 * base_res
    col_shift = (n_col_full // 2) - center_col
    os_factor = n_col_full / base_res if base_res else 1.0   # = 2x readout OS

    # PF gridding shift: align acquired k-space center onto the full-grid center.
    lin_shift = (n_lin_full // 2) - center_lin
    par_shift = (n_par_full // 2) - center_par if is_3d else 0

    col_grid = np.arange(n_col) + col_shift
    col_ok = (col_grid >= 0) & (col_grid < n_col_full)

    # As-acquired dense array (no gridding), for raw inspection in MATLAB.
    kspace_raw = np.zeros((n_col, n_lin_acq, n_par_acq, n_rep), dtype=np.complex64)
    # Zero-filled full-grid k-space (readout centred + phase/partition gridded).
    kspace_grid = np.zeros((n_col_full, n_lin_full, n_par_full, n_rep), dtype=np.complex64)

    for m in mdbs:
        line = np.asarray(m.data)[0] * kspace_scale  # single channel -> [Col]
        r, l, p = m.cRep, m.cLin, m.cPar
        kspace_raw[:, l, p, r] = line
        gl = l + lin_shift
        gp = (p + par_shift) if is_3d else 0
        if 0 <= gl < n_lin_full and 0 <= gp < n_par_full:
            kspace_grid[col_grid[col_ok], gl, gp, r] = line[col_ok]

    meta = {
        "protocol": str(yaps.get("tProtocolName", "")),
        "kspace_scale": float(kspace_scale),
        "dimension": dimension,
        "is_3d": int(is_3d),
        "normal_axis": normal_axis,
        "n_col": n_col_full,        # full readout grid used by the recon/iFFT
        "n_col_acq": n_col,         # acquired readout samples (kspace_raw)
        "center_col": center_col,
        "col_shift": col_shift,
        "os_factor": os_factor,
        "n_lin_full": n_lin_full,
        "n_lin_acq": n_lin_acq,
        "center_lin": center_lin,
        "lin_shift": lin_shift,
        "n_par_full": n_par_full,
        "n_par_acq": n_par_acq,
        "center_par": center_par,
        "par_shift": par_shift,
        "n_rep": n_rep,
        "n_cha": n_cha,
        "base_resolution": base_res,
        "phase_partial_fourier_code": int(ksp_hdr["ucPhasePartialFourier"]),
        "slice_partial_fourier_code": int(ksp_hdr["ucSlicePartialFourier"]),
        "asym_echo_mode": int(ksp_hdr["ucAsymmetricEchoMode"]),
        "pocs_flag": int(ksp_hdr["ucPOCS"]),
        "readout_fov_mm": float(slice_hdr.get("dReadoutFOV", 0.0)),
        "phase_fov_mm": float(slice_hdr.get("dPhaseFOV", 0.0)),
        "slice_thickness_mm": float(slice_hdr.get("dThickness", 0.0)),
    }
    # Full-grid sampling masks (True = acquired) on each Cartesian axis. Sampling
    # is separable, so these 1-D masks fully describe the acquired locations.
    # Readout (Col): partial when the echo is asymmetric (col_shift != 0).
    col_sampled = np.zeros(n_col_full, dtype=bool)
    col_sampled[col_grid[col_ok]] = True
    # Phase (Lin): partial when phase partial-Fourier is on.
    lin_sampled = np.zeros(n_lin_full, dtype=bool)
    lin_sampled[np.clip(np.unique(lin_idx) + lin_shift, 0, n_lin_full - 1)] = True
    # Partition (Par): partial when slice partial-Fourier is on (3D only).
    par_sampled = np.zeros(n_par_full, dtype=bool)
    if is_3d:
        par_sampled[np.clip(np.unique(par_idx) + par_shift, 0, n_par_full - 1)] = True
    else:
        par_sampled[:] = True

    return {
        "kspace_grid": kspace_grid,
        "kspace_raw": kspace_raw,
        "col_sampled": col_sampled,
        "lin_sampled": lin_sampled,
        "par_sampled": par_sampled,
        "meta": meta,
    }


def _remove_os(img: np.ndarray, meta: dict) -> np.ndarray:
    """Crop the oversampled readout (axis 0) to the base resolution FOV."""
    n_col = meta["n_col"]
    x_keep = int(round(n_col / meta["os_factor"]))
    start = (n_col - x_keep) // 2
    return img[start:start + x_keep].astype(np.complex64)


def _axis_phase_window(sampled: np.ndarray) -> np.ndarray:
    """1-D window for the POCS phase estimate along one axis.

    Fully-sampled axis -> all-ones (use its full phase content). Partial axis ->
    a Hamming taper over the *symmetric* band about DC (grid center = n//2), zero
    outside it, so the phase reference is low-resolution but artifact-free.
    """
    n = sampled.shape[0]
    if sampled.all():
        return np.ones(n, dtype=np.float32)
    acq = np.where(sampled)[0]
    c = n // 2
    half = int(min(acq.max() - c, c - acq.min()))
    win = np.zeros(n, dtype=np.float32)
    if half <= 0:                       # no symmetric band -> DC only
        win[c] = 1.0
    else:
        win[c - half:c + half + 1] = np.hamming(2 * half + 1).astype(np.float32)
    return win


def pocs_recon(kspace_grid, masks, meta, n_iter=15) -> np.ndarray:
    """POCS partial-Fourier recon over every partial Cartesian axis.

    Handles the phase (Lin) partial-Fourier AND the readout (Col) asymmetric echo
    (and partition PF in 3D) jointly: the missing samples on each axis are filled
    using a low-resolution phase estimate from the symmetric center region,
    iterating data-consistency <-> phase-constraint. Because Cartesian sampling is
    separable, the acquired region is the outer product of the per-axis masks.

    Args:
        masks: (col_sampled, lin_sampled, par_sampled) full-grid boolean vectors.
    Input  : [Col, Lin, Par, Rep]   Output: complex64 [X, Lin, Par, Rep].
    """
    is_3d = bool(meta["is_3d"])
    axes = (0, 1, 2) if is_3d else (0, 1)   # FFT axes per volume/slice
    col_s, lin_s, par_s = masks

    # Separable acquired region -> full-grid boolean mask (broadcast over Rep).
    mask = (col_s[:, None, None] & lin_s[None, :, None] & par_s[None, None, :])[..., None]
    if mask.all():
        # Fully sampled on every axis -> POCS is a no-op, just zero-fill iFFT.
        return _remove_os(_centered_ifft(kspace_grid, axes=axes), meta)

    # Low-resolution phase reference from the symmetric center region of the
    # partial axes (outer product of the per-axis symmetric windows).
    wcol = _axis_phase_window(col_s)
    wlin = _axis_phase_window(lin_s)
    wpar = _axis_phase_window(par_s) if is_3d else np.ones(par_s.shape[0], np.float32)
    win = (wcol[:, None, None] * wlin[None, :, None] * wpar[None, None, :])[..., None]

    phase = np.angle(_centered_ifft(kspace_grid * win, axes=axes))
    phase_factor = np.exp(1j * phase).astype(np.complex64)

    # Iterate: enforce phase in image space, data consistency in k-space.
    img = _centered_ifft(kspace_grid, axes=axes)
    for _ in range(n_iter):
        img = (np.abs(img) * phase_factor).astype(np.complex64)
        k = _centered_fft(img, axes=axes)
        k = np.where(mask, kspace_grid, k)      # restore measured samples
        img = _centered_ifft(k, axes=axes)

    return _remove_os(img, meta)


def orient_for_display(img: np.ndarray, meta: dict) -> np.ndarray:
    """Reorient the recon volume to a standard anatomical display.

    Input/Output: [X, Lin, Par, Rep]. The raw iFFT axes come out in scanner
    logical order (readout=X, phase=Lin), which for these coronal 129Xe GRE
    scans leaves the image upside-down (readout runs foot->head).

    Coronal (sNormal ~ dCor): X = superior-inferior, Lin = left-right. Flip X so
    the apices sit at the top and the diaphragmatic bases at the bottom; the
    left-right (Lin) order already matches radiological convention (patient-right
    on viewer-left), so it is left untouched. Non-coronal orientations are passed
    through unchanged (their display convention is not established here).
    """
    if meta.get("normal_axis") == "Cor":
        return np.flip(img, axis=0)          # superior-inferior flip (X axis)
    return img


def zerofilled_recon(kspace_grid: np.ndarray, meta: dict) -> np.ndarray:
    """Zero-filled inverse FFT with readout oversampling removed.

    Input  : [Col, Lin, Par, Rep]  (Col is oversampled)
    Output : complex64 [X, Lin, Par, Rep]  (X = Col / os_factor, image domain)
    """
    axes = (0, 1, 2) if bool(meta["is_3d"]) else (0, 1)
    img = _centered_ifft(kspace_grid, axes=axes)
    return _remove_os(img, meta)


def main():
    ap = argparse.ArgumentParser(description="Siemens twix -> k-space/image dump for 129Xe Cartesian GRE")
    ap.add_argument("--dat", required=True, help="Path to Siemens .dat file")
    ap.add_argument("--outdir", required=True, help="Directory for .mat / .png outputs")
    ap.add_argument("--kspace-scale", type=float, default=1e10,
                    help="Fixed constant applied to raw k-space (default 1e10); "
                         "does not affect DL output (per-slice normalized)")
    ap.add_argument("--method", choices=["pocs", "zerofill"], default="pocs",
                    help="Partial-Fourier handling (full-Fourier data ignores this)")
    ap.add_argument("--pocs-iter", type=int, default=15, help="POCS iterations")
    ap.add_argument("--no-png", action="store_true", help="Skip PNG preview")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.dat))[0]

    print(f"Reading {args.dat}")
    out = read_kspace(args.dat, kspace_scale=args.kspace_scale)
    meta = out["meta"]
    print("Geometry:")
    for k, v in meta.items():
        print(f"    {k:28s}: {v}")
    print(f"  kspace_raw  shape [Col,Lin,Par,Rep]: {out['kspace_raw'].shape}")
    print(f"  kspace_grid shape [Col,Lin,Par,Rep]: {out['kspace_grid'].shape}")

    masks = (out["col_sampled"], out["lin_sampled"], out["par_sampled"])
    is_full = all(bool(m.all()) for m in masks)
    print("Sampling (True=acquired):"
          f" readout {int(out['col_sampled'].sum())}/{out['col_sampled'].size},"
          f" phase {int(out['lin_sampled'].sum())}/{out['lin_sampled'].size},"
          f" partition {int(out['par_sampled'].sum())}/{out['par_sampled'].size}")
    print(f"Zero-filled iFFT recon... (fully sampled on all axes: {is_full})")
    img_zf = zerofilled_recon(out["kspace_grid"], meta)

    # POCS only matters when partial Fourier is present; otherwise reuse zero-fill.
    if args.method == "pocs" and not is_full:
        print(f"POCS partial-Fourier recon ({args.pocs_iter} iters, "
              "readout + phase + partition)...")
        img_pf = pocs_recon(out["kspace_grid"], masks, meta, n_iter=args.pocs_iter)
    else:
        img_pf = img_zf
    # Reorient to a standard anatomical display (coronal: apices up, bases down)
    # so the .mat, PNG, and downstream consumers are all correctly oriented.
    img_zf = orient_for_display(img_zf, meta)
    img_pf = orient_for_display(img_pf, meta)
    # Image used downstream (models, PNG): POCS result when PF, else zero-fill.
    img = img_pf
    print(f"  image shape [X,Lin,Par,Rep]: {img.shape} (orient: {meta['normal_axis']})")

    mat_path = os.path.join(args.outdir, f"{tag}_recon.mat")
    print(f"Saving {mat_path}")
    sio.savemat(
        mat_path,
        {
            "kspace_raw": out["kspace_raw"],
            "kspace_grid": out["kspace_grid"],
            "col_sampled": out["col_sampled"],
            "lin_sampled": out["lin_sampled"],
            "par_sampled": out["par_sampled"],
            "image_zf_complex": img_zf,
            "image_zf_mag": np.abs(img_zf).astype(np.float32),
            "image_pocs_complex": img_pf,
            "image_pocs_mag": np.abs(img_pf).astype(np.float32),
            "meta": meta,
        },
        do_compression=True,
    )

    if not args.no_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mag = np.abs(img)  # [X, Lin, Par, Rep]
            # Show partitions (3D) or dynamics (2D) of the first rep.
            n_panel = mag.shape[2] if meta["is_3d"] else mag.shape[3]
            cols = int(np.ceil(np.sqrt(n_panel)))
            rows = int(np.ceil(n_panel / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
            axes = np.atleast_1d(axes).ravel()
            for i in range(len(axes)):
                axes[i].axis("off")
                if i < n_panel:
                    panel = mag[:, :, i, 0] if meta["is_3d"] else mag[:, :, 0, i]
                    axes[i].imshow(panel, cmap="gray")
                    axes[i].set_title(f"{'par' if meta['is_3d'] else 'rep'} {i}", fontsize=7)
            fig.suptitle(tag, fontsize=9)
            fig.tight_layout()
            method_tag = "pocs" if (args.method == "pocs" and not is_full) else "zf"
            png_path = os.path.join(args.outdir, f"{tag}_{method_tag}_mag.png")
            fig.savefig(png_path, dpi=120)
            print(f"Saving {png_path}")
        except Exception as e:  # noqa: BLE001
            print(f"(PNG preview skipped: {e})")

    print("Done.")


if __name__ == "__main__":
    main()
