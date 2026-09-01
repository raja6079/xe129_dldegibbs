"""Apply the ORIGINAL (wavelet U-Net) DLDeGibbs model to a Cartesian 129Xe recon.

This is the legacy-backbone counterpart of ``run_cartesian_nafnet.py``. It
consumes the same ``*_recon.mat`` produced by ``siemens_twix_recon.py`` (reading
the POCS complex volume ``image_pocs_complex`` of shape [X, Lin, Par, Rep]) and
runs the wavelet ``DeGibbs`` model (Muckley et al. DLDeGibbs) slice-by-slice.

Only a COMPLEX wavelet checkpoint is available
(``models/complex/wavelet_unet_imnet_pffact_8_snrlog0-5_nlayers_10``), so this
script runs the complex model (in_ch=2, out_ch=2, comp2mag=True) -> magnitude.

NORMALIZATION (identical to run_cartesian_nafnet.py so outputs are comparable)
-----------------------------------------------------------------------------
Each slice is normalized by its own magnitude max before inference and rescaled
afterwards. Real/imag are divided by the SAME factor so the phase is preserved:

    max_val = max(|z|)
    z_norm  = z / max_val          # phase preserved, |z| in [0,1]
    out     = model(z_norm) * max_val

The wavelet model is a ``torch.double`` network trained at a 100x100 matrix, so
inputs are Fourier zero-filled to ``--infer_size`` (default 100) just like the
NAFNet script. Anatomical orientation is inherited from ``image_pocs_complex``
(the recon applies ``orient_for_display``); like NAFNet/ARDL this script also
emits Fourier-interpolated high-res (``--final_sizes``, default 256/512) outputs.

Example:
    python run_cartesian_dldegibbs.py \
        --mat /home/raja/xe129_dldegibbs/PythonRecon/3T_vent_data_Manchester/RECON/HV03/meas_MID00041_FID204963_gre_Xe_3D_recon.mat
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio
import torch

# The legacy wavelet DeGibbs backbone lives in the denoising-project repo.
XE129_DENOISING = "/home/raja/xe129_denoising_project"
sys.path.append(XE129_DENOISING)

DEFAULT_COMPLEX_CKPT = os.path.join(
    XE129_DENOISING, "models", "complex",
    "wavelet_unet_imnet_pffact_8_snrlog0-5_nlayers_10",
)


def _fourier_resize_2d(image: np.ndarray, target_hw) -> np.ndarray:
    """Zero-fill (k-space) resize a 2D image to target_hw, preserving FOV.

    Pads/crops the centered spectrum -> sinc interpolation. Complex in -> complex
    out; real in -> real out. Same op as run_cartesian_nafnet.py so the two
    backbones see identical inputs.
    """
    src_h, src_w = image.shape
    dst_h, dst_w = target_hw
    if (src_h, src_w) == (dst_h, dst_w):
        return image.copy()

    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    out = np.zeros((dst_h, dst_w), dtype=np.complex64)
    ch, cw = min(src_h, dst_h), min(src_w, dst_w)
    sh, sw = (src_h - ch) // 2, (src_w - cw) // 2
    dh, dw = (dst_h - ch) // 2, (dst_w - cw) // 2
    out[dh:dh + ch, dw:dw + cw] = spectrum[sh:sh + ch, sw:sw + cw]
    resized = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(out, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    # Preserve image intensity across the grid-size change (compensate ifft 1/N).
    resized = resized * (float(dst_h * dst_w) / float(src_h * src_w))
    return resized.astype(np.complex64) if np.iscomplexobj(image) else resized.real.astype(np.float32)


def _resize_volume(vol4d: np.ndarray, size: int, take_abs: bool = False) -> np.ndarray:
    """Fourier-interpolate each [X,Lin] slice of a [X,Lin,Par,Rep] volume to size x size."""
    _, _, Par, Rep = vol4d.shape
    out = np.zeros((size, size, Par, Rep), dtype=np.float32)
    for r in range(Rep):
        for p in range(Par):
            rs = _fourier_resize_2d(vol4d[:, :, p, r], (size, size))
            out[:, :, p, r] = np.abs(rs).astype(np.float32) if take_abs else rs.real.astype(np.float32)
    return out


def _resolve_ckpt(path_or_dir: str) -> str:
    ckpt = os.path.join(path_or_dir, "best_model.pt") if os.path.isdir(path_or_dir) else path_or_dir
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _load_model(exp_type, ckpt_dir, nlayers, nchans, device):
    """Load a wavelet DeGibbs model (torch.double) from a checkpoint dir/file."""
    from degibbs_model import DeGibbs

    if exp_type == "complex":
        in_ch, out_ch, comp2mag = 2, 2, True
    elif exp_type == "magnitude":
        in_ch, out_ch, comp2mag = 1, 1, False
    else:
        raise ValueError(exp_type)

    model = DeGibbs(
        nlayers=nlayers, in_ch=in_ch, out_ch=out_ch, nchans=nchans,
        comp2mag=comp2mag, leaky=False, device=device, dtype=torch.double,
    )
    ckpt = _resolve_ckpt(ckpt_dir)
    model = model.load_model(model_file=ckpt, device=device, dtype=torch.double)
    print(f"Loaded {exp_type} wavelet DeGibbs model: {ckpt}")
    return model


@torch.no_grad()
def _run_complex(model, slice_complex, device):
    """Complex model: normalize by magnitude max, run, rescale. Returns magnitude."""
    max_val = float(np.max(np.abs(slice_complex)))
    if max_val < 1e-12:
        return np.zeros(slice_complex.shape, dtype=np.float32)
    z = slice_complex / max_val                        # phase preserved, |z|<=1
    inp = np.stack([np.real(z), np.imag(z)], axis=0)   # [2,H,W]
    inp = torch.from_numpy(inp).double().unsqueeze(0).to(device)   # [1,2,H,W]
    out = model(inp)                                   # [1,1,H,W] magnitude (comp2mag)
    return (out.squeeze().cpu().numpy() * max_val).astype(np.float32)


@torch.no_grad()
def _run_magnitude(model, slice_complex, device):
    """Magnitude model: feed |z| normalized by the SAME magnitude max."""
    mag = np.abs(slice_complex).astype(np.float64)
    max_val = float(mag.max())
    if max_val < 1e-12:
        return np.zeros(mag.shape, dtype=np.float32)
    inp = torch.from_numpy(mag / max_val).double().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    out = model(inp)                                   # [1,1,H,W]
    return (out.squeeze().cpu().numpy() * max_val).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Original wavelet DLDeGibbs denoising on a Cartesian recon .mat")
    ap.add_argument("--mat", required=True, help="*_recon.mat from siemens_twix_recon.py")
    ap.add_argument("--out", default=None, help="output .mat (default: <mat>_dldegibbs.mat)")
    ap.add_argument("--var", default="image_pocs_complex",
                    help="complex volume variable in the .mat (default image_pocs_complex)")
    ap.add_argument("--complex_ckpt", default=DEFAULT_COMPLEX_CKPT)
    ap.add_argument("--mag_ckpt", default=None,
                    help="magnitude wavelet checkpoint (none trained -> magnitude skipped)")
    ap.add_argument("--nlayers", type=int, default=10)
    ap.add_argument("--nchans", type=int, default=64)
    ap.add_argument("--models", default="complex", choices=["both", "complex", "magnitude"])
    ap.add_argument("--infer_size", type=int, default=100,
                    help="Fourier zero-fill the in-plane matrix to NxN before inference "
                         "(matches training matrix; default 100). Use 0 for native.")
    ap.add_argument("--final_sizes", default="256,512",
                    help="Comma-separated final resolutions; the denoised (and noisy) "
                         "images are Fourier-interpolated to each (DICOM-intent). Empty to skip.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = args.out or (os.path.splitext(args.mat)[0] + "_dldegibbs.mat")

    do_c = args.models in ("both", "complex")
    do_m = args.models in ("both", "magnitude")
    if do_m and not args.mag_ckpt:
        print("(no magnitude wavelet checkpoint available -> skipping magnitude model)")
        do_m = False
        if not do_c:
            raise SystemExit("Nothing to run: magnitude requested but no mag checkpoint.")

    print(f"Loading {args.mat}")
    mat = sio.loadmat(args.mat)
    if args.var not in mat:
        raise KeyError(f"'{args.var}' not in {args.mat}. Keys: "
                       f"{[k for k in mat if not k.startswith('__')]}")
    vol = mat[args.var]
    if vol.ndim == 3:                 # [X, Lin, Par] -> add Rep axis
        vol = vol[:, :, :, None]
    X, Lin, Par, Rep = vol.shape
    S = args.infer_size if args.infer_size and args.infer_size > 0 else None
    Xo, Lo = (S, S) if S else (X, Lin)
    print(f"Volume '{args.var}': {vol.shape} ({vol.dtype})  device={device}")
    print(f"Inference matrix: {Xo}x{Lo}" + (f" (Fourier zero-fill from {X}x{Lin})" if S else " (native)"))

    model_c = _load_model("complex", args.complex_ckpt, args.nlayers, args.nchans, device) if do_c else None
    model_m = _load_model("magnitude", args.mag_ckpt, args.nlayers, args.nchans, device) if do_m else None

    # Outputs live on the inference grid so residuals/comparisons align.
    original_mag = np.zeros((Xo, Lo, Par, Rep), dtype=np.float32)
    den_c = np.zeros((Xo, Lo, Par, Rep), dtype=np.float32) if do_c else None
    den_m = np.zeros((Xo, Lo, Par, Rep), dtype=np.float32) if do_m else None

    n = Par * Rep
    for r in range(Rep):
        for p in range(Par):
            print(f"  slice {r * Par + p + 1}/{n}", end="\r")
            sl = vol[:, :, p, r]
            if S:
                sl = _fourier_resize_2d(sl, (S, S))     # zero-fill to inference matrix
            original_mag[:, :, p, r] = np.abs(sl).astype(np.float32)
            if do_c:
                den_c[:, :, p, r] = _run_complex(model_c, sl, device)
            if do_m:
                den_m[:, :, p, r] = _run_magnitude(model_m, sl, device)
    print()

    save = {"original_mag": original_mag, "source_var": args.var,
            "infer_size": (S or X), "backbone": "wavelet_unet",
            "complex_ckpt": args.complex_ckpt, "mag_ckpt": args.mag_ckpt or ""}
    if do_c:
        save["denoised_complex_mag"] = den_c
        save["residual_complex"] = original_mag - den_c
    if do_m:
        save["denoised_magnitude"] = den_m
        save["residual_magnitude"] = original_mag - den_m

    # High-resolution (DICOM-intent) outputs: Fourier-interpolate to each final size.
    # Noisy comes from the COMPLEX recon (proper zero-filled high-res reconstruction);
    # denoised comes from the model magnitude output.
    final_sizes = [int(s) for s in args.final_sizes.split(",") if s.strip()]
    for Sf in final_sizes:
        print(f"Fourier-interpolating to {Sf}x{Sf} ...")
        save[f"noisy_{Sf}"] = _resize_volume(vol, Sf, take_abs=True)   # from complex recon
        if do_c:
            save[f"denoised_complex_{Sf}"] = _resize_volume(den_c, Sf)
        if do_m:
            save[f"denoised_magnitude_{Sf}"] = _resize_volume(den_m, Sf)

    print(f"Saving {out_path}")
    sio.savemat(out_path, save, do_compression=True)

    if not args.no_png:
        try:
            _save_compare_png(original_mag, den_c, den_m, Par, Rep,
                              os.path.splitext(out_path)[0] + ".png")
            for Sf in final_sizes:
                _save_compare_png(save[f"noisy_{Sf}"],
                                  save.get(f"denoised_complex_{Sf}"),
                                  save.get(f"denoised_magnitude_{Sf}"),
                                  Par, Rep, f"{os.path.splitext(out_path)[0]}_{Sf}.png")
        except Exception as e:  # noqa: BLE001
            print(f"(PNG skipped: {e})")
    print("Done.")


def _save_compare_png(orig, den_c, den_m, Par, Rep, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick up to 5 representative slices by total magnitude energy.
    frames = orig.reshape(orig.shape[0], orig.shape[1], -1)   # [X, Lin, Par*Rep]
    energy = frames.reshape(-1, frames.shape[2]).sum(0)
    order = np.argsort(energy)[::-1]
    picks = sorted(order[:min(5, len(order))])

    rows = [("input", orig)]
    if den_c is not None:
        rows.append(("dldegibbs-complex", den_c))
    if den_m is not None:
        rows.append(("dldegibbs-magnitude", den_m))

    n_col = len(picks)
    fig, axes = plt.subplots(len(rows), n_col, figsize=(2.4 * n_col, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for ci, idx in enumerate(picks):
        p, r = idx % Par, idx // Par
        for ri, (name, volm) in enumerate(rows):
            ax = axes[ri, ci]
            ax.imshow(volm[:, :, p, r], cmap="gray")
            ax.axis("off")
            if ri == 0:
                ax.set_title(f"par {p}, rep {r}", fontsize=7)
            if ci == 0:
                ax.set_ylabel(name, fontsize=8)
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"Saving {png_path}")


if __name__ == "__main__":
    main()
