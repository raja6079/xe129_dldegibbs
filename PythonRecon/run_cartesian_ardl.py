"""Apply the ARDL (Adaptive Recon DL) TensorFlow model to a Cartesian 129Xe recon.

This is the ARDL counterpart of ``run_cartesian_nafnet.py`` /
``run_cartesian_dldegibbs.py``. It consumes the same ``*_recon.mat`` produced by
``siemens_twix_recon.py`` (reading the POCS complex volume
``image_pocs_complex`` of shape [X, Lin, Par, Rep]) and runs the ARDL model
**slice-by-slice** for de-ringing + denoising.

WHICH MODEL
-----------
Per the request, the **3D** SavedModel (``ardl_models/test-c3d``) is used for BOTH
the 3D volumes (Par > 1) and the 2D-dynamic data (Par == 1). ARDL is a 2D
per-slice network internally (it predicts a ringing residual and a noise residual
for a single [H, W] complex slice); the "c2d"/"c3d" distinction is just which
trained weights are used. Here every in-plane [X, Lin] slice of every (Par, Rep)
is run independently through the c3d model (no tri-planar combination), matching
the ``--models``/per-slice loop layout of the NAFNet/wavelet front-ends so the
three outputs are directly comparable.

LOADING (TF-version-robust)
---------------------------
The SavedModel was exported by an older TF/Keras and carries optimizer state that
newer TF's object-graph loader (``tf.saved_model.load`` / Keras 3 ``TFSMLayer``)
refuses to restore. We instead load the *frozen* ``serving_default`` meta-graph in
a TF-v1 session and run its named tensors directly (inference only). The serving
signature exposes one input (``inputs``, [N,H,W,2]) and two named outputs
(``noise`` and ``ring``, each [N,H,W,2] = real/imag).

NORMALIZATION (the ARDL convention)
-----------------------------------
ARDL was trained on inputs scaled so the peak magnitude is ~128 (the DL-Recon-IQ
convention; see ``cartesian3d_ardl.py`` -> ``rescale = maxImage/128``). Each slice
is therefore scaled by its OWN magnitude max to hit that peak, run, then rescaled
back. Real/imag are scaled by the SAME factor so phase is preserved:

    rescale = max(|z|) / 128
    z_s     = z / rescale                         # peak |z_s| ~ 128
    ring, noise = model(z_s)                       # complex residuals
    clean   = z_s - dering*ring - denoise*noise
    out     = |clean| * rescale                    # physical-intensity magnitude

With the defaults (dering = denoise = 1.0, as in cartesian3d_ardl.py) the two
residuals are simply summed, so the noise/ring output ordering is irrelevant; the
named-output mapping matters only if you set the two levels differently.

Example:
    python run_cartesian_ardl.py \
        --mat /home/raja/xe129_dldegibbs/PythonRecon/3T_vent_data_Manchester/RECON/HV03/meas_MID00041_FID204963_gre_Xe_3D_recon.mat
"""

import argparse
import os

import numpy as np
import scipy.io as sio

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF banner
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU model; keep off the GPUs

DEFAULT_C3D = "/home/raja/xe129_denoising_project/ardl_models/test-c3d"


def _fourier_resize_2d(image: np.ndarray, target_hw) -> np.ndarray:
    """Zero-fill (k-space) resize a 2D image to target_hw, preserving FOV.

    Pads/crops the centered spectrum -> sinc interpolation. Complex in -> complex
    out; real in -> real out. Identical op to run_cartesian_nafnet.py so the
    backbones see comparable inputs (only the target matrix differs).
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


class ARDLModel:
    """Inference-only wrapper around an ARDL TensorFlow SavedModel.

    Loads the frozen ``serving_default`` meta-graph into a TF-v1 session and runs
    the named ``noise``/``ring`` outputs for a batch of [H,W,2] slices.
    """

    def __init__(self, saved_model_dir: str):
        import tensorflow as tf  # imported here so --help stays fast

        tf.compat.v1.disable_eager_execution()
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        with self._sess.graph.as_default():
            meta = tf.compat.v1.saved_model.loader.load(
                self._sess, [tf.saved_model.SERVING], saved_model_dir)
        sig = meta.signature_def["serving_default"]
        self._in = list(sig.inputs.values())[0].name          # 'serving_default_inputs:0'
        # Map by signature key so the two residuals are unambiguous.
        self._out_noise = sig.outputs["noise"].name if "noise" in sig.outputs else None
        self._out_ring = sig.outputs["ring"].name if "ring" in sig.outputs else None
        if self._out_noise is None or self._out_ring is None:
            # Fall back to positional order (older exports may name them differently).
            names = [v.name for v in sig.outputs.values()]
            self._out_noise, self._out_ring = names[0], names[1]
        print(f"Loaded ARDL model: {saved_model_dir}")
        print(f"  input={self._in}  noise={self._out_noise}  ring={self._out_ring}")

    def predict(self, batch_hw2: np.ndarray):
        """batch_hw2: [N,H,W,2] float32 -> (noise, ring) each [N,H,W,2]."""
        noise, ring = self._sess.run(
            [self._out_noise, self._out_ring],
            feed_dict={self._in: batch_hw2.astype(np.float32)})
        return noise, ring


def _run_ardl_slice(model, slice_complex, denoise=1.0, dering=1.0):
    """Normalize to ARDL's ~128 peak, predict residuals, subtract, return magnitude."""
    max_val = float(np.max(np.abs(slice_complex)))
    if max_val < 1e-12:
        return np.zeros(slice_complex.shape, dtype=np.float32)
    rescale = max_val / 128.0
    z = slice_complex / rescale                                   # peak |z| ~ 128
    inp = np.stack([np.real(z), np.imag(z)], axis=-1)[None]       # [1,H,W,2]
    noise, ring = model.predict(inp)
    noise_c = noise[0, :, :, 0] + 1j * noise[0, :, :, 1]
    ring_c = ring[0, :, :, 0] + 1j * ring[0, :, :, 1]
    clean = z - dering * ring_c - denoise * noise_c
    return (np.abs(clean) * rescale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="ARDL denoising/de-ringing on a Cartesian recon .mat")
    ap.add_argument("--mat", required=True, help="*_recon.mat from siemens_twix_recon.py")
    ap.add_argument("--out", default=None, help="output .mat (default: <mat>_ardl.mat)")
    ap.add_argument("--var", default="image_pocs_complex",
                    help="complex volume variable in the .mat (default image_pocs_complex)")
    ap.add_argument("--model_dir", default=DEFAULT_C3D,
                    help="ARDL SavedModel dir (default: the c3d model, used for 2D and 3D)")
    ap.add_argument("--denoise_level", type=float, default=1.0)
    ap.add_argument("--dering_level", type=float, default=1.0)
    ap.add_argument("--infer_size", type=int, default=128,
                    help="Fourier zero-fill the in-plane matrix to NxN before inference. "
                         "ARDL needs a multiple of 8; default 128 (~2x the native 64). "
                         "Use 0 for native (must already be a multiple of 8).")
    ap.add_argument("--final_sizes", default="256,512",
                    help="Comma-separated final resolutions; denoised (and noisy) images "
                         "are Fourier-interpolated to each (DICOM-intent). Empty to skip.")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    out_path = args.out or (os.path.splitext(args.mat)[0] + "_ardl.mat")

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
    if S is not None and S % 8 != 0:
        S = ((S + 7) // 8) * 8
        print(f"(infer_size rounded up to multiple of 8: {S})")
    if S is None and (X % 8 or Lin % 8):
        raise SystemExit(f"Native matrix {X}x{Lin} is not a multiple of 8; "
                         f"pass --infer_size (e.g. 128).")
    Xo, Lo = (S, S) if S else (X, Lin)
    print(f"Volume '{args.var}': {vol.shape} ({vol.dtype})")
    print(f"Inference matrix: {Xo}x{Lo}" + (f" (Fourier zero-fill from {X}x{Lin})" if S else " (native)"))
    print(f"Mode: per-slice c3d ({'3D volume, Par>1' if Par > 1 else '2D data, Par==1'}); "
          f"denoise={args.denoise_level} dering={args.dering_level}")

    model = ARDLModel(args.model_dir)

    original_mag = np.zeros((Xo, Lo, Par, Rep), dtype=np.float32)
    den = np.zeros((Xo, Lo, Par, Rep), dtype=np.float32)

    n = Par * Rep
    for r in range(Rep):
        for p in range(Par):
            print(f"  slice {r * Par + p + 1}/{n}", end="\r")
            sl = vol[:, :, p, r]
            if S:
                sl = _fourier_resize_2d(sl, (S, S))     # zero-fill to inference matrix
            original_mag[:, :, p, r] = np.abs(sl).astype(np.float32)
            den[:, :, p, r] = _run_ardl_slice(model, sl, args.denoise_level, args.dering_level)
    print()

    save = {"original_mag": original_mag, "denoised_ardl_mag": den,
            "residual_ardl": original_mag - den, "source_var": args.var,
            "infer_size": (S or X), "backbone": "ardl_c3d", "model_dir": args.model_dir,
            "denoise_level": args.denoise_level, "dering_level": args.dering_level}

    # High-resolution (DICOM-intent) outputs: Fourier-interpolate to each final size.
    final_sizes = [int(s) for s in args.final_sizes.split(",") if s.strip()]
    for Sf in final_sizes:
        print(f"Fourier-interpolating to {Sf}x{Sf} ...")
        save[f"noisy_{Sf}"] = _resize_volume(vol, Sf, take_abs=True)
        save[f"denoised_ardl_{Sf}"] = _resize_volume(den, Sf)

    print(f"Saving {out_path}")
    sio.savemat(out_path, save, do_compression=True)

    if not args.no_png:
        try:
            _save_compare_png(original_mag, den, Par, Rep,
                              os.path.splitext(out_path)[0] + ".png")
            for Sf in final_sizes:
                _save_compare_png(save[f"noisy_{Sf}"], save[f"denoised_ardl_{Sf}"],
                                  Par, Rep, f"{os.path.splitext(out_path)[0]}_{Sf}.png")
        except Exception as e:  # noqa: BLE001
            print(f"(PNG skipped: {e})")
    print("Done.")


def _save_compare_png(orig, den, Par, Rep, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick up to 5 representative slices by total magnitude energy.
    frames = orig.reshape(orig.shape[0], orig.shape[1], -1)   # [X, Lin, Par*Rep]
    energy = frames.reshape(-1, frames.shape[2]).sum(0)
    order = np.argsort(energy)[::-1]
    picks = sorted(order[:min(5, len(order))])

    rows = [("input", orig), ("ardl", den)]
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
