"""Compare ARDL vs NAFNet (and wavelet DLDeGibbs) denoising on the Manchester data.

Pairs, per dataset, the outputs written next to each ``*_recon.mat``:

    <base>_denoised.mat   (run_cartesian_nafnet.py) : denoised_complex_mag,
                                                       denoised_magnitude  (100x100)
    <base>_ardl.mat       (run_cartesian_ardl.py)   : denoised_ardl_mag    (128x128)
    <base>_dldegibbs.mat  (optional, wavelet)       : denoised_complex_mag (100x100)

Each method carries its own ``original_mag`` (the zero-filled recon on that
method's inference grid) and ``residual_*``. There is NO ground truth for real
129Xe data, so we report **no-reference** metrics:

  * bg_std   - noise std in a background ring (outer frame). Lower = cleaner.
  * snr      - signal_mean / bg_std, signal mask = Otsu on the INPUT. Higher.
  * sig_ret  - denoised signal_mean / input signal_mean. ~1 = signal preserved,
               <1 = signal being suppressed along with noise.

Metrics are computed on each method's NATIVE grid (mask from that method's own
input, so the input->output gain is self-consistent). For the side-by-side PNGs
every image is Fourier-resized to a common 256 grid so the panels align; the
underlying inference matrices differ (NAFNet 100, ARDL 128) and that is a real
pipeline difference, not something normalized away.

Usage:
    python compare_nafnet_ardl.py                       # all paired datasets
    python compare_nafnet_ardl.py --subjects HV03 HV07  # subset
    python compare_nafnet_ardl.py --out_dir compare_out # where PNGs/CSV land
"""

import argparse
import csv
import glob
import os

import numpy as np
import scipy.io as sio

DISPLAY = 256


def _fourier_resize_2d(image, target_hw):
    src_h, src_w = image.shape
    dst_h, dst_w = target_hw
    if (src_h, src_w) == (dst_h, dst_w):
        return image.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image), norm=None))
    out = np.zeros((dst_h, dst_w), dtype=np.complex64)
    ch, cw = min(src_h, dst_h), min(src_w, dst_w)
    sh, sw = (src_h - ch) // 2, (src_w - cw) // 2
    dh, dw = (dst_h - ch) // 2, (dst_w - cw) // 2
    out[dh:dh + ch, dw:dw + cw] = spectrum[sh:sh + ch, sw:sw + cw]
    resized = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(out)))
    resized = resized * (float(dst_h * dst_w) / float(src_h * src_w))
    return np.abs(resized).astype(np.float32)


def _otsu(vals):
    """Otsu threshold on a 1D array of non-negative intensities."""
    hist, edges = np.histogram(vals, bins=256)
    hist = hist.astype(np.float64)
    w = np.cumsum(hist)
    total = w[-1]
    if total == 0:
        return float(vals.max())
    centers = (edges[:-1] + edges[1:]) / 2
    wsum = np.cumsum(hist * centers)
    mean_total = wsum[-1] / total
    w0 = w
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = wsum / w0
        m1 = (wsum[-1] - wsum) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = 0
    return float(centers[int(np.argmax(between))])


def _bg_ring_mask(shape, frac=0.10):
    """Boolean mask of the outer frame (background for centrally-located lungs)."""
    h, w = shape
    t = max(2, int(round(min(h, w) * frac)))
    m = np.zeros(shape, dtype=bool)
    m[:t, :] = m[-t:, :] = m[:, :t] = m[:, -t:] = True
    return m


def _metrics(input_mag, out_mag):
    """No-reference metrics for a single 2D slice pair (same grid)."""
    thr = _otsu(input_mag.ravel())
    sig = input_mag > thr
    bg = _bg_ring_mask(input_mag.shape) & ~sig
    if bg.sum() < 8 or sig.sum() < 8:
        return None
    bg_std = float(out_mag[bg].std())
    sig_mean_out = float(out_mag[sig].mean())
    sig_mean_in = float(input_mag[sig].mean())
    snr = sig_mean_out / bg_std if bg_std > 1e-9 else float("nan")
    sig_ret = sig_mean_out / sig_mean_in if sig_mean_in > 1e-9 else float("nan")
    return dict(bg_std=bg_std, snr=snr, sig_ret=sig_ret)


def _agg_metrics(input_vol, out_vol):
    """Average per-slice metrics over a [H,W,Par,Rep] volume pair."""
    _, _, Par, Rep = input_vol.shape
    acc = []
    for r in range(Rep):
        for p in range(Par):
            m = _metrics(input_vol[:, :, p, r], out_vol[:, :, p, r])
            if m:
                acc.append(m)
    if not acc:
        return None
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def _load_method(mat_path, input_key, den_keys):
    """Return (input_vol, {label: den_vol}) as [H,W,Par,Rep] float32, or None."""
    if not os.path.isfile(mat_path):
        return None
    m = sio.loadmat(mat_path)
    if input_key not in m:
        return None
    inp = m[input_key].astype(np.float32)
    if inp.ndim == 3:
        inp = inp[:, :, :, None]
    outs = {}
    for label, key in den_keys:
        if key in m:
            v = m[key].astype(np.float32)
            if v.ndim == 3:
                v = v[:, :, :, None]
            outs[label] = v
    return inp, outs


def _pick_slices(vol, k=4):
    frames = vol.reshape(vol.shape[0], vol.shape[1], -1)
    energy = frames.reshape(-1, frames.shape[2]).sum(0)
    order = np.argsort(energy)[::-1]
    return sorted(order[:min(k, len(order))].tolist())


def _panel(methods, Par, out_png):
    """methods: list of (label, input_vol, out_vol). Rows=methods, cols=slices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_in = methods[0][1]
    picks = _pick_slices(ref_in)
    rows = [("input", ref_in)] + [(lbl, ov) for lbl, _iv, ov in methods]
    n_col = len(picks)
    fig, axes = plt.subplots(len(rows), n_col, figsize=(2.4 * n_col, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for ci, idx in enumerate(picks):
        p, r = idx % Par, idx // Par
        for ri, (name, volm) in enumerate(rows):
            ax = axes[ri, ci]
            sl = volm[:, :, p % volm.shape[2], r % volm.shape[3]]
            ax.imshow(_fourier_resize_2d(sl, (DISPLAY, DISPLAY)), cmap="gray")
            ax.axis("off")
            if ri == 0:
                ax.set_title(f"par {p}, rep {r}", fontsize=7)
            if ci == 0:
                ax.set_ylabel(name, fontsize=9)
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Compare ARDL vs NAFNet denoising")
    root = "/home/raja/xe129_dldegibbs/PythonRecon/3T_vent_data_Manchester/RECON"
    ap.add_argument("--recon_root", default=root)
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subject folders")
    ap.add_argument("--out_dir", default=os.path.join(root, "..", "COMPARE"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recons = sorted(glob.glob(os.path.join(args.recon_root, "*", "*_recon.mat")))
    if args.subjects:
        recons = [f for f in recons if os.path.basename(os.path.dirname(f)) in args.subjects]

    rows = []
    for recon in recons:
        base = recon[:-len("_recon.mat")]
        subj = os.path.basename(os.path.dirname(recon))
        tag = os.path.basename(base)

        naf = _load_method(base + "_denoised.mat", "original_mag",
                           [("nafnet-complex", "denoised_complex_mag"),
                            ("nafnet-magnitude", "denoised_magnitude")])
        ardl = _load_method(base + "_ardl.mat", "original_mag",
                            [("ardl", "denoised_ardl_mag")])
        wav = _load_method(base + "_dldegibbs.mat", "original_mag",
                           [("wavelet", "denoised_complex_mag")])
        if naf is None or ardl is None:
            print(f"skip {subj}/{tag} (missing nafnet or ardl output)")
            continue

        print(f"=== {subj}/{tag}")
        methods = []
        # NAFNet input baseline
        naf_in, naf_outs = naf
        ardl_in, ardl_outs = ardl
        # metrics for each method against its OWN input
        m_in = _agg_metrics(naf_in, naf_in)
        for lbl, ov in naf_outs.items():
            mm = _agg_metrics(naf_in, ov)
            rows.append(dict(subject=subj, tag=tag, method=lbl, **(mm or {})))
            methods.append((lbl, naf_in, ov))
        if wav is not None:
            wav_in, wav_outs = wav
            for lbl, ov in wav_outs.items():
                mm = _agg_metrics(wav_in, ov)
                rows.append(dict(subject=subj, tag=tag, method=lbl, **(mm or {})))
                methods.append((lbl, wav_in, ov))
        for lbl, ov in ardl_outs.items():
            mm = _agg_metrics(ardl_in, ov)
            rows.append(dict(subject=subj, tag=tag, method=lbl, **(mm or {})))
            methods.append((lbl, ardl_in, ov))
        # input baseline row (from nafnet grid)
        rows.append(dict(subject=subj, tag=tag, method="input", **(m_in or {})))

        Par = naf_in.shape[2]
        _panel(methods, Par, os.path.join(args.out_dir, f"{subj}__{tag}__compare.png"))

    # write CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    fields = ["subject", "tag", "method", "bg_std", "snr", "sig_ret"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {csv_path}")

    # summary: mean over datasets per method
    methods = sorted({r["method"] for r in rows})
    print("\n==== MEAN METRICS ACROSS DATASETS (no ground truth) ====")
    print(f"{'method':>18} | {'bg_std':>10} | {'snr':>8} | {'sig_ret':>8}")
    print("-" * 54)
    for meth in ["input", "wavelet", "nafnet-complex", "nafnet-magnitude", "ardl"]:
        rs = [r for r in rows if r["method"] == meth and "snr" in r and r.get("snr") == r.get("snr")]
        if not rs:
            continue
        bg = np.mean([r["bg_std"] for r in rs])
        snr = np.mean([r["snr"] for r in rs])
        ret = np.mean([r["sig_ret"] for r in rs])
        print(f"{meth:>18} | {bg:10.4f} | {snr:8.2f} | {ret:8.3f}")


if __name__ == "__main__":
    main()
