"""Visualize DeGibbs model results on val/test data.

Loads N random samples from the validation set, runs the model, and saves
a PNG with columns: Gibbs-corrupted input | Ground-truth target | Model output | Difference

Usage
-----
# Use val set (default), 6 samples, current best checkpoint:
python visualize_degibbs.py

# More samples, specific output file:
python visualize_degibbs.py --n_samples 8 --output results/degibbs_vis.png

# Use test set:
python visualize_degibbs.py --test

# Specific checkpoint:
python visualize_degibbs.py --checkpoint /path/to/model_epoch_5.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_factory import build_model
from data.degibbs_loaders import init_loaders

DEFAULT_CKPT = os.path.expanduser(
    "~/xe129_denoising_project/experiments_dldegibbs/complex/"
    "nafnet_small_imnet_pffact_8_snrlog0-5/best_model.pt"
)
DATA_DIR   = os.path.expanduser("~/xe129_denoising_project/data_dldegibbs/")
TEST_DIR   = os.path.join(DATA_DIR, "test/")


def load_model(ckpt_path, device):
    model = build_model(
        model="nafnet", in_ch=2, out_ch=2, comp2mag=True,
        nafnet_config="small", noise_map_conditioning=False,
    ).to(device, dtype=torch.float32)
    model.eval()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded: {ckpt_path}")
    return model


def to_mag(tensor):
    """Convert [C, H, W] tensor to magnitude numpy array."""
    t = tensor.cpu().float().numpy()
    if t.shape[0] == 2:
        return np.sqrt(t[0] ** 2 + t[1] ** 2)
    return np.abs(t[0])


def percentile_clip(img, plo=1, phi=99):
    lo, hi = np.percentile(img, plo), np.percentile(img, phi)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


@torch.inference_mode()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    # Silence the pe_fact / dataset-size prints from init_loaders
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, val_loader = init_loaders(
            exp_type="complex",
            data_dir=DATA_DIR,
            batch_size=1,
            num_workers=2,
            gibb_size=100,
            orig_big=256,
            pf_fact=8,
            pe_fact=1,
            test=args.test,
            test_dir=TEST_DIR,
            dtype=torch.float32,
        )

    # Gather N samples
    samples = []
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(val_loader.dataset),
                         size=args.n_samples, replace=False)
    for idx in sorted(indices):
        sample = val_loader.dataset[int(idx)]
        samples.append(sample)

    # Build figure: rows = samples, cols = [Input mag | Target | Output | Diff]
    ncols = 4
    nrows = len(samples)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3, nrows * 3),
        squeeze=False,
    )

    col_titles = ["Input (Gibbs)", "Target (GT)", "Model output", "Diff (×5)"]
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=10, fontweight="bold")

    for row, sample in enumerate(samples):
        dat    = sample["dat"]     # [2, H, W] float32 tensor
        target = sample["target"]  # [1, H, W] float32 tensor

        dat_t  = dat.unsqueeze(0).to(device)   # [1, 2, H, W]
        with torch.inference_mode():
            est = model(dat_t)[0]              # [1, H, W]

        inp_mag  = to_mag(dat)
        targ_mag = to_mag(target)
        est_mag  = to_mag(est)
        diff     = np.abs(est_mag - targ_mag)

        # Normalize all panels to same scale (target max) for fair comparison
        scale = targ_mag.max() if targ_mag.max() > 1e-8 else 1.0

        panels = [inp_mag / scale, targ_mag / scale,
                  est_mag / scale, diff * 5 / scale]
        cmaps  = ["gray", "gray", "gray", "hot"]
        vmaxes = [1.0, 1.0, 1.0, None]

        for col, (panel, cmap, vmax) in enumerate(zip(panels, cmaps, vmaxes)):
            ax = axes[row][col]
            im = ax.imshow(panel, cmap=cmap, vmin=0, vmax=vmax,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")

        # Row label: PSNR
        psnr = 10 * np.log10(1.0 / (np.mean((est_mag / scale
                                              - targ_mag / scale) ** 2) + 1e-12))
        axes[row][0].set_ylabel(f"PSNR={psnr:.1f} dB", fontsize=8, rotation=90,
                                labelpad=4)
        axes[row][0].axis("on")
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])

    split = "test" if args.test else "val"
    fig.suptitle(
        f"NAFNet DeGibbs — {split} samples  |  "
        f"checkpoint: {os.path.basename(args.checkpoint)}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Visualize DeGibbs results")
    p.add_argument("--n_samples",   type=int, default=6)
    p.add_argument("--output",      default="degibbs_results.png")
    p.add_argument("--checkpoint",  default=DEFAULT_CKPT)
    p.add_argument("--test",        action="store_true",
                   help="Use test set instead of val set")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
