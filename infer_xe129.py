"""Inference script for 129Xe Gibbs artifact correction.

Applies the trained NAFNet DeGibbs model to a 3D complex 129Xe volume,
processing it slice-by-slice (axial by default).

Input:  complex 3D volume  -- numpy .npy/.npz, or NIfTI (.nii/.nii.gz)
Output: real-valued 3D magnitude volume (Gibbs corrected), same format.

Usage examples
--------------
# NumPy complex array (.npy):
python infer_xe129.py --input xe129_complex.npy --output xe129_degibbs.npy

# Separate real/imag arrays in .npz:
python infer_xe129.py --input xe129_data.npz --real_key re --imag_key im \
    --output xe129_degibbs.npz

# NIfTI (4D: X,Y,Z,2 where last dim = real,imag):
python infer_xe129.py --input xe129.nii.gz --output xe129_degibbs.nii.gz

# Override checkpoint:
python infer_xe129.py --input xe129.npy --output out.npy \
    --checkpoint /path/to/best_model.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add project root to path so imports work when called from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_factory import build_model

DEFAULT_CKPT = os.path.expanduser(
    "~/xe129_denoising_project/experiments_dldegibbs/complex/"
    "nafnet_small_imnet_pffact_8_snrlog0-5/best_model.pt"
)


def load_model(checkpoint_path, device, arch_variant="v1"):
    model = build_model(
        model="nafnet", in_ch=2, out_ch=2, comp2mag=True,
        nafnet_config="small", noise_map_conditioning=False,
        arch_variant=arch_variant,
    ).to(device, dtype=torch.float32)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip DDP and torch.compile prefixes if present
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model




def preprocess_slice(real_sl, imag_sl):
    """Max-normalize and pack into [1, 2, H, W] float32 tensor.

    Matches MriNormalize(percentile_norm='Max') used in training.
    """
    mag = np.sqrt(real_sl ** 2 + imag_sl ** 2)
    max_val = mag.max()
    if max_val < 1e-10:
        max_val = 1.0
    stacked = np.stack([real_sl / max_val, imag_sl / max_val], axis=0).astype(np.float32)
    return torch.from_numpy(stacked).unsqueeze(0), max_val  # [1,2,H,W], scalar


def load_input(args):
    """Returns (real_vol, imag_vol) float32 numpy arrays, shape (Z, H, W)."""
    path = args.input
    ext = path.lower()

    if ext.endswith(".npy"):
        arr = np.load(path)
        if np.iscomplexobj(arr):
            # (H,W,Z) -> (Z,H,W) if needed
            if arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
                arr = arr.transpose(2, 0, 1)
            return arr.real.astype(np.float32), arr.imag.astype(np.float32)
        if arr.shape[-1] == 2:
            return arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32)
        if arr.shape[0] == 2:
            return arr[0].astype(np.float32), arr[1].astype(np.float32)
        raise ValueError(f"Cannot interpret shape {arr.shape} as complex volume")

    if ext.endswith(".npz"):
        data = np.load(path)
        rk = args.real_key or "real"
        ik = args.imag_key or "imag"
        return data[rk].astype(np.float32), data[ik].astype(np.float32)

    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        import nibabel as nib
        arr = nib.load(path).get_fdata()
        if np.iscomplexobj(arr):
            arr = arr.transpose(2, 0, 1)  # (X,Y,Z)->(Z,X,Y)
            return arr.real.astype(np.float32), arr.imag.astype(np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 2:
            # (X,Y,Z,2) -> (Z,X,Y)
            return (arr[..., 0].transpose(2, 0, 1).astype(np.float32),
                    arr[..., 1].transpose(2, 0, 1).astype(np.float32))
        raise ValueError(f"NIfTI shape {arr.shape}: expected complex or (X,Y,Z,2)")

    raise ValueError(f"Unsupported format: {path}")


def save_output(output_vol, args):
    """Save (Z, H, W) float32 magnitude volume."""
    out = args.output
    ext = out.lower()
    if ext.endswith(".npy"):
        np.save(out, output_vol)
    elif ext.endswith(".npz"):
        np.savez(out, degibbs=output_vol)
    elif ext.endswith(".nii") or ext.endswith(".nii.gz"):
        import nibabel as nib
        affine = np.eye(4)
        if args.input.lower().endswith((".nii", ".nii.gz")):
            affine = nib.load(args.input).affine
        nib.save(nib.Nifti1Image(output_vol.transpose(1, 2, 0), affine), out)
    else:
        np.save(out, output_vol)
    print(f"Saved: {out}  shape={output_vol.shape}")


@torch.inference_mode()
def run_inference(model, real_vol, imag_vol, device, slice_axis=0):
    """Process volume slice-by-slice along slice_axis, return (Z, H, W)."""
    axes = list(range(real_vol.ndim))
    axes.insert(0, axes.pop(slice_axis))
    real_vol = np.transpose(real_vol, axes)
    imag_vol = np.transpose(imag_vol, axes)

    n = real_vol.shape[0]
    out_slices = []
    for i in range(n):
        t, _ = preprocess_slice(real_vol[i], imag_vol[i])
        out = model(t.to(device))             # [1, 1, H, W] — NAFNet pads internally
        sl  = out[0, 0].cpu().numpy()         # already cropped back by NAFNet
        out_slices.append(sl)
        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(f"  slice {i+1}/{n}  range=[{sl.min():.4f}, {sl.max():.4f}]")

    return np.stack(out_slices, axis=0)  # (Z, H, W)


def main():
    p = argparse.ArgumentParser(description="129Xe DeGibbs inference")
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--slice_axis", type=int, default=0,
                   help="Axis to iterate over (0=Z, 1=Y, 2=X)")
    p.add_argument("--real_key",   default=None, help=".npz key for real part")
    p.add_argument("--imag_key",   default=None, help=".npz key for imag part")
    p.add_argument("--cpu",        action="store_true")
    p.add_argument("--arch_variant", default="v1", choices=["v1","v2"],
                   help="NAFNet arch: v1 (legacy) or v2 (pixel-unshuffle + concat skip)")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    model  = load_model(args.checkpoint, device, args.arch_variant)
    print(f"Loading: {args.input}")
    real_v, imag_v = load_input(args)
    print(f"  shape={real_v.shape}")

    print("Running inference...")
    out = run_inference(model, real_v, imag_v, device, args.slice_axis)
    save_output(out, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
