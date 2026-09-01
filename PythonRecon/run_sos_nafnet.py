import argparse
import importlib.util
import os
import sys

import numpy as np
import scipy.io as sio
import torch


NAFNET_CORE_PATH = "/home/raja/xe129_dldegibbs/nafnet.py"
DEFAULT_CHECKPOINT = os.path.join(
    "/home/raja/xe129_denoising_project/experiments_dldegibbs/complex_spiral",
    "nafnet_small_v2_charbonnier_imnet_pffact_8_snrlog0-5",
)


def _load_nafnet_class(core_path: str):
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"NAFNet core file not found: {core_path}")

    spec = importlib.util.spec_from_file_location("dldegibbs_nafnet_core", core_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import NAFNet core from {core_path}")
    spec.loader.exec_module(module)

    if not hasattr(module, "NAFNetDeGibbs"):
        raise AttributeError("NAFNetDeGibbs class not found in core nafnet.py")
    return module.NAFNetDeGibbs


def _resolve_checkpoint_path(path_or_dir: str) -> str:
    if os.path.isdir(path_or_dir):
        ckpt = os.path.join(path_or_dir, "best_model.pt")
    else:
        ckpt = path_or_dir
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    return ckpt


def _fourier_resize_2d(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = image.shape
    dst_h, dst_w = target_hw
    if src_h == dst_h and src_w == dst_w:
        return image.copy()

    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image, axes=(0, 1)), axes=(0, 1)),
                               axes=(0, 1))
    resized_spectrum = np.zeros((dst_h, dst_w), dtype=np.complex64)

    copy_h = min(src_h, dst_h)
    copy_w = min(src_w, dst_w)

    src_h_start = (src_h - copy_h) // 2
    src_w_start = (src_w - copy_w) // 2
    dst_h_start = (dst_h - copy_h) // 2
    dst_w_start = (dst_w - copy_w) // 2

    resized_spectrum[
        dst_h_start:dst_h_start + copy_h,
        dst_w_start:dst_w_start + copy_w,
    ] = spectrum[
        src_h_start:src_h_start + copy_h,
        src_w_start:src_w_start + copy_w,
    ]

    resized = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(resized_spectrum, axes=(0, 1)), axes=(0, 1)),
                                axes=(0, 1))
    
    if np.iscomplexobj(image):
        return resized.astype(np.complex64)
    return resized.real.astype(np.float32)


def _resize_2d(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    return _fourier_resize_2d(image, target_hw)


def _load_complex_stack(mat_file: str) -> tuple[str, np.ndarray]:
    mat_contents = sio.loadmat(mat_file)

    img_key = None
    for key, val in mat_contents.items():
        if key.startswith("__"):
            continue
        if isinstance(val, np.ndarray) and np.iscomplexobj(val) and val.ndim == 3:
            img_key = key
            break

    if img_key is None:
        raise ValueError(
            "No 3D complex array found in input .mat. "
            "Expected shape [H, W, Slices]."
        )

    raw_data = mat_contents[img_key]
    if raw_data.ndim != 3:
        raise ValueError(f"Expected 3D input [H, W, Slices], got {raw_data.shape}")

    return img_key, raw_data


def _load_model(checkpoint: str, nafnet_config: str, device: torch.device) -> torch.nn.Module:
    sys.path.append("/home/raja/xe129_dldegibbs")
    NAFNetDeGibbs = _load_nafnet_class(NAFNET_CORE_PATH)
    model = NAFNetDeGibbs.from_config(
        nafnet_config,
        in_ch=2,
        out_ch=2,
        comp2mag=True,
        noise_map_conditioning=False,
        arch_variant = 'v2'
    ).to(device, dtype=torch.float32)
    model.eval()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


@torch.inference_mode()
def run_inference(
    mat_file: str,
    checkpoint_path: str,
    output_file: str,
    nafnet_config: str = "small",
    resize_hw: tuple[int, int] | None = None,
    cpu: bool = False,
):
    print(f"Loading {mat_file}...")
    img_key, raw_data = _load_complex_stack(mat_file)
    H, W, S = raw_data.shape
    print(f"Using variable '{img_key}' with shape {raw_data.shape} ({raw_data.dtype})")

    if resize_hw is None:
        target_h, target_w = H, W
    else:
        target_h, target_w = resize_hw

    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _resolve_checkpoint_path(checkpoint_path)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print("Initializing complex spiral NAFNet model...")
    model = _load_model(checkpoint, nafnet_config, device)

    denoised = np.zeros((H, W, S), dtype=np.float32)
    denoised_256 = np.zeros((256, 256, S), dtype=np.float32)
    denoised_512 = np.zeros((512, 512, S), dtype=np.float32)

    for z in range(S):
        print(f"Processing slice {z + 1}/{S}...", end="\r")

        slice_orig = raw_data[:, :, z]
        slice_interp = _resize_2d(slice_orig, (target_h, target_w))

        max_val = float(np.max(np.abs(slice_interp)))
        if max_val < 1e-10:
            max_val = 1.0

        slice_scaled = slice_interp / max_val
        real_part = np.real(slice_scaled).astype(np.float32)
        imag_part = np.imag(slice_scaled).astype(np.float32)

        inp = np.stack([real_part, imag_part], axis=0)
        inp = torch.from_numpy(inp).unsqueeze(0).to(device)

        out = model(inp)
        out_slice = out[0, 0].detach().cpu().numpy()
        out_slice = out_slice * max_val

        out_slice = _resize_2d(out_slice, (H, W)).astype(np.float32)
        denoised[:, :, z] = out_slice
        denoised_256[:, :, z] = _resize_2d(out_slice, (256, 256)).astype(np.float32)
        denoised_512[:, :, z] = _resize_2d(out_slice, (512, 512)).astype(np.float32)

    print("\nProcessing complete.")

    original_mag = np.abs(raw_data).astype(np.float32)
    residual_mag = original_mag - denoised

    print(f"Saving results to {output_file}...")
    sio.savemat(
        output_file,
        {
            "original_complex": raw_data,
            "original_mag": original_mag,
            "denoised": denoised,
            "denoised_256": denoised_256,
            "denoised_512": denoised_512,
            "residual_mag": residual_mag,
        },
    )
    print("Done.")


def _parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--resize must be formatted as HxW (e.g., 128x128)")
    h, w = int(parts[0]), int(parts[1])
    if h <= 0 or w <= 0:
        raise ValueError("--resize dimensions must be positive")
    return h, w


def main():
    parser = argparse.ArgumentParser(
        description="Run NAFNet denoising on complex spiral stacks stored in .mat"
    )
    parser.add_argument(
        "--mat_file",
        type=str,
        required=True,
        help="Path to input .mat file containing a 3D complex array [H,W,S]",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to best_model.pt or its parent experiment directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="nafnet_denoised.mat",
        help="Output .mat file path",
    )
    parser.add_argument(
        "--nafnet_config",
        type=str,
        default="small",
        help="NAFNet config: tiny | small | base",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional preprocessing resize as HxW before denoising (e.g., 128x128)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )
    args = parser.parse_args()

    resize_hw = _parse_resize(args.resize)
    run_inference(
        mat_file=args.mat_file,
        checkpoint_path=args.checkpoint,
        output_file=args.output_file,
        nafnet_config=args.nafnet_config,
        resize_hw=resize_hw,
        cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
