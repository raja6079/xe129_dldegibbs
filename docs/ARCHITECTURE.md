# Architecture

Detailed reference for the `xe129_dldegibbs` pipeline. For the quick-start
overview see [../CLAUDE.md](../CLAUDE.md).

## 1. Purpose

Restore 129Xe (hyperpolarized xenon) MRI images by removing Gibbs ringing and
noise, adapting the Muckley et al. DLDeGibbs approach. The network is trained on
**ImageNet natural images** with MRI acquisition artifacts *simulated* on the
fly by a chain of transforms, then applied at inference to real 129Xe volumes.

## 2. The Muckley contract

Any backbone plugged into this pipeline must satisfy:

- Input/output tensors are `[B, C, H, W]`.
- `C = 2` for complex data (channel 0 = real, channel 1 = imag), `C = 1` for
  magnitude.
- A **global residual**: the network predicts a correction added back to the
  input (`out = net(x) + x`). This is central to how the model was trained —
  removing it breaks convergence and any pretrained checkpoint.
- Optional `comp2mag`: when set, the final step returns
  `sqrt(re^2 + im^2)` (requires ≥2 output channels), producing a 1-channel
  magnitude image from complex output.

`build_model()` in [../model_factory.py](../model_factory.py) is the single
factory; nothing else should instantiate a backbone directly.

## 3. Backbones

### 3.1 Wavelet U-Net (legacy)

`UNetWavelet` in [../mrinet/unet_wavelet.py](../mrinet/unet_wavelet.py). The
original Muckley model. Residual handled internally via `resid=True`,
`wave_concat=True`. Hyperparameters: `nlayers`, `nchans`, `leaky`.

### 3.2 NAFNet (new)

Defined in [../nafnet.py](../nafnet.py). "Simple Baselines for Image
Restoration" (Chen et al., ECCV 2022).

Building blocks:
- **`SimpleGate`** — multiplicative gating, splits channels in half and
  multiplies (`C -> C/2`); replaces nonlinear activations.
- **`SimplifiedChannelAttention` (SCA)** — global avg-pool → 1×1 conv → scale.
- **`LayerNorm2d`** — channel-dim layernorm for `[B, C, H, W]`.
- **`NAFBlock`** — norm → 1×1 → depthwise 3×3 → SimpleGate → SCA → 1×1, plus an
  FFN branch, each with a learnable per-channel scale (`beta`, `gamma`) and
  optional Dropout2d.

`NAFNet` is a U-Net of NAFBlocks:
- `intro`/`ending` 3×3 convs map in/out channels ↔ `width`.
- Encoder: `enc_blks` NAFBlocks per level, then a downsample; `middle_blks` at
  the bottleneck; decoder mirrors with upsample + skip.
- **Downsample/skip depends on `arch_variant`:**
  - **v1**: `Conv2d(stride=2)` down + **additive** skip (`x + skip`). Legacy.
  - **v2**: `PixelUnshuffle(2)` + 1×1 conv down (invertible, preserves HF) +
    **concat** skip fused by a 1×1 `skip_fuse` conv. Sharper; CLI default.
- Upsample is always `Conv2d` + `PixelShuffle(2)`.
- `_check_image_size()` pads H/W up to a multiple of `padder_size =
  2**len(enc_blks)` (= 16 for all presets) before the forward pass; the output
  is cropped back to the original H/W. **Do not pre-pad inputs.**

`NAFNetDeGibbs` wraps `NAFNet` to enforce the Muckley contract: applies the
global residual, optional `comp2mag`, and optional `noise_map_conditioning`
(concatenates a `[B, 1, H, W]` per-pixel noise-variance map as an extra input
channel — defaults to ones when not supplied; intended for non-Cartesian
trajectories where noise is spatially colored).

### 3.3 Presets

Defined in both [../nafnet_config.py](../nafnet_config.py) (`NAFNET_PRESETS`,
the authoritative dataclass form) and [../configs/nafnet.yaml](../configs/nafnet.yaml)
(YAML mirror). Keep these two in sync.

| Preset | width | enc_blks | middle | dec_blks | ~params |
|--------|-------|----------|--------|----------|---------|
| tiny  | 16 | [1,1,2,2] | 4  | [1,1,1,1] | ~3.7M (smoke tests) |
| small | 32 | [2,2,4,8] | 12 | [2,2,2,2] | ~17M (main config) |
| base  | 64 | [2,2,4,8] | 12 | [2,2,2,2] | ~68M (2× A6000) |

`load_nafnet_config()` accepts a preset name **or** a path to a YAML with a
`presets:` mapping (+ `block_defaults:` for `dw_expand`/`ffn_expand`/
`drop_out_rate`).

## 4. Training wrapper: `DeGibbs`

[../degibbs_model.py](../degibbs_model.py). Owns the model, optimizer, and
train/val loops.

- `load_model()` — loads `state_dict` from a `.pt`; tolerant of missing files
  (warns and continues with random init).
- `init_optimizer()` — **dual path**:
  - `optim_cfg is None` → plain Adam (wavelet path).
  - `optim_cfg` provided → AdamW (or Adam) + optional cosine LR schedule +
    optional AMP (fp16 with `GradScaler`, or bf16) + optional grad-clip. Driven
    by the `nafnet_optim` block in [../configs/optim.yaml](../configs/optim.yaml).
    LR from config always overrides any LR restored from a checkpoint.
  - Wraps in `DistributedDataParallel` when launched under DDP, then
    `torch.compile` unless `TORCH_NO_COMPILE` is set.
- `run_train_loop()` / `run_val_loop()` — standard loops; val can compute full
  metrics (mse/ssim/nmse/psnr via [../data/metrics.py](../data/metrics.py)) and
  writes TensorBoard scalars + images. 2-ch outputs are collapsed to magnitude
  for metrics/visuals.
- `fit()` — epoch loop; saves `best_model.pt` on improving val loss, else
  `model_epoch_N.pt`. Checkpoints bundle model + optimizer (+ scheduler/scaler)
  state.

## 5. Entry point: `degibbs_main.py`

- CLI = `configs/config_init.py::generate_parser()` plus NAFNet flags:
  `--model {wavelet,nafnet}`, `--nafnet_config`, `--nafnet_arch_variant
  {v1,v2}` (default v2), `--noise_map_conditioning`, `--loss {mse,l1,charbonnier}`.
- YAML files are merged then overridden by any non-None CLI arg
  (`read_yamls()`, recursive key update).
- `_exp_folder_name()` builds the checkpoint subdirectory from backbone, config,
  arch variant, loss, and `pf_fact`. **This determines where checkpoints are
  saved and loaded** — a mismatch means the model silently trains from scratch.
  The `_nsd_moderatepsf` sim tag is appended only for `spiral` exp_types.
- DDP auto-inits when `LOCAL_RANK` is in the environment (i.e. under `torchrun`).

## 6. Data pipeline

[../data/degibbs_loaders.py](../data/degibbs_loaders.py) `init_loaders()`.
Source images are ImageNet (via [../data/imagenet.py](../data/imagenet.py)),
transformed by a `Compose` of the transforms in
[../data/transforms/](../data/transforms/). Each sample is a dict with `dat`
(network input, corrupted) and `target` (clean ground truth).

### 6.1 Cartesian Gibbs pipeline (default)

RandFlip → RandTranspose → random phase (`MriRandPhase`/`MriRandPhaseBig` for
>256px) → random ellipse → **FFT** → **center crop** (the Gibbs truncation) →
noise (`MriNoise`, log2-SNR range) → **partial-Fourier mask** (`MriMask`) →
**inverse FFT** → max-normalize → optional PF reconstruction
(`MriMargosian`/`MriPfPocs` for `complex_margosian`/`complex_pocs`) → optional
magnitude → resize target → `ComplexToTensor`.

### 6.2 Stack-of-spirals pipeline (`exp_type` contains `spiral`)

Augment → FFT → center crop (match Gibbs resolution) → inverse FFT →
**`MriSpiral`** (corrupts `dat` only) → normalize → optional magnitude →
`ComplexToTensor`.

**`MriSpiral`** ([../data/transforms/mrispiral.py](../data/transforms/mrispiral.py))
synthesizes three s3d artifact components:
1. **Colored noise** from density-compensation after non-uniform gridding — a
   radial noise-spectral-density (NSD) profile (`v1` parametric DCF saturation +
   Gaussian edge; `v2` power-law/sigmoid blend; `random` picks per sample).
2. **Complex PSF/MTF** = T2* magnitude decay × B0 off-resonance phase, both
   ∝ readout time `t ~ r²` of a center-out spiral. Skipped with `skip_psf_prob`.
3. **Through-plane (kz)** Cartesian sampling coupling (currently the multiply is
   commented out in `_corrupt`).

Output: `dat = (blend of clean & PSF-blurred) + colored_noise`; `target` clean.

## 7. Inference: `infer_xe129.py`

Loads a NAFNet-small checkpoint (strips `module.`/`_orig_mod.` prefixes from
DDP/`torch.compile`), reads a 3D complex volume (`.npy`/`.npz`/NIfTI), processes
**slice-by-slice** with per-slice max-normalization (matching
`MriNormalize('Max')` from training), and writes a real-valued magnitude volume.
`--slice_axis` chooses the iteration axis; `--arch_variant` must match the
checkpoint.

## 8. Configs

- [../configs/data.yaml](../configs/data.yaml) — `exp_dir`, train/test data dirs,
  workers.
- [../configs/model.yaml](../configs/model.yaml) — `model_params` + per-`exp_type`
  overrides (`in_ch`, `out_ch`, `comp2mag`, `leaky`).
- [../configs/optim.yaml](../configs/optim.yaml) — `training_params` +
  `nafnet_optim` (NAFNet-only optimizer/scheduler/AMP).
- [../configs/nafnet.yaml](../configs/nafnet.yaml) — NAFNet size presets.
- [../configs/transforms.yaml](../configs/transforms.yaml) — `gibb_size`,
  `orig_big`, `pf_fact`, `pe_fact` (simulation geometry).

## 9. Not yet wired in

`pinvrecon/` is present as a package but only compiled `.pyc` files exist in the
tree; its source is untracked and it is not referenced from the main training or
inference flow.
