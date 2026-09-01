# CLAUDE.md

Context for Claude Code when working in this repo. Auto-loaded each session.
Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## What this is

`xe129_dldegibbs` — an adaptation of Muckley et al.'s
[DLDeGibbs](https://github.com/mmuckley/dldegibbs) pipeline (Gibbs-ringing +
noise removal for diffusion MRI, [arXiv:1905.04176](https://arxiv.org/abs/1905.04176))
for **129Xe (hyperpolarized xenon) MRI**. The main addition over the original is
a **NAFNet restoration backbone** alongside the legacy wavelet U-Net.

Research use only — not validated for clinical use.

## Two backbones, one contract

Everything routes through [model_factory.py](model_factory.py) `build_model()` so
the rest of the code is backbone-agnostic. Both backbones honor the **Muckley
contract**: global residual (`out = net(x) + x`), tensor convention `[B, C, H, W]`,
2-ch complex (real/imag) or 1-ch magnitude I/O, optional `comp2mag` (magnitude of
complex output as final step).

- **`wavelet`** (legacy, default) — [mrinet/unet_wavelet.py](mrinet/unet_wavelet.py)
  `UNetWavelet`. Residual via `resid=True`.
- **`nafnet`** (new) — [nafnet.py](nafnet.py) `NAFNetDeGibbs` wrapping `NAFNet`.
  Residual applied in its own `forward`. Presets tiny/small/base.
  - `arch_variant`: **v1** = stride-conv down + additive skip (legacy);
    **v2** = PixelUnshuffle down + concat skip (sharper). CLI default is **v2**.
  - `noise_map_conditioning`: optional extra input channel (per-pixel noise
    variance map) for non-Cartesian trajectories.

## Key files

| Path | Role |
|------|------|
| [degibbs_main.py](degibbs_main.py) | Entry point: `train`/`val`/`test`. DDP, backbone/loss selection, `_exp_folder_name()` (checkpoint dir naming). |
| [degibbs_model.py](degibbs_model.py) | `DeGibbs` wrapper: train/val loops, checkpointing, dual optimizer path. |
| [model_factory.py](model_factory.py) | `build_model()` — the only place code should obtain a backbone. |
| [nafnet.py](nafnet.py) | NAFNet blocks + `NAFNet` U-Net + `NAFNetDeGibbs` wrapper. |
| [nafnet_config.py](nafnet_config.py) | `NAFNetConfig`, presets, YAML loader, `build_nafnet_from_config()`. |
| [infer_xe129.py](infer_xe129.py) | Slice-by-slice inference on 3D complex 129Xe volumes (.npy/.npz/NIfTI). |
| [siemens_twix_recon.py](siemens_twix_recon.py) | Siemens twix (.dat) Cartesian 129Xe recon front-end (repo root). |
| [PythonRecon/](PythonRecon/) | Real-data recon+denoise front-end scripts (`run_cartesian_nafnet.py`, `run_cartesian_dldegibbs.py`, `run_cartesian_ardl.py`, `run_sos_nafnet.py`, `run_manchester_batch.sh`) + the Manchester/SOS data trees. Import the model core (`nafnet.py`) via absolute path, so they run from anywhere. |
| [PythonRecon/run_cartesian_ardl.py](PythonRecon/run_cartesian_ardl.py) | ARDL (TensorFlow c3d SavedModel from `xe129_denoising_project/ardl_models/test-c3d`) denoise/de-ring on a Cartesian `*_recon.mat`, per-slice, for both 3D and 2D data. Loads the frozen `serving_default` meta-graph in a TF-v1 session (new TF can't restore the legacy SavedModel's optimizer). Runs in its own TF-only venv (`xe129_ardl_venv`), NOT the torch venv. Normalizes each slice to ARDL's ~128 peak. Output `*_ardl.mat` + PNGs. |
| [data/degibbs_loaders.py](data/degibbs_loaders.py) | Builds transform pipelines (Cartesian Gibbs + stack-of-spirals). |
| [data/transforms/](data/transforms/) | Composable MRI simulation transforms. |
| [data/transforms/mrispiral.py](data/transforms/mrispiral.py) | Stack-of-spirals (s3d) artifact simulation for 129Xe. |
| [configs/](configs/) | YAML config: `data`, `model`, `optim`, `nafnet`, `transforms`. |

## Runtime environment

- Code runs **on this server**, where it currently lives (`/home/raja/xe129_dldegibbs`).
- Hardware: **2× NVIDIA A6000** → train with `torchrun --standalone --nproc_per_node=2` (DDP).
- Python venv: `xe129_dldegibbs_venv/` (activate with `source xe129_dldegibbs_venv/bin/activate`).
- ARDL venv: `xe129_ardl_venv/` — separate TF-only venv (tensorflow-cpu) for
  `run_cartesian_ardl.py`. Kept apart from the torch venv because TF + torch clash
  on numpy/protobuf. The batch script calls its Python by absolute path.
- Long runs are launched inside a **`screen`** session (e.g. `screen -S nafnet_dldegibbs`)
  and logged with `tee` into `logs/` (git-ignored).

Canonical training invocation (most recent run):

```bash
cd /home/raja/xe129_dldegibbs && source xe129_dldegibbs_venv/bin/activate && mkdir -p logs && \
torchrun --standalone --nproc_per_node=2 degibbs_main.py train complex_spiral \
    --model nafnet --nafnet_config small --nafnet_arch_variant v2 --loss charbonnier \
    --num_epochs 20 2>&1 | tee logs/train_complex_spiral_skippsf04_20260624_123508.log
```

Log filenames encode the run (`train_<exp>_<variant>_<timestamp>.log`); see `logs/`.

## Running

Training data is **ImageNet** images (simulated MRI artifacts applied via
transforms). Config lives in `configs/*.yaml`; CLI flags override YAML.

```bash
# Train NAFNet-small v2, complex Cartesian, on ImageNet-simulated data
python degibbs_main.py train complex --model nafnet --nafnet_config small \
    --nafnet_arch_variant v2 --loss charbonnier --exp_dir my_exp/

# Stack-of-spirals experiment (uses MriSpiral simulation)
python degibbs_main.py train complex_spiral --model nafnet --loss charbonnier

# Multi-GPU (DDP) on the 2× A6000 — how runs are actually launched here
torchrun --standalone --nproc_per_node=2 degibbs_main.py train complex --model nafnet

# Inference on a 129Xe volume
python infer_xe129.py --input xe129.npy --output out.npy \
    --checkpoint /path/to/best_model.pt --arch_variant v2
```

Positional args: `mode` (`train`/`val`/`test`) and `exp_type` (see
[configs/model.yaml](configs/model.yaml): `complex`, `complex_margosian`,
`complex_pocs`, `magnitude`, `magnitude_lores`, `complex_spiral`,
`magnitude_spiral`).

## Config notes

- **Optimizer**: wavelet → Adam only. nafnet → AdamW + cosine LR + optional AMP +
  grad-clip, driven by `nafnet_optim` block in [configs/optim.yaml](configs/optim.yaml).
- **Checkpoints** live at `exp_dir/<exp_type>/<suffix>/best_model.pt`; the suffix
  is built by `_exp_folder_name()` in [degibbs_main.py](degibbs_main.py) from
  backbone/config/arch/loss/pf_fact — changing any of these changes the load path.
- `torch.compile` is applied unless `TORCH_NO_COMPILE` env var is set.

## Conventions & gotchas

- Tensor shape is **always** `[B, C, H, W]`; complex is 2 channels (real, imag),
  magnitude is 1 channel. Metrics/visuals collapse 2-ch to magnitude via
  `sqrt(re^2 + im^2)`.
- NAFNet pads input to a multiple of `2**len(enc_blks)` (16 for all presets) and
  crops the output back — don't add external padding.
- The **global residual is load-bearing** — see the "do not remove" comment in
  [nafnet.py](nafnet.py). Removing it breaks the Muckley training contract.
- `comp2mag` requires ≥2 output channels.
- In `MriSpiral`, only `dat` is corrupted; `target` stays clean.

## Current branch: `xe129_ventilation`

Work in progress (uncommitted at time of writing) tunes spiral-simulation realism
and fixes checkpoint naming:
- `mrispiral.py`: `skip_psf_prob` 0.7 → 0.4 (PSF applied more often).
- `degibbs_main.py`: `_nsd_moderatepsf` tag applied **only** for `spiral` exp_types
  (Cartesian doesn't simulate NSD/PSF); wavelet SNR tag `0-5` → `0-6`.
- `.gitignore`: ignore `logs/*`.

`pinvrecon/` exists as a package but only `.pyc` files are present — source is not
tracked and it's not referenced in the main flow yet.
