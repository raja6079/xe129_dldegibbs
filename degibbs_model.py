import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from mrinet.unet_wavelet import UNetWavelet
from data.metrics import MetricEval


class DeGibbs:
    """A U-Net model for Gibbs artifact correction.

    This initializes a wavelet U-Net model intended for artifact correciton in
    medical images. The script allows attachment of PyTorch data loader
    objects and associated transforms, in principle allowing the correction of
    any artifacts simulated by the transforms.

    By default, the model output parameters are saved in
    'exp_dir/best_model.pt'

    Args:
        nlayers (int): Number of layers for U-Net.
        in_ch (int): Number of input channels (usually 1 for magnitude, 2 for
            complex).
        out_ch (int): Number of output channels.
        comp2mag (boolean): A flag for whether to output magnitude outputs when
            given complex inputs. If True, then performs the magnitude operation
            as a final step of the U-Net.
        leaky (boolean): If true, use leaky ReLUs instead of normal ones.
        device (torch.device): Use torch.device('cuda') for GPU or
            torch.device('cpu') for CPU.

    Examples:
        Initialization:

        >>> ob = DeGibbs(**params)

        Loading parameters from file:

        >>> ob = ob.load_model(model_file)

        Attaching a dataloader:

        >>> ob = ob.attach_data(exp_type, train_loader, val_loader)

        Training:

        >>> ob = ob.fit(num_epochs)

        Run current model on example:

        >>> out = ob(in)
    """

    def __init__(
        self,
        nlayers,
        in_ch,
        out_ch,
        nchans,
        comp2mag,
        leaky,
        device,
        dtype=torch.double,
    ):
        self.optimizer = None
        self.train_data_dir = None
        self.model = UNetWavelet(
            ndims=2,
            nlayers=nlayers,
            in_ch=in_ch,
            out_ch=out_ch,
            top_filtnum=nchans,
            resid=True,
            wave_concat=True,
            comp2mag=comp2mag,
            leaky=leaky,
        )

        self.model = self.model.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self._rank = 0

    def __call__(self, x):
        return self.model(x)

    def _log(self, *args, **kwargs):
        """Print only from rank 0 in DDP, or always in single-GPU mode."""
        if self._rank == 0:
            print(*args, **kwargs)

    def load_model(self, model_file, device=None, dtype=None):
        """Loads model parameters from .pt file.

        Args:
            model_file (str): A directory pointing to the .pt file with
                parameters.
            device (torch.device, default=None): Device to send model to after
                loading. If None, uses device supplied by init function.

        Returns:
            self
        """
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        if dtype is not None:
            self.dtype = dtype

        self.model = self.model.to(dtype=dtype)

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self._log(f"network params: {pytorch_total_params}")

        if os.path.isfile(model_file):
            params = self.model.state_dict()

            self._log(f"loading model from file: {model_file}")
            checkpt = torch.load(model_file, map_location=self.device, weights_only=False)
            state_dict = checkpt["state_dict"]
            params.update(state_dict)
            self.model.load_state_dict(params)
        else:
            self._log(f"model file {model_file} not found")

        self.model = self.model.eval()

        return self

    def init_optimizer(self, learning_rate, loss_fn, model_file=None,
                       optim_cfg=None, total_iters=None):
        """Initialize optimizer (and optional scheduler / AMP scaler).

        When ``optim_cfg`` is None, the original Adam-only behavior is used
        (this is the wavelet U-Net path). When ``optim_cfg`` is provided, it
        configures the NAFNet path: AdamW + cosine LR schedule + optional AMP.

        Args:
            learning_rate (double): Base learning rate (used when optim_cfg is
                None, or as a fallback if optim_cfg lacks 'learning_rate').
            loss_fn (torch.loss_fn): The PyTorch loss function.
            model_file (str): If not None, loads optimizer/scheduler/scaler
                state from this checkpoint when present.
            optim_cfg (dict, default=None): NAFNet optimizer config (see
                configs/optim.yaml :: nafnet_optim).
            total_iters (int, default=None): Total training iterations for
                cosine annealing. Required when optim_cfg requests a scheduler.

        Returns:
            self
        """
        self.epoch = 0
        self.train_loss_min = np.inf
        self.val_loss_min = np.inf
        self.loss_fn = loss_fn

        # Defaults: wavelet path (Adam, no scheduler, no AMP, no grad clip).
        self.scheduler = None
        self.scaler = None
        self.grad_clip = None
        self.amp_dtype = None

        if optim_cfg is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
            self.learning_rate = learning_rate
        else:
            opt_name = str(optim_cfg.get('optimizer', 'adamw')).lower()
            lr = float(optim_cfg.get('learning_rate', learning_rate))
            betas = tuple(optim_cfg.get('betas', (0.9, 0.9)))
            weight_decay = float(optim_cfg.get('weight_decay', 0.0))

            if opt_name == 'adamw':
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr, betas=betas, weight_decay=weight_decay,
                )
            elif opt_name == 'adam':
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=lr, betas=betas, weight_decay=weight_decay,
                )
            else:
                raise ValueError(
                    f'Unsupported optimizer: {opt_name}'
                )
            self.learning_rate = lr

            sched_name = optim_cfg.get('scheduler', None)
            if sched_name is not None:
                sched_name = str(sched_name).lower()
                if sched_name == 'cosine':
                    if total_iters is None or total_iters <= 0:
                        raise ValueError(
                            'total_iters must be > 0 for cosine scheduler'
                        )
                    eta_min = float(optim_cfg.get('eta_min', 0.0))
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=total_iters, eta_min=eta_min,
                    )
                else:
                    raise ValueError(
                        f'Unsupported scheduler: {sched_name}'
                    )

            gc = optim_cfg.get('grad_clip', None)
            if gc is not None and float(gc) > 0:
                self.grad_clip = float(gc)

            if bool(optim_cfg.get('amp', False)):
                amp_dtype_str = str(
                    optim_cfg.get('amp_dtype', 'float16')
                ).lower()
                self.amp_dtype = {
                    'float16': torch.float16,
                    'fp16': torch.float16,
                    'half': torch.float16,
                    'bfloat16': torch.bfloat16,
                    'bf16': torch.bfloat16,
                }.get(amp_dtype_str, torch.float16)
                # GradScaler is only needed for fp16. bf16 has fp32-like range.
                if self.amp_dtype == torch.float16:
                    self.scaler = torch.amp.GradScaler("cuda")

        if model_file is not None and os.path.isfile(model_file):
            self._log(f"loading optimizer from file: {model_file}")
            checkpt = torch.load(model_file, map_location=self.device, weights_only=False)

            optimizer.load_state_dict(checkpt["optimizer"])
            # Override LR from config so checkpoint LR never silently persists.
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                pg['initial_lr'] = lr

            self.epoch = checkpt["epoch"] + 1
            train_loss_min = self.train_loss_min
            try:
                self.train_loss_min = checkpt["train_loss_min"]
            except KeyError:
                self.train_loss_min = train_loss_min
            val_loss_min = self.val_loss_min
            try:
                self.val_loss_min = checkpt["val_loss_min"]
            except KeyError:
                self.val_loss_min = val_loss_min

            if self.scheduler is not None and 'scheduler' in checkpt:
                try:
                    self.scheduler.load_state_dict(checkpt['scheduler'])
                except Exception as e:
                    self._log(f'could not restore scheduler state: {e}')
            if self.scaler is not None and 'scaler' in checkpt:
                try:
                    self.scaler.load_state_dict(checkpt['scaler'])
                except Exception as e:
                    self._log(f'could not restore grad scaler state: {e}')
        elif model_file is not None:
            self._log(f"model file {model_file} not found")

        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank],
                gradient_as_bucket_view=True)
            self.device = torch.device(f"cuda:{local_rank}")
            self._rank = local_rank
            print(f"Using DDP on rank {local_rank}")

        # Compile model for faster execution (PyTorch 2.0+)
        if hasattr(torch, "compile") and not os.environ.get("TORCH_NO_COMPILE"):
            self.model = torch.compile(self.model)
        self.optimizer = optimizer

        return self

    def attach_data(self, exp_type, train_loader, val_loader, train_data_dir=None):
        """Attaches a dataloader for training.

        Args:
            exp_type (str): Experiment type (used for visualizations).
            train_loader (torch.DataLoader): The loader for the training split.
            val_loader (torch.DataLoader): The loader for the validation split.
            train_data_dir (str, default=None): Stored as attribute for print
                statements.

        Returns:
            self
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_type = exp_type
        self.train_data_dir = train_data_dir

        return self

    def _save_checkpoint(self, filename):
        """Save current model state.

        Args:
            filename (str): File name for .pt file to save model.
        """
        try:
            state = {
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
        except AttributeError:
            state = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

        for attr, value in self.__dict__.items():
            if (
                (not "model" in attr)
                and (not "optimizer" in attr)
                and (not "loss_fn" in attr)
                and (attr not in ("scheduler", "scaler", "train_loader", "val_loader"))
            ):
                state[attr] = value

        if getattr(self, "scheduler", None) is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if getattr(self, "scaler", None) is not None:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, filename)

    def run_val_loop(self, visuals, rnd_seed, all_metrics=False):
        """Run a loop over the validation data split.

        Args:
            visuals (dict): A dictionary object storing all arrays for
                visualization.
            rnd_seed (int): Seed for numpy, PyTorch, and CUDA (for
                reproducibility).

        Returns:
            val_loss (double): Validation loss.
            visuals (dict): Dictionary visuals object with validation images.
        """
        self._log(f"epoch {self.epoch}: validation loop")
        # set random seed
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.cuda.manual_seed(rnd_seed)
        device = self.device

        self.model = self.model.eval()

        disp_frac_vec = np.array(range(11))
        disp_val = disp_frac_vec[0]
        disp_counter = 0

        if all_metrics:
            metrics = {"mse": [], "ssim": [], "nmse": [], "psnr": []}

        with torch.inference_mode():
            val_loss = 0
            counter = 0
            for i, val_batch in enumerate(self.val_loader):
                val_target, val_dat = (
                    val_batch["target"].to(device),
                    val_batch["dat"].to(device),
                )

                val_est = self.model(val_dat)
                val_loss = (
                    val_loss * counter + self.loss_fn(val_est, val_target).item()
                ) / (counter + 1)
                counter = counter + 1

                if all_metrics:
                    for ind in range(val_target.shape[0]):
                        nptarget = val_target[ind].cpu().numpy()
                        if nptarget.shape[0] > 1:
                            nptarget = np.sqrt(nptarget[0] ** 2 + nptarget[1] ** 2)
                        else:
                            nptarget = nptarget[0]
                        npest = val_est[ind].cpu().numpy()
                        if npest.shape[0] > 1:
                            npest = np.sqrt(npest[0] ** 2 + npest[1] ** 2)
                        else:
                            npest = npest[0]
                        metrics["mse"].append(MetricEval.mse(nptarget, npest))
                        metrics["ssim"].append(MetricEval.ssim(nptarget, npest))
                        metrics["nmse"].append(MetricEval.nmse(nptarget, npest))
                        metrics["psnr"].append(MetricEval.psnr(nptarget, npest))

                if (i / len(self.val_loader)) >= (disp_val / 10):
                    self._log(f"validation loop progress: {100 * (i + 1) / len(self.val_loader):.0f}%")
                    disp_counter += 1
                    disp_val = disp_frac_vec[disp_counter]

            self._log("validation loop finished")

            dispind = 0

            if val_target.shape[1] > 1:  # assume dim 0, 1 are real, imag
                tmp = np.squeeze(
                    val_target[dispind, ...].cpu().detach().float().numpy()
                )
                visuals["val_target"] = np.expand_dims(
                    np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
                )
            else:
                visuals["val_target"] = np.absolute(
                    val_target[dispind, ...].cpu().detach().numpy()
                )
            if val_dat.shape[1] > 1:  # assume dim 0, 1 are real, imag
                tmp = np.squeeze(val_dat[dispind, ...].cpu().detach().float().numpy())
                visuals["val_dat"] = np.expand_dims(
                    np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
                )
            else:
                visuals["val_dat"] = np.absolute(
                    val_dat[dispind, ...].cpu().detach().numpy()
                )
            if val_est.shape[1] > 1:  # assume dim 0, 1 are real, imag
                tmp = np.squeeze(val_est[dispind, ...].cpu().detach().float().numpy())
                visuals["val_est"] = np.expand_dims(
                    np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
                )
            else:
                visuals["val_est"] = np.absolute(
                    val_est[dispind, ...].cpu().detach().float().numpy()
                )

        if all_metrics:
            mse = np.array(metrics["mse"])
            mse = mse[~np.isnan(mse)]
            mse = mse[~np.isinf(mse)]
            nmse = np.array(metrics["nmse"])
            nmse = nmse[~np.isnan(nmse)]
            nmse = nmse[~np.isinf(nmse)]
            ssim = np.array(metrics["ssim"])
            ssim = ssim[~np.isnan(ssim)]
            ssim = ssim[~np.isinf(ssim)]
            psnr = np.array(metrics["psnr"])
            psnr = psnr[~np.isnan(psnr)]
            psnr = psnr[~np.isinf(psnr)]

            metrics["mse"] = mse
            metrics["nmse"] = nmse
            metrics["ssim"] = ssim
            metrics["psnr"] = psnr

            return metrics, visuals
        else:
            return val_loss, visuals

    def run_train_loop(self, global_iter, visuals, rnd_seed):
        """Run a loop over the training data split.

        The model is updated via the self.model attribute.

        Args:
            global_iter (int): Global iteration count (for print statements).
            visuals (dict): A dictionary object storing all arrays for
                visualization.
            rnd_seed (int): Seed for numpy, PyTorch, and CUDA (for
                reproducibility).

        Returns:
            global_iter (int): Updated global iteration.
            visuals (dict): Dictionary visuals object with validation images.
            disp_loss (double): A display loss for the training data taken as
                an average from the last few images.
        """
        self._log(f"epoch {self.epoch}: training loop")
        torch.manual_seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.cuda.manual_seed(rnd_seed)

        device = self.device

        self.model = self.model.train()
        losses = []

        disp_frac_vec = np.array(range(11))
        disp_val = disp_frac_vec[0]
        disp_counter = 0

        for i, batch in enumerate(self.train_loader):
            target, dat = batch["target"].to(device), batch["dat"].to(device)

            self.optimizer.zero_grad(set_to_none=True)

            use_amp = self.amp_dtype is not None and device.type == 'cuda'
            if use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    est = self.model(dat)
                    loss = self.loss_fn(est, target)
            else:
                est = self.model(dat)
                loss = self.loss_fn(est, target)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            np_train_loss = loss.item()
            losses.append(np_train_loss)
            losses = losses[-50:]
            disp_loss = np.mean(losses)

            global_iter = global_iter + 1

            if self._rank == 0 and global_iter % 500 == 0:
                self._log(f"  iter {global_iter}: rolling_loss={disp_loss:.6f}")

            if (i / len(self.train_loader)) >= (disp_val / 10):
                self._log(f"training loop progress: {100 * (i + 1) / len(self.train_loader):.0f}%")
                disp_counter += 1
                disp_val = disp_frac_vec[disp_counter]

        self._log("training loop finished")

        dispind = 0

        if target.shape[1] > 1:  # assume dim 0, 1 are real, imag
            tmp = np.squeeze(target[dispind, ...].cpu().detach().float().numpy())
            visuals["train_target"] = np.expand_dims(
                np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
            )
        else:
            visuals["train_target"] = np.absolute(
                target[dispind, ...].cpu().detach().numpy()
            )
        if dat.shape[1] > 1:  # assume dim 0, 1 are real, imag
            tmp = np.squeeze(dat[dispind, ...].cpu().detach().float().numpy())
            visuals["train_dat"] = np.expand_dims(
                np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
            )
        else:
            visuals["train_dat"] = np.absolute(dat[dispind, ...].cpu().detach().numpy())
        if est.shape[1] > 1:  # assume dim 0, 1 are real, imag
            tmp = np.squeeze(est[dispind, ...].cpu().detach().float().numpy())
            visuals["train_est"] = np.expand_dims(
                np.sqrt(tmp[0, ...] ** 2 + tmp[1, ...] ** 2), 0
            )
        else:
            visuals["train_est"] = np.absolute(
                est[dispind, ...].cpu().detach().float().numpy()
            )

        return global_iter, visuals, disp_loss

    def fit(self, num_epochs, exp_dir=None, run_eval=True, seed_offset=476):
        """Fit attached data loaders (i.e., train the model).

        The model is updated via the self.model attribute.

        Args:
            num_epochs (int): Number of epochs to train.
            exp_dir (str, default=None): String pointing to directory for
                logging tensorboardX outputs.
            run_eval (boolean, default=True): Whether to run validation loop.
            seed_offset (int, default=476): Offset for random number seed (for
                reproducibility).

        Returns:
            self
        """
        self._log("starting training")

        if exp_dir is not None:
            self._log(f"saving logs to {exp_dir}")
            writer = SummaryWriter(log_dir=exp_dir)
        visuals = {}

        self._log("initializing loss tracking")
        epochs = []
        train_losses = []
        val_losses = []

        global_iter = 0

        for epoch in range(self.epoch, num_epochs):
            rnd_seed = np.random.get_state()[1][0] + self.epoch + seed_offset
            self.epoch = epoch
            epoch_start = time.time()

            global_iter, visuals, train_loss = self.run_train_loop(
                global_iter, visuals, rnd_seed
            )

            if run_eval:
                val_loss, visuals = self.run_val_loop(visuals, seed_offset)

                if exp_dir is not None:
                    writer.add_scalar(
                        "losses/eval_loss", scalar_value=val_loss, global_step=epoch
                    )

            if exp_dir is not None:
                writer.add_scalar(
                    "losses/train_loss", scalar_value=train_loss, global_step=epoch
                )

                for label, image in visuals.items():
                    if "val" in label:
                        image = image / np.max(image)
                        writer.add_image(
                            "validation/" + label, image, global_step=epoch
                        )
                    elif "train" in label:
                        image = image / np.max(image)
                        writer.add_image("training/" + label, image, global_step=epoch)

            if run_eval:
                val_losses.append(val_loss)
                if val_loss < self.val_loss_min:
                    self.val_loss_min = val_loss
                    checkname = os.path.join(exp_dir, "best_model.pt")
                else:
                    checkname = os.path.join(exp_dir, f"model_epoch_{epoch}.pt")
            else:
                if train_loss < self.train_loss_min:
                    checkname = os.path.join(exp_dir, "best_model.pt")
                else:
                    checkname = os.path.join(exp_dir, f"model_epoch_{epoch}.pt")

            if train_loss < self.train_loss_min:
                self.train_loss_min = train_loss

            train_losses.append(train_loss)
            epochs.append(epoch)

            self._save_checkpoint(checkname)

            epoch_end = time.time()

            self._log(f"epoch finished, time: {epoch_end - epoch_start}")
            self._log(f"validation loss: {val_loss}, training loss: {train_loss}")

        return self

    def __repr__(self):
        self.model = self.model.train()
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        out = "\n" + self.__class__.__name__ + "\n"
        out += f"model: {self.model.__class__.__name__}"
        out += f"    number of trainable model parameters: {num_params}"
        out += f"optimizer: {self.optimizer.__class__.__name__}"
        out += f"train_data_dir: {self.train_data_dir}"

        return out
