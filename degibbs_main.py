import os
import pickle
import sys

import torch.distributed as dist
import torch

from configs.config_init import generate_parser, read_yamls
from data.degibbs_loaders import init_loaders
from degibbs_model import DeGibbs
from model_factory import build_model


def _build_loss(name: str):
    """Return the loss function from a CLI name."""
    if name == 'mse':
        return torch.nn.MSELoss()
    elif name == 'l1':
        return torch.nn.L1Loss()
    elif name == 'charbonnier':
        return CharbonnierLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier (smooth-L1) loss: sqrt((y-y_hat)^2 + eps^2)."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


def _exp_folder_name(params):
    """Folder for a single experiment, named after the backbone in use."""
    model = params.get('model', 'wavelet')
    pf = params['transform_params']['pf_fact']
    if model == 'nafnet':
        cfg = params.get('nafnet_config', 'small')
        arch = params.get('nafnet_arch_variant', 'v1')
        arch_tag = '' if arch == 'v1' else f'_{arch}'
        loss_name = params.get('loss', 'mse')
        loss_tag = '' if loss_name == 'mse' else f'_{loss_name}'
        # nsd + moderate-psf only describe the stack-of-spirals simulation;
        # the Cartesian pipeline applies neither, so tag it only for spiral.
        sim_tag = '_nsd_moderatepsf' if 'spiral' in params['exp_type'] else ''
        suffix = f'nafnet_{cfg}{arch_tag}{loss_tag}_imnet_pffact_{pf}_snrlog0-5{sim_tag}'
    else:
        nl = params['model_params']['nlayers']
        suffix = f'wavelet_unet_imnet_pffact_{pf}_snrlog0-6_nlayers_{nl}'
    return f'{params['exp_type']}/{suffix}'



def run_train(params, model):
    exp_folder_name = _exp_folder_name(params)

    model = model.load_model(
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        device=params['device']
    )

    train_loader, val_loader = init_loaders(
        exp_type=params['exp_type'],
        data_dir=params['train_data']['train_data_dir'],
        batch_size=params['training_params']['batch_size'],
        num_workers=params['train_data']['workers_per_gpu'] *
        torch.cuda.device_count(),
        gibb_size=params['transform_params']['gibb_size'],
        orig_big=params['transform_params']['orig_big'],
        pf_fact=params['transform_params']['pf_fact'],
        pe_fact=params['transform_params']['pe_fact'],
        dtype=params['dtype']
    )
    model = model.attach_data(
        exp_type=params['exp_type'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_data_dir=params['train_data']['train_data_dir']
    )

    optim_cfg = None
    total_iters = None
    if params.get('model', 'wavelet') == 'nafnet':
        optim_cfg = params.get('nafnet_optim', None)
        if optim_cfg is not None:
            total_iters = (
                int(params['training_params']['num_epochs'])
                * max(1, len(train_loader))
            )

    model = model.init_optimizer(
        learning_rate=params['training_params']['learning_rate'],
        loss_fn=_build_loss(params['loss']),
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        optim_cfg=optim_cfg,
        total_iters=total_iters,
    )

    model = model.fit(
        exp_dir=os.path.join(params['exp_dir'], exp_folder_name),
        num_epochs=params['training_params']['num_epochs']
    )

    return(0)


def run_val(params, model):
    exp_folder_name = _exp_folder_name(params)

    model = model.load_model(
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        device=params['device']
    )

    train_loader, val_loader = init_loaders(
        exp_type=params['exp_type'],
        data_dir=params['train_data']['train_data_dir'],
        batch_size=params['training_params']['batch_size'],
        num_workers=params['train_data']['workers_per_gpu'] *
        torch.cuda.device_count(),
        gibb_size=params['transform_params']['gibb_size'],
        orig_big=params['transform_params']['orig_big'],
        pf_fact=params['transform_params']['pf_fact'],
        dtype=params['dtype']
    )
    model = model.attach_data(
        exp_type=params['exp_type'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_data_dir=params['train_data']['train_data_dir']
    )

    visuals = {}
    val_loss, visuals = model.run_val_loop(
        visuals,
        rnd_seed=5
    )

    print(f'validation loss: {val_loss}')

    return(0)


def run_test(params, model):
    exp_folder_name = _exp_folder_name(params)

    metrics_file = os.path.join(
        params['exp_dir'], exp_folder_name, 'metrics_nonreduce.pkl')

    model = model.load_model(
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        device=params['device']
    )

    train_loader, val_loader = init_loaders(
        exp_type=params['exp_type'],
        data_dir=params['train_data']['train_data_dir'],
        batch_size=params['training_params']['batch_size'],
        num_workers=params['train_data']['workers_per_gpu'] *
        torch.cuda.device_count(),
        gibb_size=params['transform_params']['gibb_size'],
        orig_big=params['transform_params']['orig_big'],
        pf_fact=params['transform_params']['pf_fact'],
        test=True,
        test_dir=params['train_data']['test_data_dir'],
        dtype=params['dtype']
    )
    model = model.attach_data(
        exp_type=params['exp_type'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_data_dir=params['train_data']['train_data_dir']
    )
    model.epoch = 0
    model.loss_fn = _build_loss(params.get('loss', 'mse'))

    visuals = {}
    metrics, visuals = model.run_val_loop(
        visuals,
        rnd_seed=5,
        all_metrics=True
    )

    print(metrics)

    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    return(0)


def main():
    parser = generate_parser()

    # NAFNet backbone selection
    parser.add_argument(
        '--model',
        type=str,
        default='wavelet',
        choices=['wavelet', 'nafnet'],
        help='backbone architecture ("wavelet" or "nafnet")'
    )
    parser.add_argument(
        '--nafnet_config',
        type=str,
        default='small',
        help='NAFNet preset name (tiny/small/base) or path to a YAML file with presets'
    )
    parser.add_argument(
        '--noise_map_conditioning',
        action='store_true',
        help='NAFNet only: accept a per-pixel noise variance map as an extra input channel'
    )
    parser.add_argument(
        '--nafnet_arch_variant',
        type=str,
        default='v2',
        choices=['v1', 'v2'],
        help='NAFNet only: v1=stride-conv down + additive skip (legacy); '
             'v2=pixel-unshuffle down + concat skip (sharper)'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='mse',
        choices=['mse', 'l1', 'charbonnier'],
        help='Training loss function (mse / l1 / charbonnier)'
    )

    args = vars(parser.parse_args())

    params = read_yamls(args, dict())

    # Make backbone-selection args available to run_train/val/test via params.
    params['model'] = args.get('model', 'wavelet')
    params['nafnet_config'] = args.get('nafnet_config', 'small')
    params['noise_map_conditioning'] = args.get('noise_map_conditioning', False)
    params['nafnet_arch_variant'] = args.get('nafnet_arch_variant', 'v1')
    params['loss'] = args.get('loss', 'mse')

    model_params = params["model_params"]

    # Initialize DDP if launched via torchrun
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    params["device"] = device
    params["dtype"] = dtype
    params['dtype'] = dtype

    print(params)

    model = DeGibbs(
        nlayers=model_params['nlayers'],
        in_ch=model_params['in_ch'],
        out_ch=model_params['out_ch'],
        nchans=model_params['nchans'],
        comp2mag=model_params['comp2mag'],
        leaky=model_params['leaky'],
        device=torch.device('cpu'),
        dtype=dtype
    )

    # Swap in NAFNet backbone when requested
    if args.get('model', 'wavelet') == 'nafnet':
        backbone = build_model(
            model='nafnet',
            in_ch=model_params['in_ch'],
            out_ch=model_params['out_ch'],
            comp2mag=model_params['comp2mag'],
            nafnet_config=args.get('nafnet_config', 'small'),
            noise_map_conditioning=args.get('noise_map_conditioning', False),
            arch_variant=args.get('nafnet_arch_variant', 'v1'),
        )
        model.model = backbone.to(device=torch.device('cpu'), dtype=dtype)

    if params['mode'] == 'train':
        run_train(params, model)
    elif params['mode'] == 'val':
        run_val(params, model)
    elif params['mode'] == 'test':
        run_test(params, model)

    if dist.is_initialized():
        dist.destroy_process_group()
    return(0)


if __name__ == "__main__":
    sys.exit(main())
