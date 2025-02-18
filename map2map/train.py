import os
import socket
import time
import sys
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
# For CPU-only, we don't need torch.distributed or DDP.
# import torch.distributed as dist
from torch.multiprocessing import spawn
# from torch.nn.parallel import DistributedDataParallel  # Removed for CPU-only mode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import FieldDataset, DistFieldSampler  # Note: DistFieldSampler is not used in CPU-only mode.
from . import models
from .models import (
    narrow_cast, resample,
    grad_penalty_reg,
    add_spectral_norm,
    InstanceNoise,
)
from .utils import import_attr, load_model_state_dict, plt_slices, plt_power

ckpt_link = 'checkpoint.pt'


def node_worker(args):
    """
    In CPU-only mode, we bypass SLURM/distributed setups.
    Force a single node and single process and call gpu_worker directly.
    """
    args.nodes = 1
    args.gpus_per_node = 1
    args.world_size = 1
    # Directly call the worker with local_rank=0 and node=0.
    gpu_worker(0, 0, args)


def gpu_worker(local_rank, node, args):
    """
    CPU-only worker.
    Sets device to CPU, bypasses distributed initialization,
    and uses standard DataLoaders without distributed samplers.
    """
    device = torch.device('cpu')
    rank = 0  # Only one process in CPU-only mode.
    torch.manual_seed(args.seed + rank)

    # --- Bypass distributed initialization ---
    # (We do not call any distributed initialization since we are using a single process.)

    # --- Build training dataset and DataLoader (no distributed sampler) ---
    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        pin_memory=False,  # pin_memory is not needed on CPU
    )

    # --- Build validation DataLoader if validation data is provided ---
    if args.val:
        val_dataset = FieldDataset(
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
            in_norms=args.in_norms,
            tgt_norms=args.tgt_norms,
            callback_at=args.callback_at,
            augment=False,
            aug_shift=None,
            aug_add=None,
            aug_mul=None,
            crop=args.crop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop,
            crop_step=args.crop_step,
            in_pad=args.in_pad,
            tgt_pad=args.tgt_pad,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.loader_workers,
            pin_memory=False,
        )
    else:
        val_loader = None

    # Update channel information (used later by the model)
    args.in_chan, args.out_chan = train_dataset.in_chan, train_dataset.tgt_chan

    # --- Instantiate the Model ---
    ModelClass = import_attr(args.model, models, callback_at=args.callback_at)
    model = ModelClass(sum(args.in_chan), sum(args.out_chan),
                       scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)
    # Do NOT wrap the model in DistributedDataParallel in CPU-only mode.
    # model = DistributedDataParallel(model, device_ids=[device], process_group=dist.new_group())

    # --- Instantiate the Loss Criterion, Optimizer, and Scheduler ---
    criterion = import_attr(args.criterion, nn, models, callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        **args.optimizer_args,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.scheduler_args)

    # --- Set Up Adversarial Components if Enabled ---
    adv_model = adv_criterion = adv_optimizer = adv_scheduler = None
    if args.adv:
        AdvModelClass = import_attr(args.adv_model, models, callback_at=args.callback_at)
        adv_model = AdvModelClass(
            sum(args.in_chan + args.out_chan) if args.cgan else sum(args.out_chan),
            1,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        if args.adv_model_spectral_norm:
            add_spectral_norm(adv_model)
        adv_model.to(device)
        # Do not wrap in DDP.
        adv_criterion = import_attr(args.adv_criterion, nn, models, callback_at=args.callback_at)
        adv_criterion = adv_criterion()
        adv_criterion.to(device)
        adv_optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
        adv_optimizer = adv_optimizer(
            adv_model.parameters(),
            lr=args.adv_lr,
            **args.adv_optimizer_args,
        )
        adv_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            adv_optimizer, **args.scheduler_args)

    # --- Load Model State if Provided ---
    if (args.load_state == ckpt_link and not os.path.isfile(ckpt_link)) or not args.load_state:
        if args.init_weight_std is not None:
            model.apply(init_weights)
            if args.adv:
                adv_model.apply(init_weights)
        start_epoch = 0
        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)
        start_epoch = state['epoch']
        load_model_state_dict(model, state['model'], strict=args.load_state_strict)
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        if args.adv:
            if 'adv_model' in state:
                load_model_state_dict(adv_model, state['adv_model'], strict=args.load_state_strict)
            if 'adv_optimizer' in state:
                adv_optimizer.load_state_dict(state['adv_optimizer'])
            if 'adv_scheduler' in state:
                adv_scheduler.load_state_dict(state['adv_scheduler'])
        torch.set_rng_state(state['rng'].cpu())
        if rank == 0:
            min_loss = state['min_loss']
            if args.adv and 'adv_model' not in state:
                min_loss = None
            print('state at epoch {} loaded from {}'.format(state['epoch'], args.load_state), flush=True)
        del state

    # --- Set Backend Benchmark ---
    # On CPU, disable cudnn.benchmark.
    torch.backends.cudnn.benchmark = False

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # --- Initialize Logger (TensorBoard) ---
    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        print('pytorch {}'.format(torch.__version__))
        pprint(vars(args))
        sys.stdout.flush()

    if args.adv:
        args.instance_noise = InstanceNoise(args.instance_noise, args.instance_noise_batches)

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(epoch, train_loader,
                           model, criterion, optimizer, scheduler,
                           adv_model, adv_criterion, adv_optimizer, adv_scheduler,
                           logger, device, args)
        epoch_loss = train_loss

        if args.val:
            val_loss = validate(epoch, val_loader,
                                model, criterion, adv_model, adv_criterion,
                                logger, device, args)
            # Optionally, you can set epoch_loss = val_loss for monitoring validation loss.

        if args.reduce_lr_on_plateau and epoch >= args.adv_start:
            scheduler.step(epoch_loss[0])
            if args.adv:
                adv_scheduler.step(epoch_loss[0])

        if rank == 0:
            logger.flush()

            if ((min_loss is None or epoch_loss[0] < min_loss[0])
                    and epoch >= args.adv_start):
                min_loss = epoch_loss

            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }
            if args.adv:
                state.update({
                    'adv_model': adv_model.state_dict(),
                    'adv_optimizer': adv_optimizer.state_dict(),
                    'adv_scheduler': adv_scheduler.state_dict(),
                })

            state_file = 'state_{}.pt'.format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = '{}.pt'.format(time.time())
            os.symlink(state_file, tmp_link)  # Workaround to allow overwriting.
            os.rename(tmp_link, ckpt_link)

    # End of training loop.
    # (No need to destroy any distributed process group in CPU-only mode.)


def train(epoch, loader, model, criterion, optimizer, scheduler,
          adv_model, adv_criterion, adv_optimizer, adv_scheduler,
          logger, device, args):
    model.train()
    if args.adv:
        adv_model.train()

    total_loss = 0.0

    for i, data in enumerate(loader):
        batch = epoch * len(loader) + i + 1

        input, target = data['input'], data['target']
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        
        # Compute loss
        loss = criterion(output, target)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training images and power spectrum
        if batch % args.log_interval == 0:
            log_images(logger, input, target, output, batch, "train")
            log_power_spectrum(logger, input[0].cpu().numpy(), target[0].cpu().numpy(), output[0].cpu().numpy(), batch)

    # Log average training loss
    avg_loss = total_loss / len(loader)
    logger.add_scalar('Loss/train', avg_loss, epoch)

    # Update the learning rate scheduler
    scheduler.step(avg_loss)

    return avg_loss

def log_images(writer, lr, hr, sr, step, tag="train"):
    """Log 2D slices of 3D volumes to tensorboard."""
    import torch.nn.functional as F
    
    # Take center slices from each dimension
    for dim, name in enumerate(['XY', 'YZ', 'XZ']):
        # Get middle slice indices for each dimension
        idx = [slice(None)] * len(lr.shape)
        idx[2 + dim] = lr.shape[2 + dim] // 2  # +2 because of batch and channel dims
        
        # Extract slices
        lr_slice = lr[tuple(idx)]
        hr_slice = hr[tuple(idx)]
        sr_slice = sr[tuple(idx)]
        
        # Create coordinate grids for proper physical scaling
        canvas_size = hr_slice.shape[-2:]  # Use HR size for canvas
        lr_start = canvas_size[0]//4  # Start at 1/4 of canvas
        lr_end = canvas_size[0]//4 * 3  # End at 3/4 of canvas
        
        # Create empty canvases (same size as HR)
        lr_canvas = torch.zeros_like(hr_slice)
        sr_canvas = torch.zeros_like(hr_slice)
        
        # Place LR in center of its canvas
        lr_canvas[..., lr_start:lr_end, lr_start:lr_end] = F.interpolate(
            lr_slice, 
            size=(lr_end-lr_start, lr_end-lr_start), 
            mode='nearest'
        )
        
        # Ensure SR is same size as HR
        sr_canvas = F.interpolate(sr_slice, size=canvas_size, mode='nearest')
        
        # Stack slices side by side
        comparison = torch.cat([
            lr_canvas[0],      # Low res in same physical volume
            hr_slice[0],       # Original high res
            sr_canvas[0]       # Super-resolution output
        ], dim=1)
        
        # Normalize to [0,1] for visualization
        comparison = (comparison - comparison.min()) / (comparison.max() - comparison.min() + 1e-8)
        
        # Log to tensorboard
        writer.add_image(f'Slices_{name}/{tag}', comparison, global_step=step)

    # Also log some statistics
    writer.add_scalar(f'Stats/{tag}/lr_mean', lr.mean().item(), global_step=step)
    writer.add_scalar(f'Stats/{tag}/hr_mean', hr.mean().item(), global_step=step)
    writer.add_scalar(f'Stats/{tag}/sr_mean', sr.mean().item(), global_step=step)


def validate(epoch, loader, model, criterion, adv_model, adv_criterion,
             logger, device, args):
    model.eval()
    if args.adv:
        adv_model.eval()

    rank = 0
    world_size = 1
    epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)
    fake = torch.zeros([1], dtype=torch.float32, device=device)
    real = torch.ones([1], dtype=torch.float32, device=device)

    with torch.no_grad():
        for data in loader:
            input, target = data['input'], data['target']
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            if hasattr(model, 'scale_factor') and model.scale_factor != 1:
                input = resample(input, model.scale_factor, narrow=False)
            input, output, target = narrow_cast(input, output, target)

            loss = criterion(output, target)
            epoch_loss[0] += loss.detach()

            if args.adv and epoch >= args.adv_start:
                if args.cgan:
                    output = torch.cat([input, output], dim=1)
                    target = torch.cat([input, target], dim=1)

                score_out = adv_model(output)
                adv_loss_fake = adv_criterion(score_out, fake.expand_as(score_out))
                epoch_loss[3] += adv_loss_fake.detach()

                score_tgt = adv_model(target)
                adv_loss_real = adv_criterion(score_tgt, real.expand_as(score_tgt))
                epoch_loss[4] += adv_loss_real.detach()

                adv_loss = adv_loss_fake + adv_loss_real
                epoch_loss[2] += adv_loss.detach()

                loss_adv = adv_criterion(score_out, real.expand_as(score_out))
                epoch_loss[1] += loss_adv.detach()

    epoch_loss /= (len(loader) * world_size)
    logger.add_scalar('loss/epoch/val', epoch_loss[0], global_step=epoch + 1)
    if args.adv and epoch >= args.adv_start:
        logger.add_scalar('loss/epoch/val/adv/G', epoch_loss[1], global_step=epoch + 1)
        logger.add_scalars('loss/epoch/val/adv/D', {
            'total': epoch_loss[2],
            'fake': epoch_loss[3],
            'real': epoch_loss[4],
        }, global_step=epoch + 1)

    fig = plt_slices(
        input[-1], output[-1], target[-1],
        output[-1] - target[-1],
        title=['in', 'out', 'tgt', 'out - tgt'],
        **args.misc_kwargs,
    )
    logger.add_figure('fig/val', fig, global_step=epoch + 1)
    fig.clf()

    fig = plt_power(
        input, output, target,
        label=['in', 'out', 'tgt'],
        **args.misc_kwargs,
    )
    logger.add_figure('fig/val/power/lag', fig, global_step=epoch + 1)
    fig.clf()

    return epoch_loss


def init_weights(m):
    # Initialize weights for layers if applicable.
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight.data.normal_(0.0, getattr(m, 'init_weight_std', 0.02))
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                        nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
                        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if m.affine:
            m.weight.data.normal_(1.0, getattr(m, 'init_weight_std', 0.02))
            m.bias.data.fill_(0)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    """Return the norm of the gradients for the first and last weight layers."""
    grads = [p.grad.detach().norm() for n, p in model.named_parameters() if '.weight' in n]
    return [grads[0], grads[-1]]

