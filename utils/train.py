import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import scipy

from utils import commons
from utils import utils
from utils.data_utils import (
    TextAudioLoaderWithDuration,
    TextAudioCollateWithDuration,
    DistributedBucketSampler,
)
from models.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from models.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss_dtw,
    kl_loss,
)
from utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "63331"

    hps = utils.get_hparams()
    
    if hps.warmup and not hps.train.use_gt_duration:
        print("'use_gt_duration' option is automatically set to true in warmup phase.")
        hps.train.use_gt_duration = True
    
    if hps.warmup:
        print("'c_kl_fwd' is set to 0 during warmup to learn a reasonable prior distribution.")  
        hps.train.c_kl_fwd = 0

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioLoaderWithDuration(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioCollateWithDuration()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioLoaderWithDuration(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        hps.models,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator().cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if not hps.warmup:
        net_g.attach_memory_bank(hps.models)
        optim_g.add_param_group({"params": list(net_g.memory_bank.parameters())})

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = epoch_str * len(train_loader)
    except Exception as e:
        epoch_str = 0
        global_step = 0

    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 1
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if epoch % hps.train.eval_interval == 0:
            if rank == 0:
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(epoch)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(epoch)),
                )
                evaluate(hps, net_g, eval_loader, writer_eval, epoch=epoch)

        if hps.warmup and epoch >= hps.train.warmup_epochs:
            if rank == 0:
                logger.info("Epoch {epoch}: done warmup")
            break

        if rank == 0:
            train(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                train_loader,
                logger,
                writer,
            )
        else:
            train(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                train_loader,
                None,
                None,
            )

        scheduler_g.step()
        scheduler_d.step()


def train(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    train_loader,
    logger,
    writer=None,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        duration,
    ) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        duration = duration.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q, p_mask),
                W,
                y_hat_e2e,
                z_q,
                d,
                ids_slice_q,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                d=duration,
                use_gt_duration=hps.train.use_gt_duration,
            )
            y1 = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            y2 = commons.slice_segments(
                y, ids_slice_q * hps.data.hop_length, hps.train.segment_size
            )

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y1, y_hat.detach())
            y_d_hat_r_e2e, y_d_hat_g_e2e, _, _ = net_d(y2, y_hat_e2e.detach())

            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )

            loss_disc_e2e, _, _ = discriminator_loss(y_d_hat_r_e2e, y_d_hat_g_e2e)
            loss_disc_e2e *= hps.train.c_e2e

            loss_disc_all = loss_disc + loss_disc_e2e

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            # reconstruction loss (feature mapping, gan loss)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y1, y_hat)

            # reconstruction loss (mel spectrogram loss)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

            # e2e loss
            y_d_hat_r_e2e, y_d_hat_g_e2e, _, _ = net_d(y2, y_hat_e2e)
            loss_gen_e2e, losses_gen_e2e = generator_loss(y_d_hat_g_e2e)
            loss_gen_e2e *= hps.train.c_e2e

            # bwd, fwd loss
            if hps.train.use_sdtw:
                loss_kl = (
                    kl_loss_dtw(z_p, logs_q, m_p, logs_p, p_mask, z_mask.squeeze(1))
                    * hps.train.c_kl
                )
                loss_kl_fwd = (
                    kl_loss_dtw(z_q, logs_p, m_q, logs_q, z_mask.squeeze(1), p_mask)
                    * hps.train.c_kl_fwd
                )
            else:
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_kl_fwd = (
                    kl_loss(z_q, logs_p, m_q, logs_q, p_mask.unsqueeze(1))
                    * hps.train.c_kl_fwd
                )

            loss_dur = torch.sum(l_length.float()) * hps.train.c_dur

            loss_gen_all = (
                loss_gen
                + loss_gen_e2e
                + loss_fm
                + loss_mel
                + loss_dur
                + loss_kl
                + loss_kl_fwd
            )

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [
                    loss_disc,
                    loss_disc_e2e,
                    loss_gen,
                    loss_gen_e2e,
                    loss_fm,
                    loss_mel,
                    loss_dur,
                    loss_kl,
                    loss_kl_fwd,
                ]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info(
                    "[loss_disc, loss_disc_e2e, loss_gen, loss_gen_e2e, loss_fm, loss_mel, loss_dur, loss_kl, loss_kl_fwd, global_step, lr]"
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                        "loss/g/kl_fwd": loss_kl_fwd,
                    }
                )

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/W0": utils.plot_spectrogram_to_numpy(
                        W[0, 0, :, :].transpose(0, 1).data.cpu().numpy()
                    ),
                    "all/W1": utils.plot_spectrogram_to_numpy(
                        W[0, 1, :, :].transpose(0, 1).data.cpu().numpy()
                    ),
                    "all/W2": utils.plot_spectrogram_to_numpy(
                        W[0, 2, :, :].transpose(0, 1).data.cpu().numpy()
                    ),
                    "all/W3": utils.plot_spectrogram_to_numpy(
                        W[0, 3, :, :].transpose(0, 1).data.cpu().numpy()
                    ),
                }

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval, epoch=0):
    generator.eval()

    save_dir = os.path.join(writer_eval.log_dir, f"{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            duration,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]

            y_hat, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            audio = y_hat[0, 0, : y_hat_lengths[0]].cpu().numpy()
            audio_gt = y[0, 0, : y_lengths[0]].cpu().numpy()
            scipy.io.wavfile.write(
                filename=os.path.join(save_dir, f"{batch_idx}.wav"),
                rate=hps.data.sampling_rate,
                data=audio,
            )
            scipy.io.wavfile.write(
                filename=os.path.join(save_dir, f"{batch_idx}_gt.wav"),
                rate=hps.data.sampling_rate,
                data=audio_gt,
            )

            if batch_idx >= 8:
                break

    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    print("====> Epoch Evaluate: {}".format(epoch))


if __name__ == "__main__":
    main()
