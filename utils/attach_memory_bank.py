import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from text.symbols import symbols
from models.models import SynthesizerTrn
from models.models import VAEMemoryBank
from utils import utils

from utils.data_utils import (
    TextAudioLoaderWithDuration,
    TextAudioCollateWithDuration,
)

from sklearn.cluster import KMeans


def load_net_g(hps, weights_path):
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        hps.models,
    ).cuda()

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    def load_checkpoint(checkpoint_path, model, optimizer=None):
        assert os.path.isfile(checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        iteration = checkpoint_dict["iteration"]
        learning_rate = checkpoint_dict["learning_rate"]

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        saved_state_dict = checkpoint_dict["model"]

        state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except:
                print("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

        print(
            "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
        )
        return model, optimizer, learning_rate, iteration

    model, optimizer, learning_rate, iteration = load_checkpoint(
        weights_path, net_g, optim_g
    )

    return model, optimizer, learning_rate, iteration


def get_dataloader(hps):
    train_dataset = TextAudioLoaderWithDuration(hps.data.training_files, hps.data)
    collate_fn = TextAudioCollateWithDuration()
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        batch_size=1,
    )
    return train_loader


def get_zs(net_g, dataloader, num_samples=0):
    net_g.eval()
    print(len(dataloader))
    zs = []
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            duration,
        ) in enumerate(dataloader):
            rank = 0
            x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
                rank, non_blocking=True
            )
            spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
                rank, non_blocking=True
            )
            y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
                rank, non_blocking=True
            )
            duration = duration.cuda()
            with autocast(enabled=hps.train.fp16_run):
                (
                    y_hat,
                    l_length,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q, p_mask),
                    *_,
                ) = net_g(x, x_lengths, spec, spec_lengths, duration)

            zs.append(z.squeeze(0).cpu())
            if batch_idx % 100 == 99:
                print(batch_idx, zs[batch_idx].shape)

            if num_samples and batch_idx >= num_samples:
                break
    return zs


def k_means(zs):
    X = torch.cat(zs, dim=1).transpose(0, 1).numpy()
    print(X.shape)
    kmeans = KMeans(n_clusters=1000, random_state=0, n_init="auto").fit(X)
    print(kmeans.cluster_centers_.shape)

    return kmeans.cluster_centers_


def save_memory_bank(bank):
    state_dict = bank.state_dict()
    torch.save(state_dict, "./bank_init.pth")


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )
    print("Saving model to " + checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/ljs.json")
    parser.add_argument("--weights_path", type=str)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="samples to use for k-means clustering, 0 for use all samples in dataset",
    )
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    net_g, optimizer, lr, iterations = load_net_g(hps, weights_path=args.weights_path)

    dataloader = get_dataloader(hps)
    zs = get_zs(net_g, dataloader, num_samples=args.num_samples)
    centers = k_means(zs)

    memory_bank = VAEMemoryBank(
        **hps.models.memory_bank,
        init_values=torch.from_numpy(centers).cuda().transpose(0, 1)
    )
    save_memory_bank(memory_bank)

    net_g.memory_bank = memory_bank
    optimizer.add_param_group(
        {
            "params": list(memory_bank.parameters()),
            "initial_lr": optimizer.param_groups[0]["initial_lr"],
        }
    )

    p = Path(args.weights_path)
    save_path = p.with_stem(p.stem + "_with_memory").__str__()
    save_checkpoint(net_g, optimizer, lr, iterations, save_path)

    # test
    print(memory_bank(torch.randn((2, 192, 12))).shape)
