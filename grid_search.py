import argparse
import json
from copy import copy

import torch
import torch.utils
from geoopt import PoincareBall
from geoopt.optim import RiemannianAdam
from torch import nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import trange

import wandb
from src.dataset import get_data, get_loaders
from src.models import (
    ExpMap0,
    LogMap0,
    UnidirectionalPoincareMLR,
)
from src.train import train
from src.utils import fix_seed
from src.common import str2layer


def main(
    emb_dim: list[int],
    lr: list[float],
    batch_size: list[int],
    c: list[float],
    epochs: int,
    model_encoder: str,
    model_decoder: str,
    seed: int,
    data_dir: str,
    data_name: str,
    dataset_type: str,
    num_workers: int,
    dtype: str,
    device: str,
    ks: list[int],
    show_progress: bool,
    log: bool,
    conf: dict,
):
    assert (model_encoder == "Euc") == (model_decoder == "Euc"), (
        "Can't mix euclidian and hyperbolic layers"
    )
    fix_seed(seed)

    if log:
        run = wandb.init( # noqa
            project="HypRecSys", name=f"{model_encoder}_{model_decoder}", config=conf
        )

    print("Prepare data")
    train_dataset, train_val_dataset, val_dataset, test_dataset = get_data(
        data_dir, data_name, dataset_type
    )

    lr_grid = torch.tensor(lr, dtype=torch.float32)
    emb_dim_grid = torch.tensor(emb_dim, dtype=torch.float32)
    batch_sizes_grid = torch.tensor(batch_size, dtype=torch.float32)
    c_grid = torch.tensor(c, dtype=torch.float32)

    lr_grid, emb_dim_grid, batch_sizes_grid, c_grid = torch.meshgrid(
        lr_grid, emb_dim_grid, batch_sizes_grid, c_grid
    )

    lr_grid = lr_grid.flatten()
    emb_dim_grid = emb_dim_grid.flatten()
    batch_sizes_grid = batch_sizes_grid.flatten()
    c_grid = c_grid.flatten()

    best_hparams = {}
    best_val_scores = None

    print("Start grid search")

    for i in trange(lr_grid.shape[0]):
        lr = lr_grid[i].item()
        emb_dim = int(emb_dim_grid[i].item())
        batch_size = int(batch_sizes_grid[i].item())
        c = c_grid[i].item()

        history = train_end_to_end(
            emb_dim,
            lr,
            batch_size,
            c,
            epochs,
            model_encoder,
            model_decoder,
            train_dataset,
            train_val_dataset,
            val_dataset,
            test_dataset,
            dataset_type,
            num_workers,
            dtype,
            device,
            ks,
            show_progress=show_progress
        )

        val_scores = max(history, key=lambda x: x["hr@20"])

        if best_val_scores is None or val_scores["hr@20"] > best_val_scores["hr@20"]:
            best_val_scores = copy(val_scores)
            best_hparams = {"lr": lr, "emb_dim": emb_dim, "batch_size": batch_size, "c": c}

    lr, emb_dim, batch_size, c = (
        best_hparams["lr"],
        best_hparams["emb_dim"],
        best_hparams["batch_size"],
        best_hparams["c"],
    )

    history = train_end_to_end(
        emb_dim,
        lr,
        batch_size,
        c,
        epochs,
        model_encoder,
        model_decoder,
        train_dataset,
        train_val_dataset,
        val_dataset,
        test_dataset,
        dataset_type,
        num_workers,
        dtype,
        device,
        ks,
        mode="test",
        show_progress=show_progress,
        log=log,
    )
    final_result = max(history, key=lambda x: x["hr@20"])

    with open("grid_search_result.json", "w") as fp:
        json.dump(
            {
                "test_scores": final_result,
                "val_scores": best_val_scores,
                "best_hparams": best_hparams,
            },
            fp,
        )
    if log:
        artifact = wandb.Artifact(
            name=f"{model_encoder}_{model_decoder}", type="grid_search_result"
        )
        artifact.add_file(local_path="grid_search_result.json")
        artifact.save()


def train_end_to_end(
    emb_dim: int,
    lr: float,
    batch_size: int,
    c: float,
    epochs: int,
    model_encoder: str,
    model_decoder: str,
    train_dataset: torch.utils.data.Dataset,
    train_val_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    dataset_type: str,
    num_workers: int,
    dtype: str,
    device: str,
    ks: list[int],
    mode: str = "train",
    show_progress: bool = False,
    log: bool = False,
):
    train_loader, train_val_loader, val_loader, test_loader = get_loaders(
        train_dataset,
        train_val_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
    )
    num_items = test_loader.dataset.num_items
    if mode == "test":
        train_loader = train_val_loader
        val_loader = test_loader

    dtype = getattr(torch, dtype)

    ball = PoincareBall(c, learnable=True)

    encoder = str2layer[model_encoder](
        in_features=num_items,
        out_features=emb_dim,
        dtype=dtype,
        **{"manifold": ball} if model_encoder != "Euc" else {},
    )
    decoder = str2layer[model_decoder](
        in_features=emb_dim,
        out_features=num_items,
        dtype=dtype,
        **{"manifold": ball} if model_decoder != "Euc" else {},
    )
    if not isinstance(encoder, nn.Linear):
        layers = [ExpMap0(encoder.manifold), encoder, decoder]
    else:
        layers = [encoder, nn.ReLU(), decoder]

    # if not isinstance(encoder, nn.Linear) and not isinstance(
    #     decoder, UnidirectionalPoincareMLR
    # ):
    #     layers = layers + [LogMap0(decoder.manifold)]

    model = nn.Sequential(*layers).to(device)

    optimizer = RiemannianAdam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler1 = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=epochs // 10
    )
    scheduler2 = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs - epochs // 10
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[epochs // 10]
    )

    criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    return train(
        model,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        val_loader,
        dataset_type,
        epochs,
        device,
        ks,
        show_progress=show_progress,
        log=log
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--emb_dim", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4])
    parser.add_argument("--batch_size", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--c", type=float, nargs="+", default=[1.0, 1e-3, 1e-6])

    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="bag_of_items",
        choices=["bag_of_items", "sequential"],
    )

    parser.add_argument(
        "--model_encoder",
        type=str,
        choices=["Euc", "HypLinear", "Mobius", "PoincareLinear"],
        required=True,
    )
    parser.add_argument(
        "--model_decoder",
        type=str,
        choices=["Euc", "HypLinear", "Mobius", "HypMLR", "PoincareLinear"],
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])

    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log", default=False, action="store_true")

    conf = vars(parser.parse_args())

    main(**conf, conf=conf)
