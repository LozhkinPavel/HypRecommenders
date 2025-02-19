import os
import argparse

import torch
import torch.utils
from geoopt import PoincareBall, Lorentz
from geoopt.optim import RiemannianAdam
from torch import nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

import wandb
from src.dataset import get_data, get_loaders
from src.models import (
    LogMap0,
    UnidirectionalPoincareMLR,
    PoincareEmbedding
)
from src.train import train, train_embeddings
from src.utils import fix_seed
from src.common import str2layer


def main(
    emb_dim: int,
    lr: float,
    embedding_lr: float,
    burn_in_lr_factor: float,
    batch_size: int,
    embedding_batch_size: int,
    epochs: int,
    embedding_epochs: int,
    burn_in_epochs: int,
    model_decoder: str,
    seed: int,
    data_dir: str,
    data_name: str,
    dataset_type: str,
    embedding_dataset_type: str,
    embedding_num_negatives: int,
    manifold: str,
    num_workers: int,
    dtype: str,
    device: str,
    ks: list[int],
    freeze_embeddings: bool,
    show_progress: bool,
    log: bool,
    is_test: bool,
    conf: dict,
):
    fix_seed(seed)

    if log:
        run = wandb.init( # noqa
            project="HypRecSys", name=f"Embedding_{model_decoder}", config={
                "_wandb": {
                    "define_metrics": {
                        "*": {"step_metric": "custom_step"}
                    }
                },
                **conf
            }
        )

    print("Prepare data")
    train_dataset, train_val_dataset, val_dataset, test_dataset, embedding_dataset = get_data(
        data_dir, data_name, dataset_type, embedding_dataset_type=embedding_dataset_type, embedding_num_negatives=embedding_num_negatives
    )

    print("Start train")

    if os.path.isfile(f"embeddings_{emb_dim}.pth"):
        manifold = PoincareBall(c=1e-5, learnable=False) if manifold == "poincare" else Lorentz(k=1e-5)
        model_encoder = PoincareEmbedding(embedding_dataset.num_items, emb_dim, manifold, dtype=getattr(torch, dtype)).to(device)
        model_encoder.load_state_dict(torch.load(f"embeddings_{emb_dim}.pth", map_location=device))
    else:
        _, model_encoder = train_embedding(
            emb_dim, 
            embedding_lr, 
            burn_in_lr_factor, 
            embedding_batch_size, 
            embedding_epochs, 
            burn_in_epochs, 
            embedding_dataset, 
            manifold,
            num_workers, 
            dtype, 
            device, 
            show_progress, 
            log
        )

    torch.save(model_encoder.state_dict(), f"embeddings_{emb_dim}.pth")

    print(torch.any((torch.sum(model_encoder.matrix**2, dim=-1) - 2 * model_encoder.matrix[:, 0]**2) != -1))

    if freeze_embeddings:
        for param in model_encoder.parameters():
            param.requires_grad = False

    for param in model_encoder.parameters():
        print(param.requires_grad)

    final_result = train_recomendations(
        emb_dim,
        lr,
        batch_size,
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
        show_progress=show_progress,
        log=log,
        is_test=is_test
    )[-1]

    print("Final results: ", final_result)

    if log:
        artifact = wandb.Artifact(
            name=f"Embedding_{model_decoder}", type="embeddings"
        )
        artifact.add_file(local_path=f"embeddings_{emb_dim}.pth")
        artifact.save()


def train_embedding(
    emb_dim: int,
    lr: float,
    burn_in_lr_factor: float,
    batch_size: int,
    epochs: int,
    burn_in_epochs: int,
    embedding_dataset: torch.utils.data.Dataset,
    manifold: str,
    num_workers: int,
    dtype: str,
    device: str,
    show_progress: bool = False,
    log: bool = False,
):
    dtype = getattr(torch, dtype)
    manifold = PoincareBall(c=1e-5, learnable=False) if manifold == "poincare" else Lorentz(k=1e-5)
    model = PoincareEmbedding(embedding_dataset.num_items, emb_dim, manifold, dtype=dtype).to(device)
    optimizer = RiemannianAdam(model.parameters(), lr, eps=1e-15, weight_decay=1e-12, stabilize=1)
    train_loader = torch.utils.data.DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    scheduler1 = ConstantLR(optimizer, factor=burn_in_lr_factor, total_iters=burn_in_epochs)
    scheduler2 = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs - burn_in_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[burn_in_epochs])

    train_loader.dataset.burnin = True
    criterion = nn.CrossEntropyLoss().to(device)

    history = train_embeddings(model, optimizer, criterion, scheduler, train_loader, embedding_dataset.num_interactions, epochs, burn_in_epochs, device, show_progress, log)

    return history, model


def train_recomendations(
    emb_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    model_encoder: PoincareEmbedding,
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
    show_progress: bool = False,
    log: bool = False,
    is_test: bool = False
):
    train_loader, train_val_loader, val_loader, test_loader = get_loaders(
        train_dataset,
        train_val_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        num_workers,
    )
    if is_test:
        train_loader = train_val_loader
        val_loader = test_loader
    num_items = test_loader.dataset.num_items

    dtype = getattr(torch, dtype)

    ball = PoincareBall(c=1e-5, learnable=True)

    decoder = str2layer[model_decoder](
        in_features=emb_dim,
        out_features=num_items,
        dtype=dtype,
        manifold=ball
    )
    layers = [model_encoder, decoder]


    # if not isinstance(decoder, UnidirectionalPoincareMLR):
    #     layers = layers + [LogMap0(decoder.manifold)]

    model = nn.Sequential(*layers).to(device)

    optimizer = RiemannianAdam(model.parameters(), lr=lr)
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

    # model.compile()
    # criterion.compile()

    history = train(
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
        log=log,
    )

    print("Final curvature: ", torch.abs(decoder.manifold.k).item())

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dim", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--embedding_lr", type=float, required=True)
    parser.add_argument("--burn_in_lr_factor", type=float, default=1e-2)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_batch_size", type=int, default=64)

    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="bag_of_items",
        choices=["bag_of_items", "sequential"],
    )

    parser.add_argument(
        "--manifold",
        type=str,
        default="poincare",
        choices=["poincare", "lorentz"],
    )

    parser.add_argument(
        "--embedding_dataset_type",
        type=str,
        default="similarity",
        choices=["similarity", "tree"],
    )

    parser.add_argument(
        "--embedding_num_negatives",
        type=int,
        default=10
    )

    parser.add_argument(
        "--model_decoder",
        type=str,
        choices=["HypLinear", "Mobius", "HypMLR", "PoincareLinear"],
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--embedding_epochs", type=int, default=20)
    parser.add_argument("--burn_in_epochs", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])

    parser.add_argument("--freeze_embeddings", default=False, action="store_true")
    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log", default=False, action="store_true")
    parser.add_argument("--is_test", default=False, action="store_true")

    conf = vars(parser.parse_args())

    main(**conf, conf=conf)
