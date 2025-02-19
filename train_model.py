import argparse
import wandb
from torch import nn
from geoopt import PoincareBall
from geoopt.optim import RiemannianAdam
from src.models import ExpMap0, LogMap0, UnidirectionalPoincareMLR, AutoEncoder
from src.utils import fix_seed
from src.dataset import get_loaders, get_data
from src.train import train
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from src.common import str2layer


def main(
    emb_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    model_encoder: str,
    model_decoder: str,
    seed: int,
    data_dir: str,
    data_name: str,
    dataset_type: str,
    num_workers: int,
    temperature: float,
    dtype: str,
    device: str,
    ks: list[int],
    show_progress: bool,
    log: bool,
    conf: dict,
):
    assert (model_encoder == 'Euc') == (model_decoder == 'Euc'), 'Can not mix euclidian and hyperbolic layers'
    fix_seed(seed)
    
    if log:
        run = wandb.init( # noqa
            project="HypRecSys",
            config=conf
        )
    print('Prepare data')
    _, train_val_loader, val_loader, test_loader = get_loaders(*get_data(data_dir, data_name, dataset_type), batch_size, num_workers)
    num_items = test_loader.dataset.num_items

    print('Constructing model')

    dtype = getattr(torch, dtype)

    ball = PoincareBall(c=10, learnable=False)

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
    
    layers = [encoder, decoder]
    if not isinstance(encoder, nn.Linear):
        layers = [ExpMap0(encoder.manifold)] + layers

    model = nn.Sequential(*layers).to(device)
    # model = AutoEncoder(num_items, 2, emb_dim, dtype=dtype).to(device)

    optimizer = RiemannianAdam(model.parameters(), lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    scheduler1 = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=epochs // 10
    )
    scheduler2 = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs - epochs // 10
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[epochs // 10]
    )

    print('Starting train')

    history = train(
        model, 
        optimizer, 
        criterion, 
        scheduler, 
        train_val_loader, 
        test_loader, 
        dataset_type, 
        epochs, 
        device, 
        ks, 
        temperature=temperature, 
        show_progress=show_progress, 
        log=log
    )

    print(f'Final results: {max(history, key=lambda x: x["hr@20"])}')
    print(f'Final curvature: {model[-1].manifold.c.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dim", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])

    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log", default=False, action="store_true")

    conf = vars(parser.parse_args())

    main(**conf, conf=conf)