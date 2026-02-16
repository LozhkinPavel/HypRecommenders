import argparse
import wandb
from torch import nn
from geoopt import PoincareBall, SphereProjection
from geoopt.optim import RiemannianAdam
from src.models import TruePairwiseLTR
from src.utils import fix_seed
from src.dataset import get_loaders, get_data
from src.train import train
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from src.common import str2layer
from src.criterions import TruePairwiseLTRLoss


def main(
    emb_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    data_dir: str,
    data_name: str,
    num_negatives: int,
    num_workers: int,
    temperature: float,
    dtype: str,
    device: str,
    ks: list[int],
    show_progress: bool,
    log: bool,
    conf: dict,
    run_name: str
):
    fix_seed(seed)
    
    if log:
        run = wandb.init( # noqa
            project="HypRecSys",
            config=conf,
            name=run_name
        )
    print('Prepare data')
    _, train_val_loader, val_loader, test_loader = get_loaders(*get_data(data_dir, data_name, "true_pairwise_ltr", num_negatives=(num_negatives, max(ks) - 1)), batch_size, num_workers)
    num_items = test_loader.dataset.num_items
    num_users = test_loader.dataset.num_users

    print('Constructing model')

    dtype = getattr(torch, dtype)

    model = TruePairwiseLTR(emb_dim, num_users, num_items, 1, dtype=dtype).to(device)

    optimizer = Adam(model.parameters(), lr, weight_decay=1e-3)
    criterion = TruePairwiseLTRLoss(reduction='mean').to(device)

    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=epochs)

    print('Starting train')

    history = train(
        model, 
        optimizer, 
        criterion, 
        scheduler, 
        train_val_loader, 
        test_loader, 
        epochs, 
        device, 
        ks,
        eval_with_negatives=True,
        temperature=temperature, 
        show_progress=show_progress, 
        log=log
    )

    print(f'Final results: {max(history, key=lambda x: x["hr@20"])}')

    torch.save(model.state_dict(), 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dim", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--num_negatives", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100])

    parser.add_argument("--show_progress", default=False, action="store_true")
    parser.add_argument("--log", default=False, action="store_true")
    parser.add_argument("--run_name", default=str)

    conf = vars(parser.parse_args())

    main(**conf, conf=conf)