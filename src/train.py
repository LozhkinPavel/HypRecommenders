import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from src.utils import grad_norm, hr, ndcg
from src.models import PoincareEmbedding
from src.dataset import SimilarityDataset
from geoopt.manifolds.lorentz.math import lorentz_to_poincare
from geoopt import Lorentz


def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    data_loader: DataLoader,
    dataset_type: str,
    num_epoch: int,
    device: str,
    temperature: float = 1.0,
    show_progress: bool = False
):
    model.train()
    epoch_loss = 0.0

    dataset = data_loader.dataset

    dataset_sz = len(data_loader.dataset)
    if show_progress:
        data_loader = tqdm(data_loader, desc=f"Training epoch {num_epoch}")


    if dataset_type == "sequential":
        for batch, target in data_loader:
            assert torch.all(batch <= 1)
            optimizer.zero_grad()
            batch, target = batch.to(device), target.to(device)

            preds = model(batch)
            logits = preds * temperature
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.shape[0]

    else:
        for batch in data_loader:
            assert torch.all(batch <= 1)
            batch = batch.to(device)

            optimizer.zero_grad()
            if isinstance(model[0], PoincareEmbedding):
                preds = model((batch, model[-1].manifold))
            else:
                preds = model(batch)
            logits = preds * temperature
            loss = criterion(logits, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.shape[0]

    epoch_loss /= dataset_sz

    return epoch_loss


def eval_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: str,
    temperature: float = 1.0,
    ks: list[int] = [20],
    show_progress: bool = False,
):
    model.eval()
    eval_loss = 0.0

    dataset_sz = len(data_loader.dataset)
    dataset = data_loader.dataset
    users_freqs = torch.zeros((data_loader.dataset.num_users), device=device)
    hr_sum = torch.zeros((len(ks), data_loader.dataset.num_users), device=device)
    ndcg_sum = torch.zeros((len(ks), data_loader.dataset.num_users), device=device)
    items_freqs = torch.zeros(
        (len(ks), data_loader.dataset.num_items), device=device, dtype=torch.bool
    )

    if show_progress:
        data_loader = tqdm(data_loader, desc="Eval")

    for batch, pos_item, user_id in data_loader:
        assert torch.all(batch <= 1)
        batch, pos_item, user_id = (
            batch.to(device),
            pos_item.to(device),
            user_id.to(device),
        )
        with torch.no_grad():
            if isinstance(model[0], PoincareEmbedding):
                preds = model((batch, model[-1].manifold))
            else:
                preds = model(batch)
        assert torch.all(~torch.isnan(preds))
        logits = preds * temperature
        loss = criterion(logits, pos_item)

        logits[batch == 1] = -torch.inf
        _, topk_inds = torch.topk(logits, k=max(ks), dim=1, sorted=True)

        users_freqs.index_add_(0, user_id, torch.ones((batch.shape[0]), device=device))

        for i, k in enumerate(ks):
            hr_sum[i].index_add_(0, user_id, hr(topk_inds, k, pos_item))
            ndcg_sum[i].index_add_(0, user_id, ndcg(topk_inds, k, pos_item))

            items_freqs[i, topk_inds[:, :k]] = True

        eval_loss += loss.item() * batch.shape[0]

    eval_users = torch.where(users_freqs > 0)[0]
    epoch_hr = torch.mean(
        hr_sum[:, eval_users] / users_freqs[None, eval_users], dim=1
    ).cpu()
    epoch_ndcg = torch.mean(
        ndcg_sum[:, eval_users] / users_freqs[None, eval_users], dim=1
    ).cpu()
    epoch_cov = (torch.sum(items_freqs, dim=1) / items_freqs.shape[1]).cpu()

    eval_loss /= dataset_sz

    epoch_res = {"val_loss": eval_loss}
    for i, k in enumerate(ks):
        epoch_res[f"hr@{k}"] = epoch_hr[i].item()
        epoch_res[f"ndcg@{k}"] = epoch_ndcg[i].item()
        epoch_res[f"cov@{k}"] = epoch_cov[i].item()

    return epoch_res


def train(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_type: str,
    epochs: int,
    device: str,
    ks: list[int] = [20],
    temperature: float = 1.0,
    show_progress: bool = False,
    log: bool = False,
    eval_every_epoch: bool = True,
):
    eval_metrics = eval_epoch(
        model, criterion, val_loader, device, temperature, ks, show_progress
    )
    print("Epoch -1: ", eval_metrics)
    history = []
    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            dataset_type,
            epoch,
            device,
            temperature,
            show_progress,
        )
        if (eval_every_epoch and epoch % 10 == 9) or epoch == epochs - 1:
            eval_metrics = eval_epoch(
                model, criterion, val_loader, device, temperature, ks, show_progress
            )
            eval_metrics["train_loss"] = train_loss
            history.append(eval_metrics)
            if show_progress:
                print(f"Epoch {epoch}: {eval_metrics}")
            # print([manifold.c.item() for manifold in model.manifolds])
            print(model[-1].manifold.c.item())
            if log:
                cur_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )
                wandb.log(
                    {
                        **eval_metrics,
                        "learning_rate": cur_lr,
                        "grad_norm": grad_norm(model),
                        "custom_step": epoch + 1,
                        "curvature": torch.abs(model[-1].manifold.k).item() if hasattr(model[-1], "manifold") else 1.0,
                    }
                )
        scheduler.step()
    return history


def train_embeddings_epoch(
    model: PoincareEmbedding,
    optimizer: Optimizer,
    criterion: nn.Module,
    data_loader: DataLoader,
    num_epoch: int,
    device: str,
    show_progress: bool = False
):
    model.train()
    epoch_loss = 0.0

    dataset_sz = len(data_loader.dataset)
    if show_progress:
        data_loader = tqdm(data_loader, desc=f"Training epoch {num_epoch}")
    
    for batch in data_loader:
        optimizer.zero_grad()
        logits = model((batch[0].to(device), batch[1].to(device)), mode="train")
        loss = criterion(logits, torch.zeros(logits.shape[0], dtype=torch.int64).to(device))
        loss.backward()
        optimizer.step()
        assert not torch.any(torch.isnan(model.matrix))
        epoch_loss += loss.item() * batch[0].shape[0]
    
    epoch_loss /= dataset_sz

    return epoch_loss


def train_embeddings(
    model: PoincareEmbedding,
    optimizer: Optimizer,
    criterion: nn.Module,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    num_interactions: torch.Tensor,
    epochs: int,
    burn_in_epochs: int,
    device: str,
    show_progress: bool = False,
    log: bool = False,
):
    history = []
    train_loader.dataset.burn_in = True
    max_items_sizes = [20] * (epochs)
    for epoch in range(epochs):
        if epoch == burn_in_epochs:
            train_loader.dataset.burn_in = False
        if isinstance(train_loader.dataset, SimilarityDataset):
            train_loader.dataset.set_max_item_size(max_items_sizes[epoch])
        train_loss = train_embeddings_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            epoch,
            device,
            show_progress,
        )
        history.append(train_loss)
        if show_progress:
            print(f"Epoch {epoch}: {train_loss}")
        if log:
            cur_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            wandb_metrics = {
                "embeddings_train_loss": train_loss,
                "embeddings_learning_rate": cur_lr,
                "embeddings_grad_norm": grad_norm(model),
                "custom_step": epoch + 1
            }
            if model.matrix.shape[1] == 2:
                fig, ax = plt.subplots(figsize=(15, 15))
                embs = lorentz_to_poincare(model.matrix, model.manifold.k, dim=1) if isinstance(model.manifold, Lorentz) else model.matrix
        
                ax.add_patch(plt.Circle((0, 0), 1.0, ec='black', fill=False))
                ax.scatter(embs[:, 0].cpu().detach().numpy(), embs[:, 1].cpu().detach().numpy(), c=num_interactions, norm='log')
                wandb_metrics["embeddings"] = wandb.Image(fig)

                wandb.log(
                    wandb_metrics
                )
                plt.close(fig)
            else:
                wandb.log(
                    wandb_metrics
                )
        scheduler.step()
    return history