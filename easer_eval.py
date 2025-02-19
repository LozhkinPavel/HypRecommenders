from src.models import Easer
from src.dataset import get_loaders, get_data
from src.train import eval_epoch
from scipy.sparse.linalg import svds
from scipy.sparse import coo_array
import argparse
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import wandb

def main(args):
    wandb.init( # noqa
        project="HypRecSys"
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")
    train_dataset, train_val_dataset, val_dataset, test_dataset, train_interactions, train_val_interactions = get_data(args.data_dir, args.data_name, dataset_type="bag_of_items", return_df=True)
    train_loader, train_val_loader, val_loader, test_loader = get_loaders(
        train_dataset, train_val_dataset, val_dataset, test_dataset, 32, 1,
    )
    best_scores = None

    num_users = len(pd.unique(train_interactions['user_id']))
    num_items = len(pd.unique(train_interactions['item_id']))
    train_matrix = coo_array((np.ones((train_interactions.shape[0],)), (train_interactions['user_id'], train_interactions['item_id'])), shape=(num_users, num_items))
    gram_matrix = (train_matrix.T @ train_matrix).todense()

    print(gram_matrix)

    for lam in tqdm([1e7]):
        model = Easer().fit(gram_matrix, lam)
        scores = eval_epoch(model, criterion, val_loader, "cpu", ks=[1, 5, 10, 20, 50, 100])
        if best_scores is None or best_scores["hr@20"] < scores["hr@20"]:
            best_scores = scores
            best_lam = lam

    num_users = len(pd.unique(train_val_interactions['user_id']))
    train_matrix = coo_array((np.ones((train_val_interactions.shape[0],)), (train_val_interactions['user_id'], train_val_interactions['item_id'])), shape=(num_users, num_items))
    gram_matrix = (train_matrix.T @ train_matrix).todense()
    model = Easer().fit(gram_matrix, best_lam)
    scores = eval_epoch(model, criterion, test_loader, "cpu", ks=[1, 5, 10, 20, 50, 100])

    print("Final scores:", scores)
    print("Val scores:", best_scores)
    print("Best lambda:", best_lam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_dir", default="./data/")

    args = parser.parse_args()
    main(args)
