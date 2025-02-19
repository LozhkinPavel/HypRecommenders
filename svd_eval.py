from src.models import PureSVD
from src.dataset import get_loaders, get_data
from src.train import eval_epoch
from scipy.sparse.linalg import svds
from scipy.sparse import coo_array
import argparse
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np

def main(args):
    criterion = nn.CrossEntropyLoss(reduction="mean")
    train_dataset, train_val_dataset, val_dataset, test_dataset, train_interactions, train_val_interactions = get_data(args.data_dir, args.data_name, dataset_type="bag_of_items", return_df=True)
    train_loader, train_val_loader, val_loader, test_loader = get_loaders(
        train_dataset, train_val_dataset, val_dataset, test_dataset, 32, 1,
    )
    best_scores = None

    num_users = len(pd.unique(train_interactions['user_id']))
    num_items = len(pd.unique(train_interactions['item_id']))
    train_matrix = coo_array((np.ones((train_interactions.shape[0],)), (train_interactions['user_id'], train_interactions['item_id'])), shape=(num_users, num_items))
    _, s, vt = svds(train_matrix, k=8192)
    vt = vt[np.argsort(s)[::-1], :]

    print("SVD calculated")

    for rank in tqdm([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]):
        model = PureSVD(rank=rank).fit(vt[:rank, :])
        scores = eval_epoch(model, criterion, val_loader, "cpu", ks=[1, 5, 10, 20, 50, 100])
        if best_scores is None or best_scores["hr@20"] < scores["hr@20"]:
            best_scores = scores
            best_rank = rank

    num_users = len(pd.unique(train_val_interactions['user_id']))
    train_matrix = coo_array((np.ones((train_val_interactions.shape[0],)), (train_val_interactions['user_id'], train_val_interactions['item_id'])), shape=(num_users, num_items))
    *_, vt = svds(train_matrix, k=best_rank)
    model = PureSVD(rank=best_rank).fit(vt)
    scores = eval_epoch(model, criterion, test_loader, "cpu", ks=[1, 5, 10, 20, 50, 100])

    print("Final scores:", scores)
    print("Val scores:", best_scores)
    print("Best rank:", best_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_dir", default="./data/")

    args = parser.parse_args()
    main(args)
