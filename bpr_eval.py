from src.models import PureSVD, BPRWrapper
from src.dataset import get_loaders, get_data
from src.train import eval_epoch, eval_epoch_with_negatives
from scipy.sparse.linalg import svds
from scipy.sparse import coo_array, csr_matrix
import argparse
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np

def main(args):
    criterion = nn.CrossEntropyLoss(reduction="mean")
    train_dataset, train_val_dataset, val_dataset, test_dataset, train_interactions, train_val_interactions = get_data(args.data_dir, args.data_name, num_negatives=(19, 19), dataset_type="bag_of_items", return_df=True)
    train_loader, train_val_loader, val_loader, test_loader = get_loaders(
        train_dataset, train_val_dataset, val_dataset, test_dataset, 32, 1,
    )
    best_scores = None
    best_params = None

    num_users = train_interactions['user_id'].max() + 1
    num_items = len(pd.unique(train_interactions['item_id']))
    train_matrix = coo_array((np.ones((train_interactions.shape[0],)), (train_interactions['user_id'], train_interactions['item_id'])), shape=(num_users, num_items)).tocsr()

    for emb_dim in tqdm([8, 16, 32, 64, 128]):
        for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
            model = BPRWrapper(
                iterations=100,
                factors=emb_dim,
                regularization=lam,
                learning_rate=1e-3,
            )
            model.fit(train_matrix)
            scores = eval_epoch_with_negatives(model, criterion, val_loader, "cpu", ks=[1, 5, 10, 20])
            print(scores)
            if best_scores is None or best_scores["hr@10"] < scores["hr@10"]:
                best_scores = scores
                best_params = {
                    'emb_dim': emb_dim,
                    'lambda': lam
                }

    num_users = train_val_interactions['user_id'].max() + 1
    train_matrix = coo_array((np.ones((train_val_interactions.shape[0],)), (train_val_interactions['user_id'], train_val_interactions['item_id'])), shape=(num_users, num_items)).tocsr()
    model = BPRWrapper(
        iterations=100,
        factors=best_params['emb_dim'],
        regularization=best_params['lambda'],
        learning_rate=1e-3,
    )
    model.fit(train_matrix)
    scores = eval_epoch_with_negatives(model, criterion, test_loader, "cpu", ks=[1, 5, 10, 20])

    print("Final scores:", scores)
    print("Val scores:", best_scores)
    print("Best params:", best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_dir", default="./data/")

    args = parser.parse_args()
    main(args)
