import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
from copy import copy
from datasets import load_dataset
from tqdm import trange

class BagOfItemsDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, item_df: pd.DataFrame, shuffle: bool = True, seed: int = 42):
        super().__init__()
        self.num_items = len(pd.unique(interactions_df['item_id']))
        self.num_users = len(pd.unique(interactions_df['user_id']))

        self.item_df = item_df

        self.num_interactions = torch.zeros((self.num_items, ), dtype=torch.float64).index_add_(0, torch.tensor(interactions_df['item_id'].values), torch.ones(interactions_df.shape[0], dtype=torch.float64))

        interactions_df['item_id'] = list(zip(interactions_df['item_id'], interactions_df['timestamp']))
        def sort_by_2nd_key(ser):
            return list(map(lambda x: x[0], sorted(ser, key=lambda x: x[1])))

        self.interactions_df = interactions_df.groupby(by='user_id').agg({'item_id': sort_by_2nd_key})
        if shuffle:
            self.interactions_df = self.interactions_df.sample(frac=1.0, random_state=seed)

    def __getitem__(self, index):
        inds = torch.tensor(self.interactions_df.iloc[index]['item_id'])
        assert torch.unique(inds).shape[0] == inds.shape[0]
        return torch.zeros(self.num_items, dtype=torch.float64).index_add(0, inds, torch.ones(inds.shape[0], dtype=torch.float64))
    
    def __len__(self):
        return self.interactions_df.shape[0]

class SequentialDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, item_df: pd.DataFrame, shuffle: bool = True, seed: int = 42):
        super().__init__()
        self.num_items = len(pd.unique(interactions_df['item_id']))
        self.num_users = len(pd.unique(interactions_df['user_id']))

        self.item_df = item_df

        interactions_df['item_id'] = list(zip(interactions_df['item_id'], interactions_df['timestamp']))
        def sort_by_2nd_key(ser):
            return list(map(lambda x: x[0], sorted(ser, key=lambda x: x[1])))

        self.interactions_df = interactions_df.groupby(by='user_id').agg({'item_id': sort_by_2nd_key})
        if shuffle:
            self.interactions_df = self.interactions_df.sample(frac=1.0, random_state=seed)

        self.lens = torch.tensor(self.interactions_df['item_id'].apply(len).to_list()) - 1
        self.lens_cumsum = torch.cumsum(self.lens, dim=0)

    def __getitem__(self, index):
        user_id = torch.searchsorted(self.lens_cumsum, index, right=True).item()
        val_ind = self.lens[user_id] - (self.lens_cumsum[user_id] - index)
        inds = torch.tensor(self.interactions_df.iloc[user_id]['item_id'][:val_ind + 1])
        target = self.interactions_df.iloc[user_id]['item_id'][val_ind + 1]
        return torch.zeros(self.num_items, dtype=torch.float64).index_add(0, inds, torch.ones(inds.shape[0], dtype=torch.float64)), target
    
    def __len__(self):
        return self.lens_cumsum[-1]
    

class EvalSequentialDataset(Dataset):
    def __init__(self, interactions_df: pd.DataFrame, item_df: pd.DataFrame, separator_ts: int, shuffle: bool = True, seed: int = 42):
        super().__init__()
        self.num_items = len(pd.unique(interactions_df['item_id']))
        self.num_users = len(pd.unique(interactions_df['user_id']))

        self.num_interactions = torch.zeros((self.num_items, ), dtype=torch.float64).index_add_(0, torch.tensor(interactions_df['item_id'].values), torch.ones(interactions_df.shape[0], dtype=torch.float64))

        self.item_df = item_df

        interactions_df['item_id'] = list(zip(interactions_df['item_id'], interactions_df['timestamp']))
        def sort_by_2nd_key(ser):
            return sorted(ser, key=lambda x: x[1])

        self.interactions_df = interactions_df.groupby(by='user_id').agg({'item_id': sort_by_2nd_key})
        if shuffle:
            self.interactions_df = self.interactions_df.sample(frac=1.0, random_state=seed)
        def get_train_items(ls):
            return list(map(lambda x: x[0], sorted(filter(lambda x: x[1] < separator_ts, ls), key=lambda x: x[1])))
        def get_val_items(ls):
            return list(map(lambda x: x[0], sorted(filter(lambda x: x[1] >= separator_ts, ls), key=lambda x: x[1])))
        self.interactions_df['train_item_id'] = self.interactions_df['item_id'].apply(get_train_items)
        self.interactions_df['val_item_id'] = self.interactions_df['item_id'].apply(get_val_items)
        self.interactions_df = self.interactions_df[self.interactions_df['val_item_id'].apply(len) > 0]

        self.lens = torch.tensor(self.interactions_df['val_item_id'].apply(len).to_list())
        self.lens_cumsum = torch.cumsum(self.lens, dim=0)

    def __getitem__(self, index):
        user_id = torch.searchsorted(self.lens_cumsum, index, right=True).item()
        val_ind = self.lens[user_id] - (self.lens_cumsum[user_id] - index)
        train_items = torch.tensor(self.interactions_df.iloc[user_id]['train_item_id'], dtype=torch.int64)
        val_items = torch.tensor(self.interactions_df.iloc[user_id]['val_item_id'][:val_ind], dtype=torch.int64)
        inds = torch.cat((train_items, val_items))
        pos_item = self.interactions_df.iloc[user_id]['val_item_id'][val_ind]
        return torch.zeros(self.num_items, dtype=torch.float64).index_add(0, inds, torch.ones(inds.shape[0], dtype=torch.float64)), pos_item, user_id
    
    def __len__(self):
        return self.lens_cumsum[-1]
    

class TreeDataset(Dataset):
    def __init__(self, items_sims, num_interactions, num_negatives):
        super().__init__()
        self.num_interactions = num_interactions
        self.num_negatives = num_negatives
        # sums = torch.sum(interactions, dim=0)
        # items_sims = (interactions.T @ interactions) / torch.pow(sums[:, None] * sums[None, :], 0.2)
        items_dist = 1 / (items_sims + 1)

        #MST
        self.num_vertex = items_sims.shape[0]
        self.mst_edges = [[] for _ in range(self.num_vertex)]
        vis = torch.zeros(self.num_vertex, dtype=torch.int64)
        first_v = torch.argmax(num_interactions)
        min_edge = torch.full((self.num_vertex,), first_v, dtype=torch.int64)
        vis[first_v] = 1
        self.edges_list = []
        arange = torch.arange(self.num_vertex)
        for _ in range(1, self.num_vertex):
            dists = items_dist[min_edge[arange], arange]
            dists[vis == 1] = torch.inf
            cur_v = torch.argmin(dists).item()
            vis[cur_v] = 1
            self.mst_edges[min_edge[cur_v].item()].append(cur_v)
            min_edge[items_dist[min_edge[arange], arange] > items_dist[cur_v, arange]] = cur_v


        #Transitive closure
        def dfs(x):
            subtree = []
            for v in self.mst_edges[x]:
                subtree += dfs(v)
            self.edges_list += [(v, x) for v in subtree]
            return subtree + [x]
        
        dfs(first_v.item())
        
        self.neighbours = torch.zeros(self.num_vertex, self.num_vertex, dtype=torch.bool)
        for u, v in self.edges_list:
            self.neighbours[u, v] = 1
        self.neighbours[torch.arange(self.num_vertex, dtype=torch.int32), torch.arange(self.num_vertex, dtype=torch.int32)] = 1

        self.burn_in = True
        self.counts = torch.zeros(self.num_vertex)
        for edge in self.edges_list:
            self.counts[edge[1]] += 1
        

    def __len__(self):
        return len(self.edges_list)
    
    def __getitem__(self, index):
        edge = self.edges_list[index]
        pos_ind = edge[1]
        if self.burn_in:
            counts = self.counts.clone()
            counts[self.neighbours[edge[0]]] = 0
        else:
            counts = 1 - (self.neighbours[edge[0]]).to(torch.float32)
        neg_dist = Categorical(probs=counts / torch.sum(counts))
        neg_inds = neg_dist.sample((self.num_negatives,))
        return torch.cat((torch.tensor([pos_ind]), neg_inds)), edge[0]
    

# class SimilarityDataset(Dataset):
#     def __init__(self, items_sims, num_interactions, num_negatives):
#         super().__init__()
#         self.num_interactions = num_interactions
#         num_items = items_sims.shape[0]
#         items_sims[torch.arange(num_items), torch.arange(num_items)] = 0
#         self.num_negatives = num_negatives
#         sorted_sims, self.sorted_inds = torch.sort(items_sims, dim=1)

#         self.less_inds = torch.searchsorted(sorted_sims, sorted_sims, right=False)
        
#         self.max_item_size = num_items

#         self.users_thresholds = torch.cumsum(torch.minimum(torch.count_nonzero(self.less_inds, dim=1), torch.tensor([self.max_item_size])), dim=0)
        
#         self.burn_in = True
#         self.counts = torch.sum(items_sims, dim=0)

#     def __len__(self):
#         return self.users_thresholds[-1]
    
#     def set_max_item_size(self, max_items_size):
#         self.max_item_size = max_items_size

#         self.users_thresholds = torch.cumsum(torch.minimum(torch.count_nonzero(self.less_inds, dim=1), torch.tensor([max_items_size])), dim=0)
    
#     def __getitem__(self, index):
#         item_ind = torch.searchsorted(self.users_thresholds, index, right=True)
#         ind = self.users_thresholds[item_ind] - index
#         positive_ind = torch.tensor([self.sorted_inds[item_ind, -ind]])
#         less_ind = self.less_inds[item_ind, -ind]
#         if self.burn_in:
#             counts = self.counts.clone()
#         else:
#             counts = torch.ones(self.less_inds.shape[0])
#         counts[self.sorted_inds[item_ind, less_ind:]] = 0
#         counts[item_ind] = 0
#         assert torch.sum(counts) > 0, less_ind
#         neg_inds = Categorical(probs=counts / torch.sum(counts)).sample((self.num_negatives,))
#         return torch.cat((positive_ind, neg_inds)), item_ind
    

class SimilarityDataset(Dataset):
    def __init__(self, items_inds, values, weights, num_negatives, num_interactions):
        super().__init__()
        self.num_interactions = num_interactions
        self.num_items = len(items_inds)
        self.num_negatives = num_negatives
        sorted_vals, sorted_inds = zip(*[torch.sort(t) for t in values])
        sorted_vals = list(sorted_vals)
        sorted_inds = list(sorted_inds)
        
        self.items_inds = [t1[t2] for t1, t2 in zip(items_inds, sorted_inds)]
        self.less_inds = [torch.searchsorted(t, t, right=False) for t in sorted_vals]
        
        self.max_item_size = self.num_items

        self.users_thresholds = torch.cumsum(torch.minimum(torch.tensor(list(map(lambda x: x.shape[0], self.items_inds))), torch.tensor([self.max_item_size])), dim=0)
        
        self.burnin = True
        self.weights = weights

    def __len__(self):
        return self.users_thresholds[-1]
    
    def set_max_item_size(self, max_items_size):
        self.max_item_size = max_items_size

        self.users_thresholds = torch.cumsum(torch.minimum(torch.tensor(list(map(lambda x: x.shape[0], self.items_inds))), torch.tensor([self.max_item_size])), dim=0)
    
    def __getitem__(self, index):
        item_ind = torch.searchsorted(self.users_thresholds, index, right=True)
        ind = self.users_thresholds[item_ind] - index
        positive_ind = torch.tensor([self.items_inds[item_ind][-ind]])
        less_ind = self.less_inds[item_ind][-ind]
        if self.burnin:
            counts = self.weights.clone()
        else:
            counts = torch.ones(len(self.less_inds))
        counts[self.items_inds[item_ind][less_ind:]] = 0
        counts[item_ind] = 0
        # assert torch.sum(counts) > 0, less_ind
        neg_inds = Categorical(probs=counts / torch.sum(counts)).sample((self.num_negatives,))
        return torch.cat((positive_ind, neg_inds)), item_ind


str2dataset = {
    "bag_of_items": BagOfItemsDataset,
    "sequential": SequentialDataset,
    "tree": TreeDataset,
    "similarity": SimilarityDataset
}

def get_data(data_dir: str, data_name: str, dataset_type: str, embedding_dataset_type: str = None, embedding_num_negatives: int = None, seed: int = 42, return_df=False):
    if data_name == 'ml1m':
        interactions_data = pd.read_csv(os.path.join(data_dir, data_name, 'ratings.dat'), names=['userId', 'itemId', 'rating', 'timestamp'], delimiter='::', engine='python')
        interactions_data = interactions_data.rename(columns={'userId': 'user_id', 'itemId': 'item_id'})
        interactions_data = interactions_data.drop('rating', axis=1)

        item_data = pd.read_csv(os.path.join(data_dir, data_name, 'movies.dat'), names=['MovieID', 'Title', 'Genres'], delimiter='::', encoding="ISO-8859-1", engine='python')
        item_data = item_data.rename(columns={'MovieID': 'item_id'})

        timestamps = np.array(interactions_data['timestamp'])
        val_separator = np.quantile(timestamps, 0.9, method='lower')
        test_separator = np.quantile(timestamps, 0.95, method='lower')

        unique_items = pd.unique(interactions_data[interactions_data['timestamp'] < val_separator]['item_id'])
        # old_items = item_data[pd.to_numeric(item_data['Title'].str.slice(-5, -1)) < 1990]['item_id']
        # unique_items = pd.Series(list(set(unique_items) & set(old_items)))
        item2idx = {item_id: i for i, item_id in enumerate(unique_items)}

        item_data = item_data[item_data['item_id'].isin(unique_items)]
        interactions_data = interactions_data[interactions_data['item_id'].isin(unique_items)]

        interactions_data.loc[:, 'item_id'] = interactions_data['item_id'].apply(lambda x: item2idx[x])
        item_data.loc[:, 'item_id'] = item_data['item_id'].apply(lambda x: item2idx[x])

        unique_users = pd.unique(interactions_data['user_id'])
        user2idx = {user_id: i for i, user_id in enumerate(unique_users)}

        interactions_data.loc[:, 'user_id'] = interactions_data['user_id'].apply(lambda x: user2idx[x])

        train_dataset = str2dataset[dataset_type](copy(interactions_data[interactions_data['timestamp'] < val_separator]), item_data, seed=seed)
        train_val_dataset = str2dataset[dataset_type](copy(interactions_data[interactions_data['timestamp'] < test_separator]), item_data, seed=seed)

        val_dataset = EvalSequentialDataset(copy(interactions_data[interactions_data['timestamp'] < test_separator]), item_data, val_separator, seed=seed)
        test_dataset = EvalSequentialDataset(copy(interactions_data), item_data, test_separator, seed=seed)

        train_interactions = interactions_data[interactions_data['timestamp'] < val_separator]
        val_interactions = interactions_data[(interactions_data['timestamp'] < test_separator) & (interactions_data['timestamp'] >= val_separator)]

    else: 
        if data_name == "office_products":
            dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name="5core_timestamp_Office_Products", trust_remote_code=True)
            # item_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name="raw_meta_Office_Products", trust_remote_code=True)
        elif data_name == "video_games":
            dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name="5core_timestamp_Video_Games", trust_remote_code=True)
            # item_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", name="raw_meta_Video_Games", trust_remote_code=True)

        print(1)
        train_interactions = pd.DataFrame(dataset["train"]).drop('rating', axis=1)
        val_interactions = pd.DataFrame(dataset["valid"]).drop('rating', axis=1)
        test_interactions = pd.DataFrame(dataset["test"]).drop('rating', axis=1)

        print(2)

        train_interactions = train_interactions.rename(columns={'parent_asin': 'item_id'})
        val_interactions = val_interactions.rename(columns={'parent_asin': 'item_id'})
        test_interactions = test_interactions.rename(columns={'parent_asin': 'item_id'})

        unique_items = pd.unique(train_interactions['item_id'])
        item2idx = {item_id: i for i, item_id in enumerate(unique_items)}

        val_interactions = val_interactions[val_interactions['item_id'].isin(unique_items)]
        test_interactions = test_interactions[test_interactions['item_id'].isin(unique_items)]

        print(3)

        train_interactions['item_id'] = train_interactions['item_id'].apply(lambda x: item2idx[x])
        val_interactions['item_id'] = val_interactions['item_id'].apply(lambda x: item2idx[x])
        test_interactions['item_id'] = test_interactions['item_id'].apply(lambda x: item2idx[x])

        unique_users = pd.unique(pd.concat([train_interactions['user_id'], val_interactions['user_id'], test_interactions['user_id']]))
        user2idx = {user_id: i for i, user_id in enumerate(unique_users)}

        train_interactions['user_id'] = train_interactions['user_id'].apply(lambda x: user2idx[x])
        val_interactions['user_id'] = val_interactions['user_id'].apply(lambda x: user2idx[x])
        test_interactions['user_id'] = test_interactions['user_id'].apply(lambda x: user2idx[x])

        # item_data = pd.DataFrame(item_dataset["full"]).rename(columns={'parent_asin': 'item_id'})[['title', 'item_id', 'categories']]
        # item_data = item_data[item_data['item_id'].isin(unique_items)]
        # item_data['item_id'] = item_data['item_id'].apply(lambda x: item2idx[x])

        train_dataset = str2dataset[dataset_type](copy(train_interactions), None, seed=seed)
        train_val_dataset = str2dataset[dataset_type](pd.concat([train_interactions, val_interactions]), None, seed=seed)

        val_separator = val_interactions['timestamp'].min()
        test_separator = test_interactions['timestamp'].min()

        val_dataset = EvalSequentialDataset(pd.concat([train_interactions, val_interactions]), None, val_separator, seed=seed)
        test_dataset = EvalSequentialDataset(pd.concat([train_interactions, val_interactions, test_interactions]), None, test_separator, seed=seed)
        print(4)

    if embedding_dataset_type is None:
        if return_df:
            return train_dataset, train_val_dataset, val_dataset, test_dataset, train_interactions, pd.concat([train_interactions, val_interactions])
        return train_dataset, train_val_dataset, val_dataset, test_dataset
    
    if data_name == "ml1m":
        interactions = interactions_data[interactions_data['timestamp'] < val_separator]
    else:
        interactions = train_interactions
    
    if embedding_dataset_type == "tree":
        interaction_matrix = torch.zeros(len(user2idx), len(item2idx))
        interaction_matrix[interactions['user_id'].values, interactions['item_id'].values] = 1
        num_interactions = torch.sum(interaction_matrix, dim=0)
        items_sims = (interaction_matrix.T @ interaction_matrix) / torch.pow(num_interactions[:, None] * num_interactions[None, :], 0.2)
        embeddings_dataset = str2dataset[embedding_dataset_type](items_sims, num_interactions, embedding_num_negatives)
    else:
        inds = torch.zeros((2, interactions.shape[0]), dtype=torch.int64)
        vals = torch.ones(interactions.shape[0], dtype=torch.float32)

        inds[0] = torch.from_numpy(interactions['user_id'].values)
        inds[1] = torch.from_numpy(interactions['item_id'].values)

        interaction_matrix = torch.sparse_coo_tensor(inds, vals, size=(len(user2idx), len(item2idx)))

        categories = [
            "Action",
            "Adventure",
            "Animation",
            "Children\'s",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        category_matrix = torch.zeros((interaction_matrix.shape[1], len(categories)))

        for j, row in enumerate(item_data.sort_values(by=['item_id']).iterrows()):
            for i, category in enumerate(categories):
                category_matrix[j, i] = int(category in row[1]["Genres"])

        category_matrix = category_matrix.to_sparse()
        sims = (interaction_matrix.T @ interaction_matrix) * (category_matrix @ category_matrix.T)
        # sims = (interaction_matrix.T @ interaction_matrix)
        num_interactions = torch.sum(interaction_matrix, dim=0).to_dense()
        sims_indices = sims.indices()
        sims_values = sims.values()

        inds = [[] for _ in range(len(item2idx))]
        values = [[] for _ in range(len(item2idx))]
        weights = torch.zeros((len(item2idx),), dtype=torch.float32).index_add(0, sims_indices[1], sims_values)

        for i in trange(sims_indices.shape[1]):
            if sims_indices[0, i] == sims_indices[1, i]:
                continue
            inds[sims_indices[0, i]].append(sims_indices[1, i].item())
            values[sims_indices[0, i]].append((sims_values[i] / torch.pow(interaction_matrix.shape[1] + num_interactions[sims_indices[1, i]], 1.0)).item())

        nested_inds = [torch.tensor(ls, dtype=torch.int32) for ls in inds]
        nested_values = [torch.tensor(ls, dtype=torch.float32) for ls in values]

        embeddings_dataset = str2dataset[embedding_dataset_type](nested_inds, nested_values, weights, embedding_num_negatives, num_interactions)

    print(5)

    return train_dataset, train_val_dataset, val_dataset, test_dataset, embeddings_dataset


def get_loaders(train_dataset, train_val_dataset, val_dataset, test_dataset, batch_size: int = 64, num_workers: int = 1,):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, train_val_loader, val_loader, test_loader
