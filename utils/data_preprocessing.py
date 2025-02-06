import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def collate_fn(batch):
    query_img_lists, passage_imgs = zip(*batch)

    max_length = max(len(query_list) for query_list in query_img_lists)
    padded_queries = []
    query_lengths = []
    for query_list in query_img_lists:
        query_lengths.append(len(query_list))
        padded_queries.append(
            torch.stack(query_list + [torch.zeros_like(query_list[0])] * (max_length - len(query_list)))
        )

    padded_queries = torch.stack(padded_queries)

    passage_imgs = torch.stack(passage_imgs)

    return padded_queries, passage_imgs, torch.tensor(query_lengths)



class DPRDataset(Dataset):
    def __init__(
            self, 
            root_dir: str = 'dataset', 
            file_name: str = 'dataset.json', 
            transform=None
        ):
        self.root_dir = root_dir
        self.transform = transform

        self.query_passage_pairs = dict()

        with open(f'{root_dir}/{file_name}', 'r') as f:
            self.dataset = json.load(f)

        for product_id, k_list in self.dataset.items():
            if product_id not in self.query_passage_pairs:
                self.query_passage_pairs[product_id] = []
            for k in k_list:
                key = f"{product_id}_{k}_crop"
                self.query_passage_pairs[product_id].append(key)

        self.product_ids = list(self.query_passage_pairs.keys())

    def __len__(self):
        return len(self.product_ids)

    def __getitem__(self, idx):
        product_id = self.product_ids[idx]

        query_img_list = []
        for img_name in self.query_passage_pairs[product_id]:
            query_img_path = f"{self.root_dir}/wearing/{img_name}.jpg"
            query_img = Image.open(query_img_path).convert('RGB')

            if self.transform:
                query_img = self.transform(query_img)
            else:
                query_img = to_tensor(query_img)

            query_img_list.append(query_img)

        passage_img_path = f"{self.root_dir}/product/{product_id}.jpg"
        passage_img = Image.open(passage_img_path).convert('RGB')
        if self.transform:
            passage_img = self.transform(passage_img)
        else:
            passage_img = to_tensor(passage_img)

        return query_img_list, passage_img
