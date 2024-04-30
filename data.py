from datasets import load_dataset
from torch.utils.data import random_split
import torch

def data_pre(dataset_path):
    dataset = load_dataset('json', data_files={'train': dataset_path})

    dataset = dataset["train"]

    return dataset

if __name__ == "__main__":
    data_pre("/data/fengduanyu/modelcheat/test_data/test.jsonl")