import json
import random
import os
import datasets
from llm2vec.dataset.dataset import DataSample, TrainSample, Dataset #ValidationSample
from accelerate.logging import get_logger
import pandas as pd
logger = get_logger(__name__, log_level="INFO")

CXR_EMBEDDING_PROMPTS = {
    "cxr": "Retrieve semantically similar sentences",
    "cxr2": "Summarize the give findings",
}


class CXRDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "CXR",
        split: str = "validation",
        file_path: str = "cache/echo-data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        dataframe_path: str = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.dataframe_path = dataframe_path
        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)
    def get_cxr(self, dataset_name):
        cxr = pd.read_csv(self.dataframe_path)
        if dataset_name == "cxr":
            cxr = cxr.loc[cxr['dataset'] != "summary"].reset_index(drop=True)
        else:
            cxr = cxr.loc[cxr['dataset'] == "summary"].reset_index(drop=True)
        data = cxr[['caption1', 'caption2', 'neg']].values
        # Build list of dictionaries
        list_of_dict = []

        # Iterate through each row
        for i in range(len(data)):
            caption1 = data[i][0]
            caption2 = data[i][1]
            negative = data[i][2]
            # Generate a random number (0 or 1)
            rand_num = random.randint(0, 1)
            # Set query and positive based on random number
            if self.dataset_name == "cxr2":
                query = caption1
                positive = caption2
            elif self.dataset_name == "cxr":
                if rand_num == 0:
                    query = caption1
                    positive = caption2
                else:
                    query = caption2
                    positive = caption1
            else:
                query = caption1
                positive = caption2
            # Ensure neg is different from current row
            if (negative != '-1')&isinstance(negative, str):
                neg = negative
            else:
                while True:
                    rand_index = random.randint(0, len(data) - 1)
                    if rand_index != i:  # Ensure neg is different from current row
                        neg = data[rand_index][1] if rand_num == 0 else data[rand_index][0]
                        if (neg != '-1')&isinstance(neg, str):
                            break
                        else:
                            continue
            if isinstance(neg, int):
                assert 0
            # Create dictionary and add to list
            list_of_dict.append({
                'query': query,
                'positive': positive,
                'negative': neg,
                'random_num': rand_num
            })
        logger.info(f"Loaded {len(list_of_dict)} samples.")
        return list_of_dict
    def load_data(self, file_path: str = None):
        logger.info(f"Loading CXR data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        for dataset in CXR_EMBEDDING_PROMPTS:
            logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            if dataset in ["cxr", "cxr2"]:
                if self.dataframe_path is not None:
                    dataset_samples = self.get_cxr(dataset)
                else:
                    continue
            else:
                assert False, "No specified dataset"

            for i, sample in enumerate(dataset_samples):
                if dataset in ["cxr", "cxr2"]:
                    instruction = (
                        CXR_EMBEDDING_PROMPTS[dataset][sample["random_num"]]
                    )
                else:
                    instruction = (
                        CXR_EMBEDDING_PROMPTS[dataset]
                        if isinstance(CXR_EMBEDDING_PROMPTS[dataset], str)
                        else CXR_EMBEDDING_PROMPTS[dataset][i % 2]
                    )
                query = f"{instruction}; " + self.separator + sample["query"]
                pos = self.separator + sample["positive"]
                try:
                    neg = self.separator + sample["negative"]
                except:
                    print(sample['random_num'])
                    print(sample['query'])
                    print(sample['positive'])
                    print(sample['negative'])
                    assert 0

                data_map[dataset].append(id_)

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                id_ += 1


        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        # elif self.split == "validation":
        #     return ValidationSample(texts=[sample.query, sample.positive, sample.negative], label=1.0)
        else:
            assert False, "CXRData does not have a validation split."

def get_cxr_captions(file_path: str = 'custom.csv'):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'caption': 'text'})
    cxr_dataset = datasets.Dataset.from_pandas(df)
    return cxr_dataset