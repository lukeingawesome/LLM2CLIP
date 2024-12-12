from llm2vec.dataset.dataset import DataSample, TrainSample, Dataset
import numpy as np
from accelerate.logging import get_logger
import pandas as pd
import datasets
logger = get_logger(__name__, log_level="INFO")


class CXRDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "cxr",
        split: str = "train",
        file_path: str = 'cxr.csv',
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str):
        logger.info(f"Loading CXR dataset from {file_path}...")
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            caption = row['caption'].strip()
            self.data.append(
                DataSample(
                    id_=idx,
                    query=caption,
                    positive=caption,
                )
            )
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(texts=[sample.query, sample.positive], label=1.0)
        elif self.split == "validation":
            assert False, "CXRDataset does not have a validation split."


def get_cxr_captions(file_path: str = 'custom.csv'):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'caption': 'text'})
    cxr_dataset = datasets.Dataset.from_pandas(df)
    return cxr_dataset
