import logging
import os
import sys
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from transformers import AutoTokenizer

from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

# -----------------------------------------------------------
# 1. Setup Logging
# -----------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_PATH = "/data/research/tmp/checkpoint-llama8b-base/"
# BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B"
# PEFT_MODEL_PATH = "/data/research/tmp/checkpoint-12600/"
TEST_CSV_PATH = "/data/csv/llm2clip/test/openi.csv"
TEST_CSV_PATH = "/data/research/csv/openi_summarize.csv" ## Summarize
BATCH_SIZE = 32  # Adjust based on your GPU memory
MAX_LENGTH = 512  # Maximum tokenization length
NUM_WORKERS = 4  # Number of data loading workers
CANDIDATE_CAPTION_COLUMN = "impression" #"report_split"
ANCHOR_CAPTION_COLUMN = "findings" #"report"
INSTRUCTION = "Retrieve semantically similar sentences"
IS_INSTRUCTION = True

# -----------------------------------------------------------
# 3. Initialize Accelerator
# -----------------------------------------------------------
accelerator = Accelerator()
device = accelerator.device

# -----------------------------------------------------------
# 4. Define Dataset Class
# -----------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int, caption_column: str, instruction: str, is_instruction: bool):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df[caption_column].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.is_instruction = is_instruction
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.is_instruction:
            text = f"{self.instruction.strip()} !@#$%^&*(){text}" if self.instruction else f"!@#$%^&*(){text}"
        else:
            pass

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create embedding mask (1 for tokens to embed, 0 for tokens to ignore)
        embed_mask = torch.zeros_like(encoding["attention_mask"])
        # Mask padding tokens
        # embed_mask[encoding['input_ids'] == self.tokenizer.pad_token_id] = 0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'embed_mask': embed_mask.squeeze(0)
        }

# -----------------------------------------------------------
# 5. Load the Tokenizer and Model
# -----------------------------------------------------------
logger.info("Loading the tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, 
    use_fast=True
)

# -----------------------------------------------------------
# 5.1 Configure the Tokenizer Padding
# -----------------------------------------------------------
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Add a new pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Added a new pad_token: '[PAD]'.")

logger.info("Loading the LLM2Vec model with PEFT adapter...")
model = LLM2Vec.from_pretrained(
    base_model_name_or_path=BASE_MODEL_PATH,
    enable_bidirectional=True,
    peft_model_name_or_path=PEFT_MODEL_PATH,  # PEFT weights loaded here
    merge_peft=True,  # This will merge the PEFT weights with the base model
    pooling_mode="mean",
    max_length=MAX_LENGTH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

model.to(device)
model.eval()
logger.info("Model and tokenizer loaded successfully.")

# -----------------------------------------------------------
# 6. Prepare Test Data
# -----------------------------------------------------------
logger.info(f"Loading test data from {TEST_CSV_PATH}...")

# -----------------------------------------------------------
# 7. Generate Candidate Embeddings
# -----------------------------------------------------------
logger.info("Generating candidate embeddings...")

# Create DataLoader for candidate texts
candidate_dataset = TestDataset(TEST_CSV_PATH, tokenizer, MAX_LENGTH, caption_column=CANDIDATE_CAPTION_COLUMN, instruction="", is_instruction=False)
candidate_dataloader = DataLoader(
    candidate_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

candidate_dataloader = accelerator.prepare(candidate_dataloader)

candidate_embeddings = []
with torch.no_grad():
    for batch in tqdm(candidate_dataloader, desc="Encoding candidate texts"):
        encoding = {k: v.to(device) for k, v in batch.items()}
        emb = model.forward(encoding)
        emb = F.normalize(emb, p=2, dim=1)
        candidate_embeddings.append(emb)

candidate_embeddings = torch.cat(candidate_embeddings, dim=0)
logger.info("Candidate embeddings computed.")

# -----------------------------------------------------------
# 8. Inference and Evaluation
# -----------------------------------------------------------
logger.info("Starting retrieval evaluation...")
top1_correct = 0
top5_correct = 0
top10_correct = 0


# Prepare DataLoader for anchor texts
anchor_dataset = TestDataset(TEST_CSV_PATH, tokenizer, MAX_LENGTH, caption_column=ANCHOR_CAPTION_COLUMN, instruction=INSTRUCTION, is_instruction=IS_INSTRUCTION)
anchor_dataloader = DataLoader(
    anchor_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

anchor_dataloader = accelerator.prepare(anchor_dataloader)

anchor_embeddings = []
with torch.no_grad():
    for batch in tqdm(anchor_dataloader, desc="Encoding anchor texts"):
        encoding = {k: v.to(device) for k, v in batch.items()}
        emb = model.forward(encoding)
        emb = F.normalize(emb, p=2, dim=1)
        anchor_embeddings.append(emb)

anchor_embeddings = torch.cat(anchor_embeddings, dim=0)
anchor_texts = anchor_dataset.texts
# Compute similarities
similarities = torch.matmul(anchor_embeddings, candidate_embeddings.transpose(0, 1))
topk = 10
topk_scores, topk_indices = torch.topk(similarities, topk, largest=True)

# Evaluate
for idx in tqdm(range(len(anchor_texts)), desc="Evaluating"):
    topk_texts = [candidate_dataset.texts[i] for i in topk_indices[idx].tolist()]
    # cor_text = test_dataset.df['report_split'].iloc[idx]
    cor_text = candidate_dataset.texts[idx]
    
    if cor_text == topk_texts[0]:
        top1_correct += 1
    if cor_text in topk_texts[:5]:
        top5_correct += 1
    if cor_text in topk_texts:
        top10_correct += 1

# -----------------------------------------------------------
# 9. Compute and Display Accuracy
# -----------------------------------------------------------
top1_accuracy = top1_correct / len(anchor_texts)
top5_accuracy = top5_correct / len(anchor_texts)
top10_accuracy = top10_correct / len(anchor_texts)

logger.info(f"Top-1 Accuracy : {top1_accuracy:.2f}")
logger.info(f"Top-5 Accuracy : {top5_accuracy:.2f}")
logger.info(f"Top-10 Accuracy: {top10_accuracy:.2f}")

print(f"Top-1 Accuracy : {top1_accuracy:.2f}")
print(f"Top-5 Accuracy : {top5_accuracy:.2f}")
print(f"Top-10 Accuracy: {top10_accuracy:.2f}")