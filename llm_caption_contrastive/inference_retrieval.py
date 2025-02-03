import logging
import os
import sys
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import PeftModel
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
BASE_MODEL_PATH = "/data/research/tmp/result/mntp_8b"
PEFT_MODEL_PATH = "/data/research/tmp/result/supervised_8b/"
DEST_CSV_PATH = "/data/research/tmp/result/"
# BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B"
# PEFT_MODEL_PATH = "/data/research/tmp/checkpoint-12600/"
TEST_CSV_PATH = "/data/csv/llm2clip/mimic_clip_test.csv"
RETRIEVAL_CSV_PATH = '/data/research/csv/test_pool.csv'
BATCH_SIZE = 64  # Adjust based on your GPU memory
MAX_LENGTH = 512  # Maximum tokenization length
NUM_WORKERS = 4  # Number of data loading workers
CANDIDATE_CAPTION_COLUMN = "report"
ANCHOR_CAPTION_COLUMN = "caption_lite"
INSTRUCTION = ""
IS_INSTRUCTION = False
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
    peft_model_name_or_path=BASE_MODEL_PATH,  # PEFT weights loaded here
    merge_peft=True,  # This will merge the PEFT weights with the base model
    pooling_mode="mean",
    max_length=MAX_LENGTH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
logger.info("Loading the PEFT adapter2...")
model.model = PeftModel.from_pretrained(
    model.model,
    PEFT_MODEL_PATH,
)

model.model = model.model.merge_and_unload()

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
candidate_dataset = TestDataset(RETRIEVAL_CSV_PATH, tokenizer, MAX_LENGTH, caption_column=CANDIDATE_CAPTION_COLUMN, instruction="", is_instruction=False)
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
# Change k from 1 to 3 to get top 3 results
top3_scores, top3_indices = torch.topk(similarities, k=3, largest=True)

# Create results dataframe
results = []
for idx in tqdm(range(len(anchor_texts)), desc="Collecting results"):
    result_dict = {
        'query_text': anchor_texts[idx],
        'retrieved_text_1': candidate_dataset.texts[top3_indices[idx][0].item()],
        'similarity_score_1': top3_scores[idx][0].item(),
        'retrieved_text_2': candidate_dataset.texts[top3_indices[idx][1].item()],
        'similarity_score_2': top3_scores[idx][1].item(),
        'retrieved_text_3': candidate_dataset.texts[top3_indices[idx][2].item()],
        'similarity_score_3': top3_scores[idx][2].item(),
        'is_correct': candidate_dataset.texts[top3_indices[idx][0].item()] == candidate_dataset.texts[idx]
    }
    results.append(result_dict)

results_df = pd.DataFrame(results)

# Calculate accuracy
accuracy = results_df['is_correct'].mean()
logger.info(f"Top-1 Accuracy: {accuracy:.2f}")
print(f"Top-1 Accuracy: {accuracy:.2f}")

# Save results to CSV
output_path = os.path.join(os.path.dirname(DEST_CSV_PATH), 'retrieval_results_mimic_new_3.csv')
results_df.to_csv(output_path, index=False)
logger.info(f"Results saved to: {output_path}")