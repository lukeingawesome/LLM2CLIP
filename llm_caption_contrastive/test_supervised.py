import logging
import os
import sys
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B"
PEFT_MODEL_PATH = "/data2/hanbin/llm2clip/output/mntp/Llama-3.2-3B/checkpoint-9000"
TEST_CSV_PATH = "/data/csv/llm2clip/test/openi.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # Adjust based on your GPU memory
MAX_LENGTH = 512  # Maximum tokenization length

# -----------------------------------------------------------
# 3. Load the Model and Tokenizer
# -----------------------------------------------------------
logger.info("Loading the tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# -----------------------------------------------------------
# 3.1 Configure the Tokenizer Padding
# -----------------------------------------------------------
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Add a new pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Added a new pad_token: '[PAD]'.")

# -----------------------------------------------------------
# 3.2 Load Model with PEFT Adapter
# -----------------------------------------------------------
logger.info("Loading the LLM2Vec model with PEFT adapter...")
model = LLM2Vec.from_pretrained(
    base_model_name_or_path=BASE_MODEL_PATH,
    enable_bidirectional=True,
    peft_model_name_or_path=PEFT_MODEL_PATH,  # PEFT weights loaded here
    merge_peft=True,  # This will merge the PEFT weights with the base model
    pooling_mode="mean",
    max_length=MAX_LENGTH,
    torch_dtype=torch.bfloat16,
    extra_model_name_or_path=[PEFT_MODEL_PATH],
)

model.to(DEVICE)
model.eval()
logger.info("Model and tokenizer loaded successfully.")


# -----------------------------------------------------------
# 4. Load Test Data
# -----------------------------------------------------------
logger.info(f"Loading test data from {TEST_CSV_PATH}...")
df = pd.read_csv(TEST_CSV_PATH)
logger.info(f"Loaded {len(df)} test samples.")

# -----------------------------------------------------------
# 5. Prepare Candidate Embeddings
# -----------------------------------------------------------
candidate_texts = df['fancy_summary'].tolist()
logger.info("Precomputing embeddings for candidate texts...")

candidate_embeddings = model.encode(candidate_texts)
logger.info("Candidate embeddings computed.")

# -----------------------------------------------------------
# 6. Retrieval Evaluation
# -----------------------------------------------------------
logger.info("Starting retrieval evaluation...")
top1_correct = 0
top5_correct = 0
top10_correct = 0

num_rows = len(df)

anchor_embeddings = model.encode(df['report'].tolist())
correct_text = df['report_split'].tolist()
anchor_text = model.encode(df['report'].tolist())
similarities = torch.mm(anchor_text, candidate_embeddings.transpose(0,1))
topk = 10
topk_scores, topk_indices = torch.topk(similarities, topk, largest=True)
topk_texts = [candidate_texts[idx] for idx in topk_indices.tolist()]

    # Check if the correct_text is within top 1, 5, 10
if correct_text == topk_texts[0]:
    top1_correct += 1
if correct_text in topk_texts[:5]:
    top5_correct += 1
if correct_text in topk_texts:
    top10_correct += 1

# -----------------------------------------------------------
# 7. Compute and Display Accuracy
# -----------------------------------------------------------
top1_accuracy = top1_correct / num_rows
top5_accuracy = top5_correct / num_rows
top10_accuracy = top10_correct / num_rows

logger.info(f"Top-1 Accuracy : {top1_accuracy:.2f}")
logger.info(f"Top-5 Accuracy : {top5_accuracy:.2f}")
logger.info(f"Top-10 Accuracy: {top10_accuracy:.2f}")

print(f"Top-1 Accuracy : {top1_accuracy:.2f}")
print(f"Top-5 Accuracy : {top5_accuracy:.2f}")
print(f"Top-10 Accuracy: {top10_accuracy:.2f}") 