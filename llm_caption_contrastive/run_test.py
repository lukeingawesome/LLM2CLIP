import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, SequentialSampler

from accelerate import Accelerator
from accelerate.logging import get_logger

import transformers
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
from dataset.utils import load_dataset
from llm2vec.loss.utils import load_loss
from llm2vec.experiment_utils import generate_experiment_id

from tqdm import tqdm

# Suppress unnecessary warnings
transformers.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use for testing.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The PEFT model checkpoint to add on top of the base model."},
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": "The pooling mode to use in the model.",
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        },
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the test data input.
    """
    test_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the test dataset to use."},
    )
    test_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The input test data file or folder."},
    )
    dataframe_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataframe file for the test dataset."},
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker testing, truncate the number of test examples to this value if set."
        },
    )

@dataclass
class CustomArguments:
    """
    Custom arguments for the testing script.
    """
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for testing."}
    )
    output_dir: Optional[str] = field(
        default="./test_results",
        metadata={"help": "Directory to save test results."}
    )

def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path in ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-8B"]:
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text

def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    logger.info("Initialized Accelerator.")

    # Argument parsing
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, custom_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(42)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load model with PEFT
    logger.info("Loading the LLM2Vec model with PEFT adapter for testing...")
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=False,  # Typically not needed for testing
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch.float16 if model_args.torch_dtype == "float16" else getattr(torch, model_args.torch_dtype),
    )

    model.to(accelerator.device)
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = load_dataset(
        dataset_name=data_args.test_dataset_name,
        split="test",
        file_path=data_args.test_file_path,
        dataframe_path=data_args.dataframe_path,
    )

    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))
        logger.info(f"Truncated test dataset to {data_args.max_test_samples} samples.")

    # Prepare data collator
    def collate_fn(batch):
        texts = [prepare_for_tokenization(model.model, example['text'], pooling_mode=model_args.pooling_mode) for example in batch]
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=model_args.max_seq_length,
            return_tensors="pt"
        )
        labels = torch.tensor([example['label'] for example in batch])
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=custom_args.batch_size,
        collate_fn=collate_fn,
    )

    # Prepare dataloader with accelerator
    test_dataloader = accelerator.prepare(test_dataloader)
    logger.info("Test DataLoader prepared.")

    # Evaluation loop
    logger.info("Starting evaluation...")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs = {
                "input_ids": batch["input_ids"].to(accelerator.device),
                "attention_mask": batch["attention_mask"].to(accelerator.device),
            }
            labels = batch["labels"].to(accelerator.device)

            outputs = model.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(accelerator.gather(predictions).cpu().numpy())
            all_labels.extend(accelerator.gather(labels).cpu().numpy())

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

    # Save results
    os.makedirs(custom_args.output_dir, exist_ok=True)
    results_path = os.path.join(custom_args.output_dir, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")

    logger.info(f"Test results saved to {results_path}")

if __name__ == "__main__":
    main() 