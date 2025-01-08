from peft import LoraConfig, get_peft_model
import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm
from eva_clip import create_model_and_transforms, create_model_from_pretrained
from training.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
from peft import LoraConfig, get_peft_model
from PIL import Image
import torch.nn.functional as F
import argparse
import torch
from PIL import Image
from eva_clip import create_model_and_transforms, create_model_from_pretrained
from training.llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
import torch.nn as nn

class LLM2VecWithProjection(nn.Module):
    def __init__(self, llm2vec_model, projection):
        super().__init__()
        self.model = llm2vec_model
        self.projection = projection
        # self.tokenizer = llm2vec_model.tokenizer

    def forward(self, text):
        embeddings = self.model(text)
        return self.projection(embeddings)

    def lock(self, unlocked_layers=0, freeze_layer_norm=True):
        # Freeze LLM2Vec weights but keep projection trainable
        for param in self.model.parameters():
            param.requires_grad = False

    def set_grad_checkpointing(self, enable=True):
        self.model.gradient_checkpointing_enable() if enable else self.model.gradient_checkpointing_disable()
# Initialize model and preprocessing
model, preprocess_train, preprocess_val = create_model_and_transforms(
    "EVA02-CLIP-L-14-336",
    "eva_clip",
    precision="fp16",
    device="cuda",
    force_custom_clip=True,
    image_mean=None,
    image_std=None,
    cache_dir=None,
    skip_list=None,
)

model.eval()

# Load pre-trained LLM2Vec model
text_model = LLM2Vec.from_pretrained(
    base_model_name_or_path="meta-llama/Llama-3.2-3B",
    enable_bidirectional=True,
    peft_model_name_or_path='/data/research/tmp/checkpoint-12600/',
    merge_peft=True,
    pooling_mode="mean",
    max_length=512,
    torch_dtype=torch.bfloat16,
)

# Add a trainable projection layer
projection_layer = nn.Sequential(
    nn.LayerNorm(text_model.config.hidden_size),
    nn.Linear(text_model.config.hidden_size, model.visual.head.out_features)
).to('cuda')

# Wrap LLM2Vec with projection
model.text = LLM2VecWithProjection(text_model.model, projection_layer)
# Load

## Load from pretrained checkpoint
model.load_state_dict(torch.load('/model/llm2clip/logs/T_vitl336_mimic-2025_01_06-11/checkpoints/output/pytorch_model.bin'), strict=False)
model.eval()
for param in model.parameters():
    # Check if parameter dtype is  Float (float32)
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.float16)
model.eval()

## Inference
import pandas as pd
df = pd.read_csv('/data/research/csv/rsna_test.csv')
df.head()


df['caption2'][0]

def load_image(image_path):
    """Load and preprocess the image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_val(image).unsqueeze(0).to('cuda')
    return image_tensor.to(torch.float16)


def encode_image(image_tensor):
    """Encode the image using the visual model."""
    with torch.no_grad():
        image_embedding = model.visual(image_tensor)
    return image_embedding

def encode_text(text, model, text_model, image_tensor):
    """Encode the text using the LLM2Vec model with projection."""
    original = text_model.tokenizer(
        # ["The patient's gender is female", "The patient's gender is male", "The patient's gender is unknown"],
        # ["This is a PA view", "This is a AP view"],
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    embed_mask = torch.zeros_like(original["attention_mask"])
    original["embed_mask"] = embed_mask
    l2v = LLM2Vec(model.text.model, text_model.tokenizer, pooling_mode="mean", max_length=512).to('cuda:0')
    text_features = l2v.forward(original.to(device='cuda:0'))
    with torch.no_grad():
        text_features = model.text.projection(text_features.to(dtype=image_tensor.dtype))
    return text_features

def compute_similarity(image_embedding, text_features):
    # Normalize the vectors
    image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # Compute cosine similarity
    # This will give you a [1, 10] tensor where each value is the cosine similarity
    # between the image embedding and each text feature
    similarity = (image_embedding @ text_features.T)
    
    return similarity


index = 4
text_test = {'gender': ["The patient's gender is female.", "The patient's gender is male."], 'view': ["This is a PA view, CXR image of the whole chest.", "This is a AP view, CXR image of the whole chest."],
             'report2': df['caption2'].tolist(), 'report': df['caption'].tolist(), 'age': [f"The patient's age is {x} years old." for x in df['age']],
             'pneumonia': ["Pneumonia is present.", "No signs of pneumonia"], 'pneumothorax': ['There is pneumothorax.', 'There is no pneumothorax.']}
print(df['caption2'][index])

image_tensor = load_image(df['img_path'].tolist()[index])
image_embedding = encode_image(image_tensor)

text_embedding = encode_text(text_test['report'], model, text_model, image_tensor)
similarity = compute_similarity(image_embedding, text_embedding)*model.logit_scale
print(f"Similarities shape: {similarity.shape}")  # Should be [1, 10]
print(f"Similarities: {similarity}")