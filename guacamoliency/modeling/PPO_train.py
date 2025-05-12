from pathlib import Path


from tqdm import tqdm


from transformers import TrainingArguments
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
import tokenizers
import argparse 
from transformers import Trainer
import pandas as pd
from torch import nn
import torch
from datasets import Dataset

from functools import partial
import trl   


def configure_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.model_max_length = 128
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<bos>"
    tokenizer.eos_token = "<eos>"
    return tokenizer

def tokenize_func(examples, tokenizer):
    smiles = examples["SMILES"]
    smiles = [str(s) for s in smiles if isinstance(s, str) or s is not None]

    return tokenizer(
        smiles, 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )


def main():
    

if __name__ == "__main__":
    main()