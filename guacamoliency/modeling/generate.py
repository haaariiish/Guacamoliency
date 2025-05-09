from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse 
import pandas as pd
from torch import nn
import torch
from datasets import Dataset

from functools import partial




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default='moses',
                    help="on which datasets to use for the training", required=True)
    parser.add_argument('--output_dir', type = str, default='data/external',
                        help="where save our outputs", required=False)
    args = parser.parse_args()

    
    model_path = "models/trained_" + args.model
    tokenizer_path = "data/tokenizers/"+args.model+"_trained"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    
    """ generated_ids = model.generate(, max_new_tokens=50)
    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
    """


if __name__ == "__main__":
    main()
