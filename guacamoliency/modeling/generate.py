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

    parser.add_argument('--model', type = str, default='moses_canonical',
                    help="on which datasets to use for the training", required=True)
    
    parser.add_argument('--model_dir', type = str, default='models/trained_moses_canonical/2/final_model',
                        help="where is your model file", required=True)

    
    parser.add_argument('--num_sequence', type = int, default=1000,
                        help="number of sequences generated", required=False)
    
    parser.add_argument('--temperature', type = float, default=1,
                        help="temperature of generation", required=False)
    
    parser.add_argument('--output_dir', type = str, default='data/generated/moses_canonical.csv',
                        help="where save our outputs", required=True)
    
    parser.add_argument('--max_length', type = int, default= 15,
                        help="max number of tokens use in prompt and generation", required=False)
    
    args = parser.parse_args()

    

    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    # load Model
    model = AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True)

    generated_ids = model.generate(
      max_length = args.max_length,
      num_return_sequences = args.num_sequence,
      pad_token_id = tokenizer.pad_token_id,
      bos_token_id = tokenizer.bos_token_id,
      eos_token_id = tokenizer.eos_token_id,
      do_sample = True,
      temperature = args.temperature,
      return_dict_in_generate = True,
  )
    #print(generated_ids.keys())
    generated_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_ids['sequences']]

    smiles_set = pd.DataFrame()

    smiles_set["SMILES"] = generated_smiles
    

    smiles_set.to_csv(args.output_dir)
if __name__ == "__main__":
    main()
