from reward_model import reward_QED1
from functools import partial
from reward_model import train_reward_model

from transformers import Autotokenizer

import argparse 
import pandas as pd

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer_path', type = str, default="data/tokenizersBEP/moses_canonical",
            help="which directory for the tokenizer", required=True
    )
    
    parser.add_argument('--model_dir', type = str,default='models/trained_moses_canonical',
                         help="where save our model", required=False)
    
    parser.add_argument('--dataset_dir', type = str, default='data/traning_data/moses_canonical.csv',
                    help="which directory for the dataset", required=True
    )

    args = parser.parse_args()

    train_data = pd.read_csv(args.dataset_dir)["SMILES"].to_list()

    finetuned_model = train_reward_model(train_data,args.model_dir,args.tokenizer_path,reward_QED1)

    finetuned_model.save()

