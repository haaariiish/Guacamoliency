from trl import AutoModelForCausalLMWithWalueHead , PPOConfig, PPOTrainer
from transformers import AutoTokenizer
from torch import nn
import torch

import pandas as pd
import argparse 
from datasets import Dataset
from functools import partial
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import QED


def reward_QED(smile):
    return QED.qed(Chem.MolFromSmiles(smile))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type = str, default='moses_canonical',
                    help="which type of datasets to use for the training", required=True)

    parser.add_argument('--tokenizer_path', type = str, default="data/tokenizersBEP/moses_canonical",
            help="which directory for the tokenizer", required=True
    )

    parser.add_argument('--log_dir', type = str, default='reports',
                        help="where save our logs", required=False)
    
    parser.add_argument('--model_dir', type = str,default='models/trained_moses_canonical',
                         help="where save our model", required=False)
    
    parser.add_argument('--learning_rate',type=float,default= 6e-4,
                        help="learning rate used in training", required=False)
    
    parser.add_argument('--max_steps',type=int,default= 41300,
                        help="max steps used in training", required=False)
    
    parser.add_argument('--batch_size',type=int,default= 384,
                        help="batch size per device used in training", required=False)
    
    parser.add_argument('--save_steps',type=int,default= 5000,
                        help="how many steps between saves in training", required=False)
    
    parser.add_argument('--save_total_limit',type=int,default= 5,
                        help="how many time are we able to save during training", required=False)
    
    parser.add_argument('--n_embd',type=int,default= 256,
                        help="", required=False)
    
    parser.add_argument('--n_layer',type=int,default= 8,
                        help="", required=False)
    
    parser.add_argument('--n_head',type=int,default= 8,
                        help="", required=False)
    
    parser.add_argument('--resid_pdrop',type=float,default= 0.1,
                        help="", required=False)
    
    parser.add_argument('--embd_pdrop',type=float,default= 0.1,
                        help="", required=False)
    
    parser.add_argument('--attn_pdrop',type=float,default= 0.1,
                        help="", required=False)
    
    parser.add_argument('--warmup_steps',type=int,default= 413,
                        help="warmup_steps before learning rate decrease", required=False)
    
    parser.add_argument('--lr_scheduler_type',type=str,default="cosine_with_min_lr",
                        help="Scheduler use by the optimiser for learning rate", required=False)
    
    
    parser.add_argument('--num_workers',type=int,default= 10,
                        help="", required=False)
    
    parser.add_argument('--tokenizer_type',type=str,
                        help="type de tokenizer utilisé", required=True)
    
    parser.add_argument('--loss_fc',type=str,default="Weighted_Cross_Entropy" ,
                        help="loss function used during training", required=False)

    args = parser.parse_args()

    #configure tokenizer with the path given in the arguments, this has to be the path for the folder containing the tokenizer json files
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    #load our pretrained model
    model = AutoModelForCausalLMWithWalueHead(args.model_dir)
    ref_model = AutoModelForCausalLMWithWalueHead(args.model_dir)


    ppo_config = PPOConfig(output_dir = args.model_save_folder,
            
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            eval_strategy="steps",
            per_device_train_batch_size=args.batch_size,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            logging_dir=f"{args.log_dir}",
            report_to="tensorboard",
            logging_first_step = 10,
            logging_strategy = "steps",
            logging_steps=2000,
            warmup_steps=args.warmup_steps, #should be 1% of max_steps
            dataloader_num_workers=args.num_workers,
            gradient_accumulation_steps=1,
            fp16=True,
            remove_unused_columns=False,
            lr_scheduler_type = args.lr_scheduler_type, #use of cosine with min_lr scheduler
            lr_scheduler_kwargs ={
            "min_lr": 0.1*args.learning_rate },
            adam_beta1 = 0.9,
            adam_beta2 = 0.95,
            weight_decay = 0.1)
    
    ppo_trainer = PPOTrainer(ppo_config,model,ref_model,tokenizer) 

    context = ""
    

    generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": tokenizer.model_max_length,
}
    

    

    reward



