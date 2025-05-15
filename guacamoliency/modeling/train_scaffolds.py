from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import tokenizers
import argparse 
import pandas as pd
from torch import nn
import torch
from datasets import Dataset
from functools import partial
from dataset import ScaffoldCompletionDataset
import os


def main():


    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type = str, default='moses_canonical',
                    help="which type of datasets to use for the training", required=True)
    
    parser.add_argument('--dataset_dir', type = str, default='data/traning_data/moses_canonical.csv',
                    help="which directory for the dataset", required=True
    )

    parser.add_argument('--model_dir', type = str, default='models/trained_moses_canonical/2/final_model',
                            help="where is your model file", required=True)

    parser.add_argument('--log_dir', type = str, default='reports',
                        help="where save our logs", required=False)
    
    parser.add_argument('--model_save_folder', type = str,default='models/conditional_moses_canonical',
                         help="where save our model", required=False)
    
    parser.add_argument('--learning_rate',type=float,default= 5e-4,
                        help="learning rate used in training", required=False)
    
    parser.add_argument('--max_steps',type=int,default= 100000,
                        help="max steps used in training", required=False)
    
    parser.add_argument('--batch_size',type=int,default= 64,
                        help="batch size per device used in training", required=False)
    
    parser.add_argument('--save_steps',type=int,default= 20000,
                        help="how many steps between saves in training", required=False)
    
    parser.add_argument('--save_total_limit',type=int,default= 5,
                        help="how many time are we able to save during training", required=False)

    args = parser.parse_args()


    #LOAD EXISTING TOKENIZER AND MODEL

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True)
    #load datasets
    raw_dataset = pd.read_csv(args.dataset_dir)
    training_datasets = ScaffoldCompletionDataset(tokenizer=tokenizer,scaffolds=raw_dataset[raw_dataset["SPLIT"]=="train"]["MURCKO_SCAFFOLDS_SMILES"],full_smiles=raw_dataset[raw_dataset["SPLIT"]=="train"]["SMILES"])
    eval_datasets =  ScaffoldCompletionDataset(tokenizer=tokenizer,scaffolds=raw_dataset[raw_dataset["SPLIT"]=="test_scaffolds"]["MURCKO_SCAFFOLDS_SMILES"],full_smiles=raw_dataset[raw_dataset["SPLIT"]=="test_scaffolds"]["SMILES"])

    vocab_size = tokenizer.vocab_size

    #the model and verification of GPU good usage
    model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available(): 
        model.to("cuda")
    else :
        raise Exception("Install correctly CUDA or check your drivers")
    print(model.device)
    

    #Construction of valid log and model saving folder if it's already exist
    model_save_folder = args.model_save_folder 
    id_save = 1
    while os.path.isdir(os.getcwd() + "/"+model_save_folder +"/"+ str(id_save)):
        id_save +=1
    model_save_folder = model_save_folder+"/"+str(id_save)
    log_dir_end = args.datasets +"/"+ str(id_save)


    #training arguments
    training_args = TrainingArguments(
            output_dir = model_save_folder,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            eval_strategy="steps",
            per_device_train_batch_size=args.batch_size,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            logging_dir=f"{args.log_dir}/logs/"+log_dir_end,
            report_to="tensorboard",
            logging_steps=10_000,
            warmup_steps=10_000,
            dataloader_num_workers=4,
            gradient_accumulation_steps=1,
            fp16=True,
            remove_unused_columns=False
        )
    

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_datasets,
        eval_dataset=eval_datasets,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("Training start")

    trainer.train()
    trainer.save_model(model_save_folder+"/final_model")
    
    print("Data from this directory : " + args.dataset_dir)
    print("Tokenizer from this directory : " + args.tokenizer_path)
    print("log files are in this directory : " + f"{args.log_dir}/logs/"+log_dir_end)
    print("Model and checkpoint of training has been save in this directory : " + model_save_folder)
    print("END OF train.py")
if __name__ == "__main__":
    main()

