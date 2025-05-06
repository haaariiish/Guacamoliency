from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

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





from guacamoliency.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()



def configure_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.model_max_length = 128
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<bos>"
    tokenizer.eos_token = "<eos>"
    return tokenizer

def tokenize_func(examples):
    smiles = examples["SMILES"]
    smiles = [str(s) for s in smiles if isinstance(s, str) or s is not None]
    
    tokenized = tokenizer(
        smiles, 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )


    return tokenized

@app.command()
def main(
):
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type = str, default='moses',
                        help="which datasets to use for the training", required=True)
    parser.add_argument('--output_dir', type = str, default='reports',
                        help="where save our outputs", required=False)
    args = parser.parse_args()
    #configure tokenizer
    tokenizer = configure_tokenizer("data/tokenizers/"+args.datasets+"/tokenizer.json")

    #load datasets
    
    data_set = pd.read_csv("data/interim/"+args.datasets+".csv")
    training_set = data_set[data_set['SPLIT']=='train']

    


    training_set = Dataset.from_pandas(training_set)

    


    eval_set = data_set[data_set['SPLIT']!='train']

    eval_set = Dataset.from_pandas(eval_set)


    #encoded_training_set = training_set.map(tokenize_func, batched=True, remove_columns=["SMILES"])

    encoded_eval_set = eval_set.map(tokenize_func, batched=True, remove_columns=eval_set.column_names)
    encoded_training_set = encoded_eval_set


    vocab_size = tokenizer.vocab_size

    #configuration of model

    config =   GPT2Config(
            vocab_size=vocab_size,  # 10,000 tokens
            n_positions=128,
            n_ctx=128,
            n_embd=256,
            n_layer=8,
            n_head=8,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )

    #the model
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available(): 
        model.to("cuda")
    else :
        print("Le modèle est chargé sur un CPU, attention !!!!!!")
        model.to("cpu")
    print(model.device)
    #training arguments

    training_args = TrainingArguments(
            output_dir = args.output_dir,
            
            learning_rate=5e-4,
            max_steps=100_000,
            per_device_train_batch_size=128,
            save_steps=10_000,
            save_total_limit=3,
            logging_dir=f"{args.output_dir}/logs/"+args.datasets,
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
        train_dataset=encoded_training_set,
        eval_dataset=encoded_eval_set,
        processing_class=tokenizer,
        data_collator=data_collator

    )

    print("Training start")


    trainer.train()
    tokenizer.save_pretrained("data/tokenizers/"+args.datasets+"_trained")
    trainer.save_model("models/trained_"+args.datasets)

if __name__ == "__main__":
    app()
