
#function for tokenization of SMILES
def tokenize_func(examples, tokenizer,max_length):
    smiles = examples["SMILES"]
    smiles = [s for s in smiles if isinstance(s, str) or s is not None]

    return tokenizer(
        smiles, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )



#function for tokenization of inchkey_encoding
def tokenize_func2(examples, tokenizer,max_length):
    smiles = examples["inchkey_encoding"]
    smiles = [s for s in smiles if isinstance(s, str) or s is not None]

    return tokenizer(
        smiles, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
#function for tokenization of SELFIES 
def tokenize_func3(examples, tokenizer,max_length):
    smiles = examples["SELFIES"]
    smiles = [s for s in smiles if isinstance(s, str) or s is not None]

    return tokenizer(
        smiles, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
#function for tokenization of blocksmiles
def tokenize_func_blocksmiles(examples, tokenizer,max_length):
    smiles = examples["SMILES"]
    blocks= examples["block"]
    
    return tokenizer(
        blocks, smiles,
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )


"""def debug_func(example,tokenizer):
    ids = tokenizer(example["SMILES"])["input_ids"]
    
    for token_id in ids:
        if len(token_id) >= tokenizer.vocab_size:
            print(" Token ID exceeds vocab size:", token_id, ">= vocab_size", tokenizer.vocab_size)
    return tokenizer(example["SMILES"])
"""
from trainer import WeightedLanguageModelLoss, CustomWeightedTrainer

def main():
    from pathlib import Path


    from tqdm import tqdm


    from transformers import TrainingArguments
    from transformers import GPT2Config
    from transformers import GPT2LMHeadModel
    from transformers import AutoTokenizer
    from transformers import DataCollatorForLanguageModeling
    import tokenizers
    import argparse 
    from transformers import Trainer
    import pandas as pd
    from torch import nn
    import torch
    from datasets import Dataset
    from functools import partial
    import numpy as np


    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type = str, default='moses_canonical',
                    help="which type of datasets to use for the training", required=True)
    
    parser.add_argument('--dataset_dir', type = str, default='data/traning_data/moses_canonical.csv',
                    help="which directory for the dataset", required=True
    )

    parser.add_argument('--tokenizer_path', type = str, default="data/tokenizersBEP/moses_canonical",
            help="which directory for the tokenizer", required=True
    )

    parser.add_argument('--log_dir', type = str, default='reports',
                        help="where save our logs", required=False)
    
    parser.add_argument('--model_save_folder', type = str,default='models/trained_moses_canonical',
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
    
    parser.add_argument('--loss_fc',type=str,default="Cross_Entropy" ,
                        help="loss function used during training", required=False)

    args = parser.parse_args()


    #configure tokenizer with the path given in the arguments, this has to be the path for the folder containing the tokenizer json files
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)



    #load raw datasets containing the data for train our model (SMiles or SElfies)
    data_set = pd.read_csv(args.dataset_dir)

    #split between training and evaluation set
    training_set = data_set[data_set['SPLIT']=='train']
    training_set = Dataset.from_pandas(training_set)

    eval_set = data_set[data_set['SPLIT']=='test']
    eval_set = Dataset.from_pandas(eval_set)

    #different way to tokenize the data depending on the data, must be specified in the argument "tokenizer_type"
    if args.tokenizer_type == "INCHIES":
        encoded_training_set = training_set.map(partial(tokenize_func2, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True,  remove_columns=training_set.column_names)

        encoded_eval_set = eval_set.map(partial(tokenize_func2, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True, remove_columns=eval_set.column_names)
    
    elif args.tokenizer_type == "SELFIES":
        encoded_training_set = training_set.map(partial(tokenize_func3, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True,  remove_columns=training_set.column_names)

        encoded_eval_set = eval_set.map(partial(tokenize_func3, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True, remove_columns=eval_set.column_names)
    
    elif args.tokenizer_type == "blocksmiles":
        encoded_training_set = training_set.map(partial(tokenize_func_blocksmiles, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True,  remove_columns=training_set.column_names)

        encoded_eval_set = eval_set.map(partial(tokenize_func_blocksmiles, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True, remove_columns=eval_set.column_names)
    
    else :
        encoded_training_set = training_set.map(partial(tokenize_func, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True,  remove_columns=training_set.column_names)
        encoded_eval_set = eval_set.map(partial(tokenize_func, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True, remove_columns=eval_set.column_names)
        
    
    vocab_size = tokenizer.vocab_size
    #print(vocab_size)
    #configuration of model
    config =   GPT2Config(
            vocab_size=vocab_size,  # 10,000 tokens( pour BEP )
            n_positions=tokenizer.model_max_length , # ça ne génèrera que des smiles de la même taille
            n_ctx=tokenizer.model_max_length,  # ça ne génèrera que des smiles de la même taille
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            resid_pdrop=args.resid_pdrop,
            embd_pdrop=args.embd_pdrop,
            attn_pdrop=args.attn_pdrop
        )

    #the model and verification of GPU good usage
    model = GPT2LMHeadModel(config)
    #model.resize_token_embeddings(len(tokenizer))
    #print(len(tokenizer))

    #use of an other loss function than the default cross entropy
    if args.loss_fc == "Weighted_Cross_Entropy":
        idf =[0]*len(tokenizer) 
        D = len(data_set["SMILES"])
        for k in tqdm(tokenizer.get_vocab().keys()):
            for l in data_set["SMILES"]:
                if k in l:
                    idf[tokenizer.get_vocab()[k]] +=1
        idf = torch.tensor([np.log(D/k) if k!=0 else 0 for k in idf], dtype=torch.float32)
        print("Idf built")

    #check if cuda is available and set the model to the good device
    print("Is cuda/gpu available : ")
    print(torch.cuda.is_available())
    if torch.cuda.is_available(): 
        model.to("cuda")
        print("Cuda correctly setup to device")
        if args.loss_fc == "Weighted_Cross_Entropy":
            print("Change of device for idf done")
            idf = idf.to("cuda")
    else :
        raise Exception("Install correctly CUDA or check your drivers")
    print(model.device)
    
    
    #Construction of valid log and model saving folder if it's already exist
    model_save_folder = args.model_save_folder + "_" + args.tokenizer_type
    id_save = 1
    while os.path.isdir(os.getcwd() + "/"+model_save_folder +"/"+ str(id_save)):
        id_save +=1
    model_save_folder = model_save_folder+"/"+str(id_save)
    log_dir_end = args.datasets + "_" + args.tokenizer_type +"/"+ str(id_save)
    os.makedirs(model_save_folder, exist_ok=True)
    if args.loss_fc == "Weighted_Cross_Entropy":
        criterion = WeightedLanguageModelLoss(vocab_weights=idf)
        print("cross-entropy weighted function done")
    

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
            weight_decay = 0.1
        )
    

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)
    
    #different trainer with different loss function
    if args.loss_fc == "Weighted_Cross_Entropy":
        print("cross-entropy WEIGHTED function used")   
        trainer = CustomWeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_training_set,
            eval_dataset=encoded_eval_set,
            processing_class=tokenizer,
            data_collator=data_collator,
            loss_function=criterion

        )
    else : 
        print("cross-entropy function used")   
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_training_set,
        eval_dataset=encoded_eval_set,
        processing_class=tokenizer,
        data_collator=data_collator
        )


    print("Training start")
    #training 
    trainer.train()
    
    #save the model and the tokenizer
    trainer.save_model(model_save_folder+"/final_model")
    
    print("Data from this directory : " + args.dataset_dir)
    print("Tokenizer from this directory : " + args.tokenizer_path)
    print("log files are in this directory : " + f"{args.log_dir}/logs/"+log_dir_end)
    print("Model and checkpoint of training has been save in this directory : " + model_save_folder)
    print("END OF train.py")
if __name__ == "__main__":

    import os
    # Disable parallelism in tokenizers to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()
