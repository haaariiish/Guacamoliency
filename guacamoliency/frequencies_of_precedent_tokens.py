from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import  AutoTokenizer
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import os
import numpy as np

# 4. Forward personnalisé : utilise les embeddings au lieu de input_ids
def forward_func(embeds,model):
    outputs = model(inputs_embeds=embeds)
    # Prédiction du prochain token (dernier logit)
    return outputs.logits[:, -1, :]  # shape [1, vocab_size]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default='moses_canonical_BEP',
                    help="on which datasets to use for the training", required=True)
    
    parser.add_argument('--model_dir', type = str, default='models/trained_moses_canonical_BEP/2/final_model',
                        help="where is your model file", required=True)
    
    parser.add_argument('--dataset', type = str, default='data/generated/moses_canonical_BEP_2.csv',
                        help="where is your generated data", required=True)
    
    parser.add_argument('--output_dir', type = str, default='reports/data/moses_canonical_BEP/2/frequency_precedent_token.png',
                        help="where save our outputs", required=True)

    parser.add_argument('--verified_token', type = str, default=")",
                        help="the token you choose to check before , must be in the vocabulary of your tokenizer", required=False)
    
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.model_max_length += 1 
    dataset = pd.read_csv(args.dataset)["SMILES"].to_list()
    dataset = [k for k in dataset if isinstance(k,str)]

    counting = [0] * len(tokenizer)

    for k in tqdm(dataset):
        inputs_tokens = tokenizer.tokenize(k, return_tensors="pt")
        
        for l in range(len(inputs_tokens)):
            if inputs_tokens[l] == args.verified_token:
                if l!=0:
                    counting[tokenizer.convert_tokens_to_ids(inputs_tokens[l-1])]+=1

    

    
    x_bins = [ k  for k in range(len(tokenizer)+1)]
    #print(len(tokenizer))
    #print(len(counting))
    print(counting)
    vocabulary =  tokenizer.convert_ids_to_tokens(x_bins[:len(x_bins)-1])
    #print(len(vocabulary))
    fig, ax = plt.subplots()

    saliency_data = pd.DataFrame()
    saliency_data["vocabulary"] = vocabulary
    saliency_data["counting"] = counting

    spliting = os.path.splitext(args.output_dir)
    
    dir_for_data = spliting[0]+"_sample_length"+str(len(dataset))

    saliency_data.to_csv(dir_for_data+".csv")



    vocabulary = ["" if counting[k]==0 else vocabulary[k] for k in range(len(vocabulary))]

    ax.bar(vocabulary,counting, alpha=0.6, label=args.model)

    
    ax.set_ylabel('Number per bar')
    ax.set_xlabel('tokens')
    ax.set_title('Frequence of the precedent token - token looked at : '+ args.verified_token)
    ax.legend()
    plt.savefig(dir_for_data+spliting[1])  
    
    

if __name__ == "__main__":
    main()
