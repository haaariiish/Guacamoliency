from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import torch 
from captum.attr import Saliency
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
    
    parser.add_argument('--output_dir', type = str, default='reports/data/moses_canonical_BEP/2/relation_between_saliency_tokens-distance.png',
                        help="where save our outputs", required=True)

    parser.add_argument('--verified_token', type = str, default=")",
                        help="the token you choose for the saliency analysis , must be in the vocabulary of your tokenizer", required=False)
    
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.model_max_length += 1 
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    dataset = pd.read_csv(args.dataset)["SMILES"].to_list()
    dataset = [k for k in dataset if isinstance(k,str)]

    total_scoring = [0] * tokenizer.model_max_length
    counting_occurences = [0] * tokenizer.model_max_length
    sf = torch.nn.Softmax(1)
    sf2 = torch.nn.Softmax(0)
    saliency = Saliency(partial(forward_func, model=model))
    for k in tqdm(dataset):
        inputs_tokens = tokenizer.tokenize(k, return_tensors="pt")
        inputs = tokenizer(k, return_tensors="pt")
        
        idxs = []
        for l in range(len(inputs_tokens)):
            if inputs_tokens[l] == args.verified_token:
                idxs.append(l)

        for idx in idxs:
            
            input_ids = inputs["input_ids"][0][:idx+1]
            input_ids = torch.tensor([[k for k in input_ids]])
            
            inputs_embeds = model.transformer.wte(input_ids)
            inputs_embeds.requires_grad_()
            #prédiction du token suivant
            
            with torch.no_grad():
                out = model(input_ids=input_ids)
                #print(out.logits.shape)
                #pred_token_id = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        
                #print(tokenizer.convert_ids_to_tokens(pred_token_id))
                target_ids = tokenizer.convert_tokens_to_ids(args.verified_token)
                
                attributions = saliency.attribute(inputs_embeds, target=target_ids, abs=True)
                scores = attributions.sum(dim=-1).squeeze()
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])


            max_arg = sf2(scores)

            for i in range(len(max_arg)):
                total_scoring[len(max_arg)-i-1] += max_arg[i]
                counting_occurences[len(max_arg)-i-1] += 1
            
    #print(total_scoring)
    total_scoring = [(total_scoring[k]/counting_occurences[k]).item() if counting_occurences[k]!=0 else total_scoring[k].item() for k in range(len(total_scoring)) ]
    x_bins = [k for k in range(len(total_scoring))]
    #print(total_scoring)
    #print(counting_occurences)

    
    fig, ax = plt.subplots()


    spliting = os.path.splitext(args.output_dir)

    dir_for_data = spliting[0]+"_sample_length"+str(len(dataset))

    df = pd.DataFrame()
    df["total_scores"] = total_scoring
    df["occurences"] = counting_occurences
    df.to_csv(dir_for_data+".csv")

    ax.plot(x_bins,total_scoring,label=args.model)
    ax.set_ylabel('Mean of normalised score')
    ax.set_xlabel('Distance to the generated token')
    ax.set_title('Saliency of models according to the distance - token chosen : '+ args.verified_token)
    ax.legend()

    plt.savefig(dir_for_data+spliting[1])  

if __name__ == "__main__":
    main()
