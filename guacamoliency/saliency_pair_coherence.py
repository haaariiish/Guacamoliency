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
from rdkit import Chem

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
    
    parser.add_argument('--output_dir', type = str, default='reports/data/moses_canonical_BEP/2/pair_coherence_analysis.csv',
                        help="where save our outputs", required=True)

    parser.add_argument('--second_paired_token', type = str, default=")",
                        help="the token you choose for the saliency analysis , must be in the vocabulary of your tokenizer and form a pair as 11 or ()", required=False)

    parser.add_argument('--first_paired_token', type = str, default="(",
                        help="the token must be the first pair of the other token you choose , must be in the vocabulary of your tokenizer and form a pair as 11 or ()", required=False)
    
    parser.add_argument('--threshold', type = int, default=0,
                        help="threshold to consider when picking the maximum score, it is wise to take a threshold < 2", required=False)


    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.model_max_length += 1 
    
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    dataset = pd.read_csv(args.dataset)["SMILES"].to_list()
    dataset = [k for k in dataset if isinstance(k,str)] # si le modèle n'a rien généré, la case correspondante dans le csv est vide ce qui pose problème dans la suite
    dataset = [k for k in dataset if Chem.MolFromSmiles(k)]

    sf = torch.nn.Softmax(1)
    sf2 = torch.nn.Softmax(0)
    saliency = Saliency(partial(forward_func, model=model))

    info_dico = {
            "position of first of a pair": [],
            "position of second of a pair" : [],
            "score of first of a pair" : [],
            "token before the pair" : [],
            "score of the token before the pair" : [],
            "position of max score in saliency map" : [],
            "max score in saliency map" : [],
            "max score token": [],
            "sum of score between pair": [],
            "which smile" : []
        }
    
    data_dico = dict()
    sequence_dico = dict()
    count = 0
    iteration =0
    for k in tqdm(dataset):

        inputs_tokens = tokenizer.tokenize(k, return_tensors="pt")
        inputs = tokenizer(k, return_tensors="pt")

        idxs_first= dict()
        idxs_intermediaire = []
        for l in range(len(inputs_tokens)):
            "print(idxs_intermediaire)"
            if inputs_tokens[l] == args.first_paired_token:
                
                idxs_intermediaire.append(l)
            if inputs_tokens[l] == args.second_paired_token:
                #introduction d'un décalage de 1 car en tokenisant on rajoute le caractère spécial <bos> après
                idxs_first[l+1] = idxs_intermediaire.pop(-1)+1
            
        
        
        for idx in idxs_first.keys():
            info_dico["position of second of a pair"].append(idx)
            info_dico["position of first of a pair"].append(idxs_first[idx])
            
            input_ids = inputs["input_ids"][0][:idx]
            input_ids = torch.tensor([[k for k in input_ids]])
            
            inputs_embeds = model.transformer.wte(input_ids)
            inputs_embeds.requires_grad_()
            #prédiction du token suivant
            
            with torch.no_grad():
                out = model(input_ids=input_ids)
                #print(out.logits.shape)
                #pred_token_id = torch.argmax(out.logits[:, -1, :], dim=-1).item()
        
                #print(tokenizer.convert_ids_to_tokens(pred_token_id))


                target_ids = tokenizer.convert_tokens_to_ids(args.second_paired_token)
                
                attributions = saliency.attribute(inputs_embeds, target=target_ids, abs=True)
                scores = attributions.sum(dim=-1).squeeze()
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            scores = sf2(scores)
            scores = scores[:len(scores)-args.threshold]
            max_arg =scores.argmax()

           
            max_token = tokens[max_arg]


            
            precedent_score = scores[idxs_first[idx]-1].item()
            precedent_token = tokens[idxs_first[idx]-1]

            sum_score = torch.sum(scores[idxs_first[idx]+1:]).item()
            score_first_pair = scores[idxs_first[idx]].item()

            info_dico['sum of score between pair'].append(sum_score)
            info_dico['token before the pair'].append(precedent_token)
            info_dico['score of the token before the pair'].append(precedent_score)
            info_dico['position of max score in saliency map'].append(max_arg.item())
            info_dico['score of first of a pair'].append(score_first_pair)
            info_dico["max score in saliency map"].append(scores[max_arg].item())
            info_dico["max score token"].append(max_token)
            info_dico["which smile"].append(count)
            
            scores = scores.tolist()
            while tokenizer.model_max_length >= len(scores):
                scores.append(0)
            scores = [float("%.3g" % k) for k in scores]
            data_dico[iteration] = scores
            detokenize_sequence = ""
            
            for i in tokens:
                detokenize_sequence += i
            sequence_dico[iteration] = [detokenize_sequence]
            iteration +=1
        count += 1 

    spliting = os.path.splitext(args.output_dir)
    
    dir_for_data = spliting[0]+"_threshold"+str(args.threshold)+"_sample_length"+str(len(dataset))

    df = pd.DataFrame.from_dict(info_dico)
    df.to_csv(dir_for_data+spliting[1])

    df = pd.DataFrame.from_dict(data_dico)
    df.to_csv(spliting[0]+"_scores"+"_threshold"+str(args.threshold)+spliting[1])

    df = pd.DataFrame.from_dict(sequence_dico)
    df.to_csv(spliting[0]+"_sequences"+"_threshold"+str(args.threshold)+spliting[1])

if __name__ == "__main__":
    main()