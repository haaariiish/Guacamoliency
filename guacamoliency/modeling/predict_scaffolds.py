from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse

def conversion(liste_tokens):
    merge = ""
    for k in liste_tokens:
        merge += k
    return merge


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default='moses_canonical_mscaffolds',
                    help="on which datasets to use for the training", required=True)
    
    parser.add_argument('--model_dir', type = str, default='models/trained_moses_canonical_mscaffolds/1/final_model',
                        help="where is your model file", required=True)

    parser.add_argument('--num_sequence', type = int, default=1000,
                        help="number of sequences generated from a scaffolds", required=False)
    
    parser.add_argument('--temperature', type = float, default=1,
                        help="temperature of generation", required=False)
    
    parser.add_argument('--output_dir', type = str, default='data/generated/moses_canonical_scaffolds.csv',
                        help="where save our outputs", required=True)
    
    parser.add_argument('--max_length', type = int, default= 120,
                        help="max number of tokens use in prompt and generation", required=False)
    
    parser.add_argument('--scaffolds', type = str, default='c1ccccc1',
                        help="which scaffold to use for prediction ", required=True)
    
    
    args = parser.parse_args()


    df_exit = pd.DataFrame()
    outputs_list = []
    inputs_list = [args.scaffolds] * args.num_sequence

    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokens = tokenizer(args.scaffolds,return_tensors="pt")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    for k in range(args.num_sequence):
        outputs = model.generate(tokens["input_ids"], max_new_tokens=tokenizer.model_max_length //2+1 , do_sample=True, top_k=50, top_p=0.95, attention_mask=tokens["attention_mask"])
        outputs_list.append(tokenizer.batch_decode(outputs[0][len(tokens["input_ids"][0]):len(outputs[0])-1], skip_special_tokens=False))
        #outputs_list.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
    outputs_list = [conversion(k) for k in outputs_list]
    df_exit["SMILES"] = outputs_list
    df_exit["SCAFFOLDS"] = inputs_list
    df_exit.to_csv(args.output_dir)


if __name__ == "__main__":
    main()
