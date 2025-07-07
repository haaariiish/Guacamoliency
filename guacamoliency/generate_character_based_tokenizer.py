from pathlib import Path

from tokenizers import Regex
from tqdm import tqdm
from tokenizers.processors import TemplateProcessing
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Split
import re
from transformers import PreTrainedTokenizerFast
import pandas as pd



def main(

):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type = str, default='moses_canonical',
                    help="which datasets to use for the training", required=True)
    parser.add_argument('--input_dir',type=str,default="data/data_tokenizer/moses_canonical.csv")
    args = parser.parse_args()
   
    # dataset loaded
    dataset = pd.read_csv(args.input_dir)
    # if dataset has a column "block", we concatenate the SMILES and block columns
    if "block" in dataset.columns:
        dataset = dataset['SMILES'].tolist()+ dataset['block'].tolist()
    #else if dataset has a column "SMILES", we only take the SMILES column
    else:
        dataset = dataset['SMILES'].tolist()
    # Remove non-string entries from the dataset
    dataset = [k for k in dataset if type(k) == str ]

    special_tokens = ["<bos>", "<eos>", "<pad>","<unk>"]




     # approche simple mais faut ajouter tous les termes entre crochet (le pattern va splitter [nH] en "[nH]" mais ne prend pas en compte les crochets seuls)

    # pattern used to split the SMILES
    pattern =r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|[0-9])"
    
    
    
    """alphabet = [
        "[", "]", "<", ">", "Br", "Cl", "B", "C", "N", "O", "S", "P", "F", "I",
        "b", "c", "n", "o", "s", "p",
        "(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", "*", "$"
    ]  + [str(i) for i in range(10)]"""
    #initialisation of the alphabet with special tokens
    # and the tokens used in the pattern
    alphabet = []
    alphabet += special_tokens
    alphabet = set(alphabet)
    model_max_length = 0
    for k in tqdm(dataset):
        sequence = re.findall(pattern,k)
        model_max_length = max(len(sequence),model_max_length)
        for l in sequence:
            alphabet.add(l)

    # Convert the alphabet into a dict with tokens as keys and indices as values
    vocabulary = {token: idx for idx, token in enumerate(alphabet)}
    # Wordlevel tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocabulary,unk_token = "<unk>"))

    # Utiliser Split comme pré-tokeniseur basé sur regex
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)

    tokenizer.add_special_tokens(special_tokens)
    # what shape of sequence are we expecting
    tokenizer.post_processor = TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
            ("<unk>", tokenizer.token_to_id("<unk>"))
        ]
    )
        

    # Conversion PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.unk_token = "<unk>"
    # Set the maximum length of the tokenizer
    # if the input is blocksmiles, we need to double the max length and add 3 for the special tokens
    if "blocksmiles" in args.input:
        model_max_length = model_max_length*2 + 3 
    fast_tokenizer.model_max_length = model_max_length

    # Save the tokenizer
    fast_tokenizer.save_pretrained("data/tokenizers_character_level/"+args.input)
    print("Done")

if __name__ == "__main__":
    main()
