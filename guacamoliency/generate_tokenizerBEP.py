from pathlib import Path


from tqdm import tqdm
from tokenizers.processors import TemplateProcessing
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Split,ByteLevel
import re
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast
import pandas as pd
#from guacamoliency.config import PROCESSED_DATA_DIR





def main(

):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type = str, default='moses',
                    help="which datasets to use for the training", required=True)
    
    parser.add_argument('--input_dir',type=str,default="data/data_tokenizer/moses.csv")
    
    parser.add_argument('--vocab_size', type = int, default=10000,
                    help="tokenizer vocab size", required=False)
    
    args = parser.parse_args()
   # dataset loaded
   
    
    
    
    dataset = pd.read_csv(args.input_dir)
     # if dataset has a column "block", we concatenate the SMILES and block columns
    if "block" in dataset.columns:
        dataset = dataset['SMILES'].tolist()+ dataset['block'].tolist()
    else: # else if dataset has a column "SMILES", we only take the SMILES column
        dataset = dataset['SMILES'].tolist()
    # Remove non-string entries from the dataset
    dataset = [k for k in dataset if type(k) == str ]

    special_tokens = ["<bos>", "<eos>", "<pad>"]
    # pattern used to split the SMILES
    pattern =r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|[0-9])"
    
    """alphabet = [
        "[", "]", "<", ">", "Br", "Cl", "B", "C", "N", "O", "S", "P", "F", "I",
        "b", "c", "n", "o", "s", "p",
        "(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", "*", "$"
    ] + [f"%{i:02d}" for i in range(100)] + [str(i) for i in range(10)]"""
    #initialisation of the alphabet with special tokens
    alphabet = []
    alphabet += special_tokens
    alphabet = set(alphabet)
    model_max_length = 0
    for k in tqdm(dataset):
        sequence = re.findall(pattern,k)
        model_max_length = max(len(sequence),model_max_length)
        for l in sequence:
            alphabet.add(l)
    alphabet = sorted(list(alphabet))

    
    #tokenizer initialization
    tokenizer = Tokenizer(models.BPE())

    #use a BPE tokenizer with an initial alphabet, and a vocab size that we can set (check argparser)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=alphabet,
        special_tokens=special_tokens
    )
    # BPE encoding algorithm
    tokenizer.train_from_iterator(dataset, trainer=trainer)
    tokenizer.add_special_tokens(special_tokens)
    # what shape of sequence are we expecting
    tokenizer.post_processor = TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
            ("<pad>", tokenizer.token_to_id("<pad>"))
        ]
    )
        

    # Conversion PreTrainedTokenizerFast 
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.unk_token = "<unk>"
    if "blocksmiles" in args.input:  # if the input is blocksmiles, we need to double the max length
        model_max_length = model_max_length*2 + 3
    fast_tokenizer.model_max_length = model_max_length

    # Save tokenizer
    fast_tokenizer.save_pretrained("data/tokenizersBEP/"+args.input)

if __name__ == "__main__":
    main()
