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

    parser.add_argument('--input_dir',type=str,default="data/training_data/blocksmiles_simple.csv",
                        required=False)
    parser.add_argument('--conditional',action="store_true",default=False)
    args = parser.parse_args()
   


    special_tokens = ["<bos>", "<eos>", "<pad>","<unk>"]
    # load the initial alphabet from the CSV file
    alphabet = set()
    data = pd.read_csv(args.input_dir)["SMILES"].tolist()
    model_max_lenght = 0
    for k in data:
        splitting = k.split('_')
        model_max_lenght = max(model_max_lenght,len(splitting))
        for l in splitting:
            alphabet.add(l)
    
    for k in special_tokens:
        alphabet.add(k)
    alphabet.add("_")
    
    # Create a vocabulary mapping each token to a unique index
    vocabulary = {token: idx for idx, token in enumerate(alphabet)}

    # create a tokenizer with the vocabulary
    tokenizer = Tokenizer(models.WordLevel(vocabulary,unk_token = "<unk>"))

    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[_]"), behavior="removed")

    tokenizer.add_special_tokens(special_tokens)

    tokenizer.post_processor = TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
            ("<unk>", tokenizer.token_to_id("<unk>")),
            ("<pad>", tokenizer.token_to_id("<pad>"))
        ]
    )
        
    if args.conditional:
        model_max_lenght = 2*model_max_lenght + 3
    #PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.unk_token = "<unk>"
    fast_tokenizer.model_max_length = model_max_lenght

    # Save the tokenizer and test it in a basic way
    if args.conditional:
        fast_tokenizer.save_pretrained("data/tokenizers_blocksmiles/moses_canonical_conditional")
    else :
        fast_tokenizer.save_pretrained("data/tokenizers_blocksmiles/moses_canonical")
    print("Done")
    if args.conditional: 
        test = "C1=C-N2-C(-O-1)=N-C1=C-2-C(=O)-N(-C(=O)-N-1-C)"
        test_string = 'C1-C=C-C=C-C=1_C1=C-N2-C(-O-1)=N-C1=C-2-C(=O)-N(-C(=O)-N-1-C)_C-C_C-C'
        encoded = fast_tokenizer.encode(test,test_string)
        decoded = fast_tokenizer.decode(encoded)
    else:
        test_string = 'C1-C=C-C=C-C=1_C1=C-N2-C(-O-1)=N-C1=C-2-C(=O)-N(-C(=O)-N-1-C)_C-C_C-C'
        encoded = fast_tokenizer.encode(test_string)
        decoded = fast_tokenizer.decode(encoded)

    print("Input:", test_string)
    print("Encoded IDs:", encoded)
    print("Decoded Text:", decoded)

if __name__ == "__main__":
    main()
