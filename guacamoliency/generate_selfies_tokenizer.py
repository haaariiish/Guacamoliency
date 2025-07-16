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

    parser.add_argument('--input', type = str, default='moses_selfies',
                    help="which datasets to use for the training", required=False)
    parser.add_argument('--input_dir',type=str,default="data/training_data/moses_selfies.csv")
    parser.add_argument('--model_max_length', type = int, default=56,
                    help="words max length", required=False)
    args = parser.parse_args()
    # load dataset
    dataset = pd.read_csv(args.input_dir)
    dataset = dataset['SELFIES'].tolist()
    dataset = [k for k in dataset if type(k) == str ]

    special_tokens = ["<bos>", "<eos>", "<pad>","<unk>"]

    # Regex to catch all terms between brackets 
    pattern = r'\[[^\]]+\]'
     # approche simple mais faut ajouter tous les termes entre crochet (le pattern va splitter [nH] en "[nH]" mais ne prend pas en compte les crochets seuls)

    alphabet = set()

    for k in special_tokens:
        alphabet.add(k)
    
    for k in tqdm(dataset):
        for l in re.findall(pattern,k):
            alphabet.add(l)

    vocabulary = {token: idx for idx, token in enumerate(alphabet)}
    
    # initialize tokenizer with WordLevel model with the vocabulary
    # Note: models.WordLevel expects a vocabulary dictionary where keys are tokens and values are their corresponding indices.
    # The unk_token is set to "<unk>" to handle unknown tokens.
    tokenizer = Tokenizer(models.WordLevel(vocabulary,unk_token = "<unk>"))

    # Use Split pre-tokenizer to split the input text into tokens based on the regex pattern
    # The behavior is set to "isolated" to ensure that the tokens are treated as separate entities.
    # The invert parameter is set to False to ensure that the regex pattern is applied
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)

    tokenizer.add_special_tokens(special_tokens)

    tokenizer.post_processor = TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
            ("<unk>", tokenizer.token_to_id("<unk>"))
        ]
    )
        

    # Convert as a PreTrainedTokenizerFast 
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.unk_token = "<unk>"
    fast_tokenizer.model_max_length = args.model_max_length+2

    # Save tokenizer to the specified directory
    fast_tokenizer.save_pretrained("data/tokenizers_selfies/"+args.input)
    print("Done")

if __name__ == "__main__":
    main()
