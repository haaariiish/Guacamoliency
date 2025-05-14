from pathlib import Path


from tqdm import tqdm

import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Split
import re
from transformers import PreTrainedTokenizerFast
import pandas as pd
#from guacamoliency.config import PROCESSED_DATA_DIR





def main(

):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type = str, default='moses',
                    help="which datasets to use for the training", required=True)
    parser.add_argument('--input_dir',type=str,default="data/data_tokenizer/moses.csv")
    args = parser.parse_args()
   
    dataset = pd.read_csv(args.input_dir)
    dataset = dataset['SMILES'].tolist()
    dataset = [k for k in dataset if type(k) == str ]

    special_tokens = ["<bos>", "<eos>", "<pad>","Ġ"]

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    alphabet = [
        "[", "]", "<", ">", "Br", "Cl", "B", "C", "N", "O", "S", "P", "F", "I",
        "b", "c", "n", "o", "s", "p",
        "(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", "*", "$"
    ] + [f"%{i:02d}" for i in range(100)] + [str(i) for i in range(10)]



    tokenizer = Tokenizer(models.BPE())

    # Utiliser Split comme pré-tokeniseur basé sur regex
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, behavior="isolated", invert=False)
    trainer = trainers.BpeTrainer(
        vocab_size=10000,
        initial_alphabet=alphabet,
        special_tokens=["<pad>", "<bos>", "<eos>","Ġ"]
    )

    tokenizer.train_from_iterator(dataset, trainer=trainer)

    # Conversion vers PreTrainedTokenizerFast / UTILE? 
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.add_special_tokens({'additional_special_tokens': ["Ġ"]})

    # Sauvegarde du tokenizer
    fast_tokenizer.save_pretrained("data/tokenizers/"+args.input)


if __name__ == "__main__":
    main()
