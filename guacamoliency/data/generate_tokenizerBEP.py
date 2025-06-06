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

    parser.add_argument('--model_max_length', type = int, default=60,
                    help="words max length", required=True)
    
    parser.add_argument('--vocab_size', type = int, default=10000,
                    help="tokenizer vocab size", required=False)
    
    args = parser.parse_args()
   
    dataset = pd.read_csv(args.input_dir)
    dataset = dataset['SMILES'].tolist()
    dataset = [k for k in dataset if type(k) == str ]

    special_tokens = ["<bos>", "<eos>", "<pad>"]

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    alphabet = [
        "[", "]", "<", ">", "Br", "Cl", "B", "C", "N", "O", "S", "P", "F", "I",
        "b", "c", "n", "o", "s", "p",
        "(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", "*", "$"
    ] + [f"%{i:02d}" for i in range(100)] + [str(i) for i in range(10)]



    tokenizer = Tokenizer(models.BPE())

 
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=alphabet,
        special_tokens=special_tokens
    )

    tokenizer.train_from_iterator(dataset, trainer=trainer)
    tokenizer.add_special_tokens(special_tokens)

    tokenizer.post_processor = TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
            ("<pad>", tokenizer.token_to_id("<pad>"))
        ]
    )
        

    # Conversion vers PreTrainedTokenizerFast / UTILE? 
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.model_max_length = args.model_max_length

    # Sauvegarde du tokenizer
    fast_tokenizer.save_pretrained("data/tokenizersBEP/"+args.input)

if __name__ == "__main__":
    main()
