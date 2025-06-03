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
    parser.add_argument('--model_max_length', type = int, default=60,
                    help="words max length", required=True)
    args = parser.parse_args()
   
    dataset = pd.read_csv(args.input_dir)
    dataset = dataset['SMILES'].tolist()
    dataset = [k for k in dataset if type(k) == str ]

    special_tokens = ["<bos>", "<eos>", "<pad>","<unk>"]




     # approche simple mais faut ajouter tous les termes entre crochet (le pattern va splitter [nH] en "[nH]" mais ne prend pas en compte les crochets seuls)


    pattern =r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    
    
    
    alphabet = [
        "[", "]", "<", ">", "Br", "Cl", "B", "C", "N", "O", "S", "P", "F", "I",
        "b", "c", "n", "o", "s", "p",
        "(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", "*", "$"
    ]  + [str(i) for i in range(10)]
    alphabet += special_tokens
    alphabet = set(alphabet)

    for k in tqdm(dataset):
        for l in re.findall(pattern,k):
            alphabet.add(l)

    vocabulary = {token: idx for idx, token in enumerate(alphabet)}
    

    tokenizer = Tokenizer(models.WordLevel(vocabulary,unk_token = "<unk>"))

    # Utiliser Split comme pré-tokeniseur basé sur regex
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
        

    # Conversion vers PreTrainedTokenizerFast / UTILE? 
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.unk_token = "<unk>"
    fast_tokenizer.model_max_length = args.model_max_length

    # Sauvegarde du tokenizer
    fast_tokenizer.save_pretrained("data/tokenizers_character_level/"+args.input)
    print("Done")

if __name__ == "__main__":
    main()
