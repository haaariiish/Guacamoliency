from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Split
import re
from transformers import PreTrainedTokenizerFast
import pandas as pd
#from guacamoliency.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(

):
   

   
    guacamol = pd.read_csv('data/interim/guacamol.csv')
    guacamol = guacamol['SMILES'].tolist()
    moses = pd.read_csv('data/interim/moses.csv')
    moses = moses['SMILES'].tolist()

    special_tokens = ["<bos>", "<eos>", "<pad>"]

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    alphabet = [
        "[", "]", "<", ">", "Br", "Cl", "B", "C", "N", "O", "S", "P", "F", "I",
        "b", "c", "n", "o", "s", "p",
        "(", ")", ".", "=", "#", "-", "+", "\\", "/", ":", "~", "@", "?", "*", "$"
    ] + [f"%{i:02d}" for i in range(100)] + [str(i) for i in range(10)]

    #FOR Guacamol

    tokenizer = Tokenizer(models.BPE())

    # Utiliser Split comme pré-tokeniseur basé sur regex
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, behavior="isolated", invert=False)
    trainer = trainers.BpeTrainer(
        vocab_size=10000,
        initial_alphabet=alphabet,
        special_tokens=["<pad>", "<bos>", "<eos>"]
    )

    tokenizer.train_from_iterator(moses, trainer=trainer)

    # Conversion vers PreTrainedTokenizerFast / UTILE? 
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"

    # Sauvegarde du tokenizer
    fast_tokenizer.save_pretrained("data/tokenizers/moses")




    #FOR MOSES
    tokenizer = Tokenizer(models.BPE())

    # Utiliser Split comme pré-tokeniseur basé sur regex
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, behavior="isolated", invert=False)
    trainer = trainers.BpeTrainer(
        vocab_size=10000,
        initial_alphabet=alphabet,
        special_tokens=["<pad>", "<bos>", "<eos>"]
    )

    tokenizer.train_from_iterator(guacamol, trainer=trainer)


    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    # Sauvegarde du tokenizer
    fast_tokenizer.save_pretrained("data/tokenizers/guacamol")
    # -----------------------------------------


if __name__ == "__main__":
    app()
