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

    parser.add_argument('--input_dir',type=str,default="data/training_data/inchie_logic_data/unique_inchi_keys.csv",
                        required=False)
    parser.add_argument('--model_max_length', type = int, default=14,
                    help="words max length", required=False)
    args = parser.parse_args()
   


    special_tokens = ["<bos>", "<eos>", "<pad>","<unk>"]
    # load the initial alphabet from the CSV file
    alphabet = pd.read_csv(args.input_dir)["inchi_key"].tolist()
    
    alphabet += special_tokens
    alphabet += ["_"]
    alphabet = set(alphabet)

    
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
            ("<unk>", tokenizer.token_to_id("<unk>"))
        ]
    )
        

    #PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.pad_token = "<pad>"
    fast_tokenizer.bos_token = "<bos>"
    fast_tokenizer.eos_token = "<eos>"
    fast_tokenizer.unk_token = "<unk>"
    fast_tokenizer.model_max_length = args.model_max_length

    # Save the tokenizer and test it in a basic way
    fast_tokenizer.save_pretrained("data/tokenizers_inchies/moses_canonical")
    print("Done")
    test_string = "QUGUFLJIAFISSW-UHFFFAOYSA-N_QGZKDVFQNNGYKY-UHFFFAOYSA-N_RAHZWNYVWXNFOC-UHFFFAOYSA-N_AJKNNUJQFALRIK-UHFFFAOYSA-N"
    encoded = fast_tokenizer.encode(test_string)
    decoded = fast_tokenizer.decode(encoded)

    print("Input:", test_string)
    print("Encoded IDs:", encoded)
    print("Decoded Text:", decoded)

if __name__ == "__main__":
    main()
