from pathlib import Path

#from guacamol import SMILESDataset
import argparse
from tqdm import tqdm
import pandas as pd

from torch.utils.data import Dataset

class ScaffoldCompletionDataset(Dataset):
    def __init__(self, tokenizer, scaffolds, full_smiles, max_length=128):
        self.tokenizer = tokenizer
        self.scaffolds = scaffolds.to_list()
        self.full_smiles = full_smiles.to_list()
        self.max_length = max_length

    def __len__(self):
        return len(self.scaffolds)

    def __getitem__(self, idx):
        prompt = self.scaffolds[idx]
        target = self.full_smiles[idx]

        input_text = target  # on donne tout comme entrée
        encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = encoding.input_ids.clone()
        # Masquer les tokens du scaffold (pas d’apprentissage sur eux)
        num_prompt_tokens = len(self.tokenizer(prompt).input_ids) - 1  # enlever <eos>
        labels[0][:num_prompt_tokens] = -100

        return {
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
            "labels": labels.squeeze()
        }


"""def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type = str, default='moses',
                        help="which datasets to use for the training", required=True)
    parser.add_argument('--output_dir', type = str, default='data/interim',
                        help="where save our outputs", required=False)
    args = parser.parse_args()

    if args.datasets == 'moses':
        train = moses.get_dataset('train')
        test = moses.get_dataset('test')
        test_scaffolds = moses.get_dataset('test_scaffolds')
        SPLIT = []
        for k in range(len(test) + len(train)+len(test_scaffolds)):
            if k< len(train) : 
                SPLIT.append("train")
            elif k>= len(train) and k< len(train)+len(test):
                SPLIT.append("test")
            else :
                SPLIT.append("test_scaffolds")
        smiles = train + test + test_scaffolds

        dataset = pd.DataFrame({"SMILES":smiles
                                ,"SPLIT": SPLIT})
    
    elif args.datasets == 'guacamol':
        dataset = SMILESDataset(path_to_file)
        smiles_list = dataset.smiles



    dataset.to_csv(args.output_dir + "/" + args.datasets)
if __name__ == "__main__":
    main()"""
