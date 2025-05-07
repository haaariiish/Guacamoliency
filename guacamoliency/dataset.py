from pathlib import Path
import moses 
from guacamol import SMILESDataset
import argparse

from tqdm import tqdm

import pandas as pd


def main():
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
    main()
