from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time 
import psutil

from random import sample
import multiprocessing as mp
from multiprocessing import Pool

import os.path

def mol_to_smiles(mol):
    return Chem.MolToSmiles(mol)

def get_logp(mol):
    return MolLogP(mol)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def main():
    process = psutil.Process()
    start = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--our_data', type = str, default="data/generated/moses_canonical_CL_1.csv",
                        help="where is your sample", required=True)

    parser.add_argument('--training_data', type = str, default='data/training_data/moses_canonical.csv',
                        help="where are training data to compare", required=True)
    
    parser.add_argument('--output_dir', type = str, default='reports/figures/moses_canonical_CL/1',
                        help="where save our outputs", required=False)
    
    
    parser.add_argument('--number_worker', type = int, default=5,
                        help="number of worker", required=False)
    
    parser.add_argument('--sample_training_set', type =int, default=0,
                        help="a bool representing if we take a smaller part of training set for analysis because there are issues with memory", required=False)

    args = parser.parse_args()


    df = pd.read_csv(args.our_data)
    dataset_generated = df["SMILES"].to_list()
    del df

    df = pd.read_csv(args.training_data)
    #test_set = df[df["SPLIT"]=="test"]["SMILES"].to_list()
    training_set = df[df["SPLIT"]=="train"]["SMILES"].to_list()
    
    del df
    if args.sample_training_set :

        training_set = sample(training_set,1000000)

    mp.set_start_method('spawn')
    p = Pool(args.number_worker)
    with p :
        list_mol_generated = p.map(get_mol, dataset_generated)
    #getting the valid mols
    print("Mols Generated")
    generated_mol_valid = [k for k in list_mol_generated if isinstance(k,Chem.rdchem.Mol)]

    p = Pool(args.number_worker)
    with p :
        training_mol = p.map(get_mol, training_set)
    print("Mol of training set Generated")


    p = Pool(args.number_worker)
    with p :
        generated_logp = p.map(get_logp, generated_mol_valid)
    print("LOGP Generated")
        
    split_txt = os.path.splitext(args.training_data)
    if not os.path.isfile(split_txt[0] + "_LOGP"+split_txt[1]):
        p = Pool(args.number_worker)
        with p : 
            training_logp = p.map(get_logp, training_mol)
        df = pd.DataFrame()
        df["LOGP"] = training_logp
        df.to_csv(split_txt[0] + "_LOGP"+split_txt[1])
        del df
    else :
        training_logp = pd.read_csv(split_txt[0] + "_LOGP"+split_txt[1])["LOGP"].to_list()
    end = time.time()
    print(str(end-start) + " is the time to get after logp calculation")
    print("LOGP of training set Generated")
    p = Pool(args.number_worker)
    with p :
        generated_qed = p.map(qed, generated_mol_valid)
    print("QED Generated")
    if not os.path.isfile(split_txt[0] + "_QED"+split_txt[1]):
        p = Pool(args.number_worker)
        with p :  
            training_qed = p.map(qed, training_mol)
        df = pd.DataFrame()
        df["QED"] = training_qed
        try:
            df.to_csv(split_txt[0] + "_QED"+split_txt[1])
        except FileNotFoundError as e:
            print(e)
        del df
    else:
        training_qed = pd.read_csv(split_txt[0] + "_QED"+split_txt[1])["QED"].to_list()
    print("QED of training set Generated")
    end = time.time()
    print(str(end-start) + " is the time to get after qed calculation")
    print(str(process.memory_info().rss/1000000000) + " GB of usage") 
    
    #Creating the save folder of plots
    
    try:
        os.makedirs(args.output_dir)
        print(f"Directory '{args.output_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{args.output_dir}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{args.output_dir}'.")
    except Exception as e:
        print(f"An error occurred: {args.output_dir}")

    #PLOTING STUFFS

    qed_max1 = max(generated_qed)
    qed_min1 = min(generated_qed)
    qed_max2 = max(training_qed)
    qed_min2 = min(training_qed)
    qed_min = min(qed_min1, qed_min2)
    qed_max = max(qed_max1, qed_max2)

    bin_width = (qed_max - qed_min) / 500
    x_bins = [qed_min + k * bin_width for k in range(501)]
    # Create the plot
    fig, ax = plt.subplots()
    # Plot histograms
    ax.hist(generated_qed, bins=x_bins, alpha=0.6, label='generated sample', density=True)
    ax.hist(training_qed, bins=x_bins, alpha=0.6, label='training sample', density=True)
    
    # Optionally: overlay scatter points at y=0 for both datasets
    ax.plot(generated_qed, [0]*len(generated_qed), 'd', label='generated points', alpha=0.5)
    ax.plot(training_qed, [0]*len(training_qed), 'o', label='training points', alpha=0.5)

    # Labels and title
    ax.set_ylabel('Number per bin')
    ax.set_xlabel('QED')
    ax.set_title('Density of SMILES according to QED ')
    ax.legend()
    plt.savefig(args.output_dir +"/QED_distribution.png")

    qed_max1 = max(generated_logp)
    qed_min1 = min(generated_logp)
    qed_max2 = max(training_logp)
    qed_min2 = min(training_logp)
    qed_min = min(qed_min1, qed_min2)
    qed_max = max(qed_max1, qed_max2)

    bin_width = (qed_max - qed_min) / 500
    x_bins = [qed_min + k * bin_width for k in range(501)]
    # Create the plot
    fig, ax = plt.subplots()
    # Plot histograms
    ax.hist(generated_logp, bins=x_bins, alpha=0.6, label='generated sample', density=True)
    ax.hist(training_logp, bins=x_bins, alpha=0.6, label='training sample', density=True)
    
    # Optionally: overlay scatter points at y=0 for both datasets
    ax.plot(generated_logp, [0]*len(generated_logp), 'd', label='generated points', alpha=0.5)
    ax.plot(training_logp, [0]*len(training_logp), 'o', label='training points', alpha=0.5)

    # Labels and title
    ax.set_ylabel('Number per bin')
    ax.set_xlabel('LogP')
    ax.set_title('Density of SMILES according to LogP ')
    ax.legend()
    plt.savefig(args.output_dir +"/LogP_distribution.png")

if __name__ == "__main__":
    main()
