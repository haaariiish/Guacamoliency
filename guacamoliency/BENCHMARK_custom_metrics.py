from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP


import pandas as pd
import argparse
import numpy as np

import multiprocessing as mp
from multiprocessing import Pool


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

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type = str, default="data/generated/moses_canonical_BEP_2.csv",
                        help="where is your generated sample file", required=False)
    

    parser.add_argument('--output_dir', type = str, default='reports/data/moses_canonical_BEP/2/BENCHMARK_custom.csv',
                        help="where save our outputs", required=False)
    
    
    parser.add_argument('--number_worker', type = int, default=1,
                        help="number of worker", required=False)

    args = parser.parse_args()

    df = pd.read_csv(args.input_dir)


    dataset = df["SMILES"].to_list()



    #get the mol of each generated smiles
    mp.set_start_method('spawn')
    p = Pool(args.number_worker)
    with p :

        list_mol = p.map(get_mol, dataset)

    
    #print(list_mol)
    #getting the valid mols
    list_mol_valid = [k for k in list_mol if isinstance(k,Chem.rdchem.Mol)]
    print(len(list_mol_valid))
    #validity rate
    validness = len(list_mol_valid)/len(list_mol)
    print("the rate of valid mols generated is : ")
    print(validness)
    
    #calculation of QED of Mols and mean of QED

    p = Pool(args.number_worker)
    with p :
        list_qed = p.map(qed, list_mol_valid)
    meanqed = 0

    for k in list_qed:
        meanqed += k
    meanqed = meanqed/len(list_qed)
    print("the mean QED is : ")

    print(meanqed)

    #calculation of the uniqueness
    

    p = Pool(args.number_worker)
    with p :
        list_canon_smiles = p.map(mol_to_smiles,list_mol_valid)

    
    set_canon_smiles = set(list_canon_smiles)
    uniqueness = len(set_canon_smiles)/len(list_canon_smiles)
    print("the rate of unique mols generated is : ")
    print(uniqueness)
    

   #calculation of logp of Mols and mean of logp
    
    p = Pool(args.number_worker)
   
    with p :

        list_logp = p.map(get_logp, list_mol_valid)

    meanlogp = 0
    for k in list_logp:
        meanlogp += k
    meanlogp = meanlogp/len(list_logp)
    print("the mean logP is : ")
    print(meanlogp)

    """data_benchmark = pd.DataFrame()
    data_benchmark["METRICS"] = keys
    data_benchmark["VALUE"] = value
    data_benchmark.to_csv(args.output_dir)"""
    metrics =  ["Validity","Unique","Qed","LogP"]
    value = [validness,uniqueness,meanqed,meanlogp]
    df = pd.DataFrame()
    df["METRICS"] = metrics
    df["VALUE"] = value
    df.to_csv(args.output_dir)




if __name__ == "__main__":
    main()