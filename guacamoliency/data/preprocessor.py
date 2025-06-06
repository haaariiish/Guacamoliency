import pandas as pd
from typing import List, Dict, Any, Optional
from functools import partial
from datasets import Dataset

class SMILESPreprocessor:
    """Préprocesseur pour données SMILES"""
    
    def __init__(self, config: SMILESConfig):
        self.config = config
    
    def is_valid_smiles(self, smiles: Any) -> bool:
        """Vérifie si un SMILES est valide"""
        return isinstance(smiles, str) and smiles is not None and len(smiles.strip()) > 0
    
    def is_valid_scaffold(self, scaffold: Any) -> bool:
        """Vérifie si un scaffold est valide"""
        return isinstance(scaffold, str) and scaffold is not None
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie le DataFrame"""
        # Filtrer les SMILES valides
        df = df[df["SMILES"].apply(self.is_valid_smiles)].copy()
        
        # Pour les scaffolds, filtrer si nécessaire
        if self.config.use_scaffolds and "MURCKO_SCAFFOLDS_SMILES" in df.columns:
            df["is_scaffolds"] = df["MURCKO_SCAFFOLDS_SMILES"].apply(self.is_valid_scaffold)
            df = df[df["is_scaffolds"]]
        
        return df
    
    def tokenize_smiles(self, examples: Dict[str, List], tokenizer, max_length: int) -> Dict[str, Any]:
        """Tokenise les SMILES simples"""
        smiles = examples["SMILES"]
        smiles = [s for s in smiles if self.is_valid_smiles(s)]
        
        return tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    def tokenize_scaffolds(self, examples: Dict[str, List], tokenizer, max_length: int) -> Dict[str, Any]:
        """Tokenise les SMILES avec scaffolds"""
        smiles = examples["SMILES"]
        scaffolds = examples["MURCKO_SCAFFOLDS_SMILES"]
        
        return tokenizer(
            scaffolds, smiles,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    def prepare_datasets(self, df: pd.DataFrame, tokenizer) -> Dict[str, Dataset]:
        """Prépare les datasets d'entraînement et validation"""
        # Nettoyer les données
        df_clean = self.clean_dataframe(df)
        
        # Séparer train/test
        train_df = df_clean[df_clean['SPLIT'] == 'train']
        test_split = 'test_scaffolds' if self.config.use_scaffolds else 'test'
        test_df = df_clean[df_clean['SPLIT'] == test_split]
        
        # Convertir en Dataset
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Choisir la fonction de tokenisation
        tokenize_func = self.tokenize_scaffolds if self.config.use_scaffolds else self.tokenize_smiles
        
        # Tokeniser
        tokenizer.model_max_length = 2 * tokenizer.model_max_length if self.config.use_scaffolds else tokenizer.model_max_length
        
        encoded_train = train_dataset.map(
            partial(tokenize_func, tokenizer=tokenizer, max_length=tokenizer.model_max_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        encoded_test = test_dataset.map(
            partial(tokenize_func, tokenizer=tokenizer, max_length=tokenizer.model_max_length),
            batched=True,
            remove_columns=test_dataset.column_names
        )
        
        return {
            "train": encoded_train,
            "test": encoded_test
        }
