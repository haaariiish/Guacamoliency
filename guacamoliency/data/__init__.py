"""
Module data : préprocessing et gestion des données SMILES
"""

from .preprocessor import SMILESPreprocessor

__all__ = [
    "SMILESPreprocessor"
]

# Configuration par défaut pour le preprocessing
DEFAULT_PREPROCESSING_CONFIG = {
    "min_smiles_length": 3,
    "max_smiles_length": 200,
    "remove_invalid": True,
    "clean_whitespace": True
}

# Fonctions utilitaires pour valider les SMILES
def is_valid_smiles(smiles) -> bool:
    """Vérifie si un SMILES est valide"""
    return isinstance(smiles, str) and smiles is not None and len(smiles.strip()) > 0

def is_valid_scaffold(scaffold) -> bool:
    """Vérifie si un scaffold est valide"""
    return isinstance(scaffold, str) and scaffold is not None and len(scaffold.strip()) > 0

def clean_smiles_string(smiles: str) -> str:
    """Nettoie une chaîne SMILES"""
    if not isinstance(smiles, str):
        return ""
    return smiles.strip().replace(" ", "")

# Factory function pour créer un preprocessor
def create_preprocessor(config=None) -> SMILESPreprocessor:
    """Factory function pour créer un preprocessor"""
    if config is None:
        from ..core.config import SMILESConfig
        config = SMILESConfig()
    return SMILESPreprocessor(config)

__all__.extend([
    "DEFAULT_PREPROCESSING_CONFIG",
    "is_valid_smiles",
    "is_valid_scaffold", 
    "clean_smiles_string",
    "create_preprocessor"
])