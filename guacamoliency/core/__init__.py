"""
Module core : classes de base et configuration pour SMILES
"""

from .base_model import BaseSMILESModel
from .config import SMILESConfig

__all__ = [
    "BaseSMILESModel",
    "SMILESConfig"
]

# Configuration par défaut pour les modèles SMILES
DEFAULT_MODEL_CONFIG = {
    "n_embd": 256,
    "n_layer": 8,
    "n_head": 8,
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1
}

DEFAULT_TRAINING_CONFIG = {
    "learning_rate": 6e-4,
    "batch_size": 384,
    "max_steps": 41300,
    "warmup_steps": 413,
    "save_steps": 5000
}

DEFAULT_GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "num_sequences": 10000
}

# Helper pour créer des configs personnalisées
def create_config(**kwargs) -> SMILESConfig:
    """Crée une configuration personnalisée"""
    return SMILESConfig(**kwargs)

def create_training_config(**kwargs) -> SMILESConfig:
    """Crée une config optimisée pour l'entraînement"""
    config_dict = DEFAULT_TRAINING_CONFIG.copy()
    config_dict.update(kwargs)
    return SMILESConfig(**config_dict)

def create_generation_config(**kwargs) -> SMILESConfig:
    """Crée une config optimisée pour la génération"""
    config_dict = DEFAULT_GENERATION_CONFIG.copy()
    config_dict.update(kwargs)
    return SMILESConfig(**config_dict)

__all__.extend([
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_TRAINING_CONFIG", 
    "DEFAULT_GENERATION_CONFIG",
    "create_config",
    "create_training_config",
    "create_generation_config"
])