"""
Module models : entraînement et génération de molécules SMILES
"""

from .trainer import SMILESTrainer
from .generator import SMILESGenerator

__all__ = [
    "SMILESTrainer",
    "SMILESGenerator"
]

# Registry des types de modèles supportés
MODEL_TYPES = {
    "gpt2": "GPT2LMHeadModel",
    "standard": "GPT2LMHeadModel"
}

# Registry des types de tokenizers supportés
TOKENIZER_TYPES = {
    "BPE": "Byte-Pair Encoding",
    "WordPiece": "WordPiece Tokenization",
    "SentencePiece": "SentencePiece Tokenization"
}

# Factory functions pour créer des modèles
def create_trainer(config_dict: dict = None, use_scaffolds: bool = False):
    """Factory function pour créer un trainer"""
    from ..core.config import SMILESConfig
    
    if config_dict is None:
        config_dict = {}
    
    config_dict["use_scaffolds"] = use_scaffolds
    config = SMILESConfig(**config_dict)
    return SMILESTrainer(config)

def create_generator(model_path: str, config_dict: dict = None):
    """Factory function pour créer un générateur"""
    from ..core.config import SMILESConfig
    
    if config_dict is None:
        config_dict = {}
    
    config = SMILESConfig(**config_dict)
    return SMILESGenerator(model_path, config)

def create_scaffolds_trainer(config_dict: dict = None):
    """Factory function pour créer un trainer avec scaffolds"""
    return create_trainer(config_dict, use_scaffolds=True)

# Configurations prédéfinies

PRESET_CONFIGS = {
    "small": {
        "n_embd": 256,
        "n_layer": 8,
        "n_head": 8,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1
    },
    "medium": {
        "n_embd": 512,
        "n_layer": 12,
        "n_head": 12,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1
    }
}