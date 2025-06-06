from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class SMILESConfig:
    """Configuration pour les modèles SMILES"""
    
    # Dataset config
    dataset_name: str = "moses_canonical"
    dataset_path: str = "data/training_data/moses_canonical.csv"
    
    # Tokenizer config
    tokenizer_path: str = "data/tokenizersBEP/moses_canonical"
    tokenizer_type: str = "BPE"
    
    # Model architecture
    n_embd: int = 256
    n_layer: int = 8
    n_head: int = 8
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    
    # Training config
    learning_rate: float = 6e-4
    max_steps: int = 41300
    batch_size: int = 384
    warmup_steps: int = 413
    save_steps: int = 5000
    save_total_limit: int = 5
    num_workers: int = 10
    lr_scheduler_type: str = "cosine_with_min_lr"
    
    # Generation config
    temperature: float = 1.0
    max_length: int = 120
    num_sequences: int = 1000
    top_k: int = 50
    top_p: float = 0.95
    
    # Paths
    model_save_folder: str = "models/trained"
    log_dir: str = "reports"
    output_dir: str = "data/generated"
    
    # Scaffolds specific
    use_scaffolds: bool = False
    scaffold_input: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)