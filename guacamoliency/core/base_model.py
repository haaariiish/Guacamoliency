from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class BaseSMILESModel(ABC):
    """Classe de base pour tous les modèles SMILES"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_device()
    
    def _setup_device(self):
        """Configuration du device avec vérification CUDA"""
        if not torch.cuda.is_available() and self.config.get('force_cuda', True):
            raise Exception("Install correctly CUDA or check your drivers")
        print(f"Using device: {self.device}")
    
    @abstractmethod
    def build_model(self):
        """Construction du modèle"""
        pass
    
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Préparation des données"""
        pass
    
    def save_model(self, path: Path, save_tokenizer: bool = True):
        """Sauvegarde du modèle et tokenizer"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model:
            self.model.save_pretrained(path)
        
        if self.tokenizer and save_tokenizer:
            self.tokenizer.save_pretrained(path)
        
        # Sauvegarder la config
        with open(path / "model_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
    
    def load_model(self, path: Path):
        """Chargement du modèle et tokenizer"""
        path = Path(path)
        
        # Charger la config si elle existe
        config_path = path / "model_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                saved_config = json.load(f)
                self.config.update(saved_config)
        
        # Charger tokenizer et modèle
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()