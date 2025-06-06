class SMILESGenerator(BaseSMILESModel):
    """Générateur de molécules SMILES"""
    
    def __init__(self, model_path: Union[str, Path], config: Optional[SMILESConfig] = None):
        if config is None:
            config = SMILESConfig()
        
        super().__init__(config.to_dict())
        self.smiles_config = config
        self.model_path = Path(model_path)
        self.load_model(self.model_path)
    
    def build_model(self):
        """Le modèle est chargé via load_model"""
        pass
    
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Non utilisé pour la génération"""
        pass
    
    def _merge_tokens(self, token_list: List[str]) -> str:
        """Fusionne une liste de tokens en string"""
        return "".join(token_list)
    
    def generate_from_scratch(self, num_sequences: int = None) -> List[str]:
        """Génère des SMILES à partir de rien"""
        if num_sequences is None:
            num_sequences = self.smiles_config.num_sequences
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                max_length=self.tokenizer.model_max_length,
                num_return_sequences=num_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.smiles_config.temperature,
                top_k=self.smiles_config.top_k,
                top_p=self.smiles_config.top_p,
                return_dict_in_generate=True,
            )
        
        # Décoder les séquences
        generated_smiles = [
            self.tokenizer.decode(output, skip_special_tokens=True).replace(" ", "")
            for output in generated_ids['sequences']
        ]
        
        return generated_smiles
    
    def generate_from_scaffold(self, scaffold: str, num_sequences: int = None) -> List[str]:
        """Génère des SMILES à partir d'un scaffold"""
        if num_sequences is None:
            num_sequences = self.smiles_config.num_sequences
        
        # Tokeniser le scaffold
        tokens = self.tokenizer(scaffold, return_tensors="pt")
        
        # Configuration de génération
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        
        outputs_list = []
        
        for _ in range(num_sequences):
            with torch.no_grad():
                outputs = self.model.generate(
                    tokens["input_ids"],
                    max_new_tokens=self.tokenizer.model_max_length // 2 + 1,
                    do_sample=True,
                    top_k=self.smiles_config.top_k,
                    top_p=self.smiles_config.top_p,
                    temperature=self.smiles_config.temperature,
                    attention_mask=tokens["attention_mask"]
                )
            
            # Extraire uniquement la partie générée (pas le prompt)
            generated_part = outputs[0][len(tokens["input_ids"][0]):len(outputs[0])-1]
            decoded = self.tokenizer.batch_decode(generated_part, skip_special_tokens=False)
            outputs_list.append(self._merge_tokens(decoded))
        
        return outputs_list
    
    def generate_batch(self, scaffolds: Optional[List[str]] = None, num_sequences_per_scaffold: int = None) -> pd.DataFrame:
        """Génère un batch de molécules"""
        if scaffolds is None:
            # Génération from scratch
            generated_smiles = self.generate_from_scratch(num_sequences_per_scaffold)
            df = pd.DataFrame({"SMILES": generated_smiles})
        else:
            # Génération à partir de scaffolds
            all_smiles = []
            all_scaffolds = []
            
            for scaffold in scaffolds:
                generated = self.generate_from_scaffold(scaffold, num_sequences_per_scaffold)
                all_smiles.extend(generated)
                all_scaffolds.extend([scaffold] * len(generated))
            
            df = pd.DataFrame({
                "SMILES": all_smiles,
                "SCAFFOLDS": all_scaffolds
            })
        
        return df
    
    def save_generated(self, molecules_df: pd.DataFrame, output_path: Union[str, Path]):
        """Sauvegarde les molécules générées"""
        molecules_df.to_csv(output_path, index=False)
        print(f"Generated molecules saved to: {output_path}")