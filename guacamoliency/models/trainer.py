from transformers import (
    GPT2Config, GPT2LMHeadModel, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import os
from pathlib import Path

class SMILESTrainer(BaseSMILESModel):
    """Entraîneur pour modèles SMILES"""
    
    def __init__(self, config: SMILESConfig):
        super().__init__(config.to_dict())
        self.smiles_config = config
        self.preprocessor = SMILESPreprocessor(config)
        self.trainer = None
    
    def build_model(self):
        """Construction du modèle GPT2 pour SMILES"""
        # Charger le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.smiles_config.tokenizer_path)
        
        if self.smiles_config.use_scaffolds:
            self.tokenizer.model_max_length = 2 * self.tokenizer.model_max_length
        
        # Configuration du modèle
        model_config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_positions=self.tokenizer.model_max_length,
            n_ctx=self.tokenizer.model_max_length,
            n_embd=self.smiles_config.n_embd,
            n_layer=self.smiles_config.n_layer,
            n_head=self.smiles_config.n_head,
            resid_pdrop=self.smiles_config.resid_pdrop,
            embd_pdrop=self.smiles_config.embd_pdrop,
            attn_pdrop=self.smiles_config.attn_pdrop
        )
        
        # Créer le modèle
        self.model = GPT2LMHeadModel(model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        return self.model
    
    def prepare_data(self, data_path: Union[str, Path]) -> Dict[str, Any]:
        """Prépare les données d'entraînement"""
        df = pd.read_csv(data_path)
        datasets = self.preprocessor.prepare_datasets(df, self.tokenizer)
        return datasets
    
    def _create_output_directory(self) -> str:
        """Crée le répertoire de sortie avec versioning"""
        model_save_folder = f"{self.smiles_config.model_save_folder}_{self.smiles_config.tokenizer_type}"
        if self.smiles_config.use_scaffolds:
            model_save_folder += "_scaffolds"
        
        id_save = 1
        while os.path.isdir(os.getcwd() + "/" + model_save_folder + "/" + str(id_save)):
            id_save += 1
        
        final_path = model_save_folder + "/" + str(id_save)
        return final_path
    
    def train(self, data_path: Union[str, Path]):
        """Lance l'entraînement"""
        if not self.model:
            self.build_model()
        
        # Préparer les données
        datasets = self.prepare_data(data_path)
        
        # Créer le répertoire de sortie
        output_dir = self._create_output_directory()
        log_dir_end = f"{self.smiles_config.dataset_name}_{self.smiles_config.tokenizer_type}"
        if self.smiles_config.use_scaffolds:
            log_dir_end += "_scaffolds"
        log_dir_end += f"/{output_dir.split('/')[-1]}"
        
        # Arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.smiles_config.learning_rate,
            max_steps=self.smiles_config.max_steps,
            eval_strategy="steps",
            per_device_train_batch_size=self.smiles_config.batch_size,
            save_steps=self.smiles_config.save_steps,
            save_total_limit=self.smiles_config.save_total_limit,
            logging_dir=f"{self.smiles_config.log_dir}/logs/{log_dir_end}",
            report_to="tensorboard",
            logging_first_step=True,
            logging_strategy="steps",
            logging_steps=2000,
            warmup_steps=self.smiles_config.warmup_steps,
            dataloader_num_workers=self.smiles_config.num_workers,
            gradient_accumulation_steps=1,
            fp16=True,
            remove_unused_columns=False,
            lr_scheduler_type=self.smiles_config.lr_scheduler_type,
            lr_scheduler_kwargs={"min_lr": 0.1 * self.smiles_config.learning_rate},
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=0.1
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Créer le trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        print("Training start")
        self.trainer.train()
        
        # Sauvegarder le modèle final
        final_model_path = output_dir + "/final_model"
        self.trainer.save_model(final_model_path)
        
        print(f"Training completed. Model saved to: {final_model_path}")
        return final_model_path
