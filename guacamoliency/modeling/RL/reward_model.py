from torch import nn
import torch
import numpy as np

class CustomRewardModel(nn.Module):
    def __init__(self, scoring_function, tokenizer):
        super().__init__()
        self.scoring_function = scoring_function
        self.tokenizer = tokenizer
    
    def forward(self, input_ids, attention_mask=None):
        # Decode the generated text
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Apply your scoring function
        scores = []
        for text in texts:
            try:
                score = self.scoring_function(text)
                scores.append(float(score))
            except Exception as e:
                print(f"Error scoring text: {e}")
                scores.append(0.0)
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float).unsqueeze(-1)
    

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, Autotokenizer
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import QED



def reward_QED1(smile, target):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        score = QED.qed(mol)
        if abs(score-target)< 0.05:
            return 1
        return 0
    return 0

class RewardDataset(Dataset):
    def __init__(self, texts, rewards, tokenizer):
        self.texts = texts
        self.rewards = rewards
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        reward = self.rewards[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(reward, dtype=torch.float)
        }

# Generate training data using your reward function
def create_reward_training_data(texts, reward_function):
    rewards = [reward_function(text) for text in texts]
    return texts, rewards

# Train the reward model
def train_reward_model(texts, base_model_name, tokenizer_dir,reward_function):
    tokenizer = Autotokenizer.from_pretrained(tokenizer_dir)
    texts, rewards = create_reward_training_data(texts, reward_function)
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=1
    )
    
    train_dataset = RewardDataset(texts, rewards, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='reward_model',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        logging_steps=100,
         )
    
    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        
    )

  
  
  
    
    trainer.train()
    return reward_model


