from transformers import Trainer

import torch
import torch.nn.functional as F

def Weighted_cross_entropy_loss(logits, labels):
    # Calculate standard cross-entropy loss first.
    ce_loss = F.cross_entropy(logits, labels, )
    
    
    
    # Compute focal loss.
    return ce_loss.mean()

class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_fn= Weighted_cross_entropy_loss, **kwargs):
        super().__init__(*args, **kwargs)
        # Store your custom loss function.
        # This should take (logits, labels) as arguments.
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        # Assume your inputs include "labels" and your model returns logits.
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute the custom loss using your loss function.
        loss = self.loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    