from transformers import Trainer

import torch
import torch.nn.functional as F

from torch import nn
    # ==================== OPTION 1: Classe Loss personnalisée ====================
class WeightedLanguageModelLoss(nn.Module):
    def __init__(self, vocab_weights=None, ignore_index=-100, label_smoothing=0.0,):
        """
        Loss function pour un language model avec poids personnalisés
        
        Args:
            vocab_weights: Tensor de poids pour chaque token du vocabulaire [vocab_size]
            ignore_index: Index à ignorer dans le calcul (padding tokens)
            label_smoothing: Lissage des labels (0.0 = pas de lissage)
        """
        super().__init__()
        self.vocab_weights = vocab_weights
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size] ou [batch_size*seq_len, vocab_size]
            labels: [batch_size, seq_len] ou [batch_size*seq_len]
        """
        # Si les logits ont 3 dimensions, on fait le shift pour LM
        if logits.dim() == 3:
            # Shift pour prédiction du token suivant
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Reshape pour cross_entropy
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
        else:
            shift_logits = logits
            shift_labels = labels
        
        # Calcul de la cross entropy
        loss = F.cross_entropy(
            shift_logits.float(),
            shift_labels,
            weight=self.vocab_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )
        
        return loss
    
    



class CustomWeightedTrainer(Trainer):
    def __init__(self, loss_function=None, *args, **kwargs):
        """
        Trainer personnalisé avec loss function custom
        
        Args:
            loss_function: Instance de votre WeightedCrossEntropyLoss
        """
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Surcharge de la méthode compute_loss pour utiliser votre loss personnalisée
        """
        # FIX: Use labels from inputs for language modeling
        labels = inputs.get("labels")
        if labels is None:
            # If no explicit labels, use input_ids shifted for causal LM
            labels = inputs.get("input_ids")

        # Forward pass du modèle
        outputs = model(**inputs)

        # Calcul de la loss avec votre fonction personnalisée
        if self.loss_function is not None and labels is not None:
            loss = self.loss_function(outputs.logits, labels)
        else:
            # Fallback sur la loss par défaut du modèle
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Si pas de loss par défaut, utiliser cross_entropy standard
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100  # Ignore padding tokens
                )

        return (loss, outputs) if return_outputs else loss