import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AARTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2
        self.contrastive_alpha = config.contrastive_alpha
        
        # Initialize weights with smaller values
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.01)
        
        # Move to device
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        
        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
        
    def compute_contrastive_loss(self, annotator_embeds):
        # Compute cosine similarity matrix
        norm_embeds = F.normalize(annotator_embeds, p=2, dim=1)
        similarity = torch.mm(norm_embeds, norm_embeds.t())
        
        # Create labels for contrastive loss (1 for same annotator, 0 for different)
        labels = torch.eye(similarity.size(0), device=similarity.device)
        
        # Compute loss with numerical stability
        similarity = similarity.clamp(-1 + 1e-7, 1 - 1e-7)  # Prevent exact ±1
        contrastive_loss = -(labels * torch.log(1 + similarity + 1e-7) + 
                            (1 - labels) * torch.log(1 - similarity + 1e-7)).mean()
        
        return contrastive_loss
        
    def forward(self, input_ids, attention_mask, annotator_id, label=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get annotator embeddings
        annotator_embeds = self.annotator_embeddings(annotator_id)
        
        # Combine text and annotator representations with scaling
        combined = pooled_output * annotator_embeds * 0.1  # Scale down the multiplication
        logits = self.classifier(combined)
        
        if label is not None:
            # Classification loss with stability
            loss_fct = nn.BCEWithLogitsLoss()
            cls_loss = loss_fct(logits.view(-1), label.float().view(-1))
            
            # Contrastive loss for annotator embeddings
            contra_loss = self.compute_contrastive_loss(annotator_embeds)
            
            # Total loss with scaled components
            loss = cls_loss + self.lambda2 * contra_loss
            
            # Check for NaN and clip if necessary
            if torch.isnan(loss):
                print(f"NaN detected! cls_loss: {cls_loss}, contra_loss: {contra_loss}")
                return torch.tensor(0.0, requires_grad=True, device=self.device)
                
            return loss
            
        return logits