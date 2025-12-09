import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class AARTRankingRobustModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.backbone.config.hidden_size
        self.annotator_embeddings = nn.Embedding(config.num_annotators, hidden_size)
        self.lambda2 = config.lambda2 if config.lambda2 is not None else 0.1
        
        # Ranking Robust InfoNCE hyperparameters
        self.num_classes = getattr(config, "num_classes", 5)  # 0-4 Likert scale
        
        # Per-rank robustness parameters (q values)
        # More uniform q values for uniform noise - less trust in any specific distance
        default_q_values = [0.6, 0.65, 0.7, 0.75, 0.8][:self.num_classes]
        self.q_values = getattr(config, "q_values", default_q_values)
        
        # Per-rank temperature values  
        # More uniform tau values for uniform noise - similar temperature across ranks
        default_tau_values = [0.1, 0.12, 0.14, 0.16, 0.18][:self.num_classes]
        self.tau_values = getattr(config, "tau_values", default_tau_values)
        
        # Per-rank lambda values for RRINCE denominator term
        # Higher lambda for higher ranks (more distant labels) to emphasize separation
        default_lambda_values = [0.5, 0.5, 0.5, 0.5, 0.5][:self.num_classes]
        self.lambda_values = getattr(config, "lambda_values", default_lambda_values)
        
        # Optional: make q, tau, and lambda learnable
        self.learnable_params = getattr(config, "learnable_ranking_params", False)
        if self.learnable_params:
            self.q_params = nn.Parameter(torch.tensor(self.q_values))
            self.tau_params = nn.Parameter(torch.tensor(self.tau_values))
            self.lambda_params = nn.Parameter(torch.tensor(self.lambda_values))
        else:
            self.register_buffer('q_params', torch.tensor(self.q_values))
            self.register_buffer('tau_params', torch.tensor(self.tau_values))
            self.register_buffer('lambda_params', torch.tensor(self.lambda_values))
        
        # Rank weighting (optional - weight importance of each rank's loss)
        default_rank_weights = [1.0] * self.num_classes
        self.rank_weights = torch.tensor(getattr(config, "rank_weights", default_rank_weights))
        self.register_buffer('rank_weights_buffer', self.rank_weights)
        
        # Initialize embeddings
        nn.init.normal_(self.annotator_embeddings.weight, mean=0.0, std=0.1)
        
        # Move to device
        self.annotator_embeddings = self.annotator_embeddings.to(self.device)
        
        # Handle multi-GPU if needed
        if config.n_gpu > 1:
            self.annotator_embeddings = nn.DataParallel(self.annotator_embeddings)
    
    def compute_ranking_robust_loss(self, annotator_embeds, labels, text_ids):
        """
        Ranking Robust InfoNCE loss for ordinal labels.
        
        Args:
            annotator_embeds: [B, D] - annotator embeddings
            labels: [B] - labels (0-4 for 5-point Likert)
            text_ids: [B] - text instance IDs
            
        Returns:
            Scalar loss tensor
        """
        device = annotator_embeds.device
        B = annotator_embeds.size(0)
        
        if B == 0:
            return torch.tensor(0.0, device=device)
        
        # Normalize embeddings for cosine similarity
        normed_embeds = F.normalize(annotator_embeds, dim=-1)  # [B, D]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(normed_embeds, normed_embeds.T)  # [B, B]
        
        # Create masks
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        same_text = text_ids.unsqueeze(0) == text_ids.unsqueeze(1)  # [B, B]
        
        # Compute label distances (0 to 4 for 5-point scale)
        label_distances = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))  # [B, B]
        
        # Mask out self-similarities and different texts
        valid_mask = same_text & eye_mask  # [B, B]
        
        total_loss = 0.0
        loss_components = {}  # For debugging/monitoring
        
        # Process each rank (distance level)
        for distance in range(self.num_classes):
            # Get q, tau, and lambda for this rank
            q = self.q_params[distance] if distance < len(self.q_params) else self.q_params[-1]
            tau = self.tau_params[distance] if distance < len(self.tau_params) else self.tau_params[-1]
            lambda_r = self.lambda_params[distance] if distance < len(self.lambda_params) else self.lambda_params[-1]
            rank_weight = self.rank_weights_buffer[distance] if distance < len(self.rank_weights_buffer) else 1.0
            
            # Masks for current rank
            rank_mask = (label_distances == distance) & valid_mask  # Positives at this distance
            
            # For denominator: all samples at distance >= current distance
            denom_mask = (label_distances >= distance) & valid_mask
            
            # Skip if no positive pairs at this distance
            if not rank_mask.any():
                loss_components[f'rank_{distance}'] = 0.0
                continue
            
            # Compute loss for this rank using robust formulation (RRINCE)
            rank_loss = self.compute_robust_infonce_for_rank(
                sim_matrix, rank_mask, denom_mask, q, tau, lambda_r, device
            )
            
            # Weight the loss for this rank
            weighted_loss = rank_weight * rank_loss
            total_loss += weighted_loss
            
            # Store for monitoring
            loss_components[f'rank_{distance}'] = rank_loss.item()
        
        # Optional: Print loss components periodically for debugging
        if hasattr(self, 'batch_count') and self.batch_count % 100 == 0:
            print(f"Ranking loss components: {loss_components}")
        
        # Return total loss (no normalization - sum of weighted rank losses as per paper)
        return total_loss
    
    def compute_robust_infonce_for_rank(self, sim_matrix, numerator_mask, denominator_mask, q, tau, lambda_r, device):
        """
        RRINCE: Ranking Robust InfoNCE
        
        For each rank r, apply RINCE formula:
        L_r = -(1/q)*sum(exp(q*s_pos/tau)) + (lambda_r/q)*(sum(exp(s_all/tau)))^q
        
        This preserves the robustness property of RINCE while enforcing ranking.
        The negative first term encourages high similarity for positives (minimize loss = maximize similarity).
        The positive second term with lambda_r penalizes all similarities in denominator.
        
        Args:
            sim_matrix: [B, B] similarity matrix (cosine similarities)
            numerator_mask: [B, B] mask for positive pairs at this rank
            denominator_mask: [B, B] mask for all pairs in denominator (current and higher ranks)
            q: robustness parameter for this rank
            tau: temperature parameter for this rank
            lambda_r: lambda parameter for this rank (controls denominator term strength)
            device: torch device
            
        Returns:
            Scalar loss for this rank
        """
        B = sim_matrix.size(0)
        scaled_sims = sim_matrix / tau
        
        losses = []
        
        for i in range(B):
            pos_mask_i = numerator_mask[i]
            if not pos_mask_i.any():
                continue
            
            pos_sims = scaled_sims[i][pos_mask_i]
            
            denom_mask_i = denominator_mask[i]
            if not denom_mask_i.any():
                continue
            
            denom_sims = scaled_sims[i][denom_mask_i]
            
            # First term: -(1/q) * sum(exp(q * s_pos/tau))
            # Using logsumexp for numerical stability
            q_pos_sims = torch.clamp(q * pos_sims, max=20.0)
            first_term = -torch.exp(torch.logsumexp(q_pos_sims, dim=0)) / q
            
            # Second term: (lambda_r/q) * (sum(exp(s_all/tau)))^q
            # Using logsumexp for numerical stability, then raise to power q
            denom_logsumexp = torch.logsumexp(torch.clamp(denom_sims, max=20.0), dim=0)
            second_term = (lambda_r / q) * torch.exp(q * denom_logsumexp)
            
            loss_i = first_term + second_term
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()
    
    def forward(self, input_ids, attention_mask, annotator_id, label=None, text_id=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get annotator embeddings
        annotator_embeds = self.annotator_embeddings(annotator_id)
        
        # Combine text and annotator representations
        combined = pooled_output + annotator_embeds
        
        # Apply LayerNorm
        combined = F.layer_norm(combined, combined.size()[1:])
        
        logits = self.classifier(combined)
        
        if label is not None:
            # Classification loss for multiclass
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(logits, label.long())
            
            # Ranking Robust contrastive loss for annotator embeddings
            ranking_loss = self.compute_ranking_robust_loss(
                annotator_embeds=self.annotator_embeddings(annotator_id),
                labels=label,
                text_ids=text_id
            )
            
            # Track batch count for debugging
            if hasattr(self, 'batch_count'):
                self.batch_count += 1
            else:
                self.batch_count = 0
            
            if self.batch_count % 100 == 0:
                print(f"Batch {self.batch_count} - Ranking Loss: {ranking_loss.item():.4f}, "
                      f"Classification Loss: {cls_loss.item():.4f}")
            
            # Total loss
            loss = cls_loss + self.lambda2 * ranking_loss
            
            return loss
            
        return logits

    def get_config_dict(self):
        """Return configuration for saving/loading"""
        return {
            'num_classes': self.num_classes,
            'q_values': self.q_values,
            'tau_values': self.tau_values,
            'lambda_values': self.lambda_values.tolist() if isinstance(self.lambda_values, torch.Tensor) else self.lambda_values,
            'rank_weights': self.rank_weights.tolist(),
            'learnable_params': self.learnable_params,
            'lambda2': self.lambda2
        }