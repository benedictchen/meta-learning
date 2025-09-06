"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Matching Networks Algorithm Implementation
=========================================

Advanced matching networks implementation with attention mechanisms,
bidirectional LSTM processing, and full-context embeddings for few-shot learning.
"""

# TODO: PHASE 3.2 - MATCHING NETWORKS ALGORITHM IMPLEMENTATION
# TODO: Create MatchingNetworks class extending nn.Module
# TODO: - Implement __init__ with encoder, attention mechanism, and LSTM components
# TODO: - Add support for both simple and full context embeddings
# TODO: - Include bidirectional LSTM for enhanced support set processing
# TODO: - Support attention-weighted query embedding based on support set
# TODO: - Add numerical stability for attention computation and cosine similarity

# TODO: Implement matching_networks_loss() function for end-to-end training
# TODO: - Use distance-based similarity between query and support embeddings
# TODO: - Support both cosine similarity and learned distance metrics
# TODO: - Add temperature scaling for similarity calibration
# TODO: - Include support for different aggregation methods (mean, weighted)
# TODO: - Add regularization options for attention weights

# TODO: Implement attention mechanism components
# TODO: - Create AttentionLSTM class for bidirectional processing
# TODO: - Add full_context_embeddings() for enhanced support set representation
# TODO: - Implement cosine_similarity() with numerical stability
# TODO: - Create attention_weights() function with softmax normalization
# TODO: - Add memory-efficient implementation for large support sets

# TODO: Add integration with existing meta-learning framework
# TODO: - Integrate with Episode data structure for few-shot tasks
# TODO: - Add to algorithm selector as attention-based option
# TODO: - Include in A/B testing framework for performance comparison
# TODO: - Connect with failure prediction for attention mechanism monitoring
# TODO: - Add to performance monitoring dashboard with attention metrics

# TODO: Implement advanced features
# TODO: - Add support for different attention mechanisms (additive, multiplicative)
# TODO: - Implement learnable temperature parameter for similarity scaling
# TODO: - Add support for multi-head attention for richer representations
# TODO: - Support hierarchical attention for multi-scale feature matching
# TODO: - Include uncertainty quantification for attention-based predictions

# TODO: Add Phase 4 ML-powered enhancements integration
# TODO: - Connect with AlgorithmSelector for automatic matching networks usage
# TODO: - Integrate with ABTestingFramework for attention mechanism comparison
# TODO: - Add failure prediction hooks for attention computation failures
# TODO: - Support performance monitoring for real-time attention optimization
# TODO: - Add cross-task knowledge transfer for attention weight initialization

# TODO: Add comprehensive testing and validation
# TODO: - Test mathematical correctness against original Matching Networks paper
# TODO: - Validate attention mechanisms with various support set sizes
# TODO: - Test numerical stability with extreme similarity values
# TODO: - Benchmark performance against MAML and prototypical networks
# TODO: - Add regression tests for attention weight distributions

from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from ..core.episode import Episode


class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with attention mechanism for support set processing."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.hidden_size = hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence with bidirectional LSTM.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Bidirectional LSTM output [batch_size, seq_len, 2*hidden_size]
        """
        output, (h_n, c_n) = self.lstm(x)
        return output


class MatchingNetworks(nn.Module):
    """
    Matching Networks implementation for few-shot learning with attention mechanisms.
    
    Based on "Matching Networks for One Shot Learning" by Vinyals et al. (2016).
    Implements full context embeddings with bidirectional LSTM and attention.
    """
    
    def __init__(
        self, 
        encoder: nn.Module,
        use_full_context: bool = True,
        use_lstm: bool = True,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 1,
        attention_type: str = "cosine",
        temperature: float = 1.0
    ):
        """
        Initialize Matching Networks model.
        
        Args:
            encoder: Feature encoder (e.g., CNN backbone)
            use_full_context: Whether to use full context embeddings
            use_lstm: Whether to use bidirectional LSTM for support processing
            lstm_hidden_size: Hidden size for LSTM layers
            num_lstm_layers: Number of LSTM layers
            attention_type: Type of attention ('cosine', 'learned')
            temperature: Temperature scaling for similarities
        """
        super().__init__()
        self.encoder = encoder
        self.use_full_context = use_full_context
        self.use_lstm = use_lstm
        self.attention_type = attention_type
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, *self._get_input_shape())
            encoder_output = encoder(dummy_input)
            self.encoder_dim = encoder_output.shape[-1]
        
        # Initialize LSTM components if needed
        if use_lstm:
            self.support_lstm = AttentionLSTM(
                input_size=self.encoder_dim,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers
            )
            
            self.query_lstm = AttentionLSTM(
                input_size=self.encoder_dim,
                hidden_size=lstm_hidden_size, 
                num_layers=num_lstm_layers
            )
            
            # Update embedding dimension after LSTM
            self.final_dim = lstm_hidden_size * 2  # Bidirectional
        else:
            self.final_dim = self.encoder_dim
            
        # Learned attention mechanism
        if attention_type == "learned":
            self.attention_mlp = nn.Sequential(
                nn.Linear(self.final_dim * 2, self.final_dim),
                nn.ReLU(),
                nn.Linear(self.final_dim, 1)
            )
    
    def _get_input_shape(self):
        """Infer input shape from encoder."""
        if hasattr(self.encoder, 'input_shape'):
            return self.encoder.input_shape
        
        # Try to infer from encoder structure
        if hasattr(self.encoder, 'in_features'):
            # Linear layer - return as 1D feature vector
            return (self.encoder.in_features,)
        elif isinstance(self.encoder, nn.Sequential) and len(self.encoder) > 0:
            # Sequential network - check first layer
            first_layer = self.encoder[0]
            if hasattr(first_layer, 'in_features'):
                return (first_layer.in_features,)
            elif hasattr(first_layer, 'in_channels'):
                # Conv layer - return default image shape
                return (first_layer.in_channels, 84, 84)
        
        # Default shape for common vision encoders
        return (3, 84, 84)
    
    def encode_support(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        Encode support set with full context embeddings.
        
        Args:
            support_data: Support set data [n_support, *input_shape]
            support_labels: Support set labels [n_support]
            
        Returns:
            Support embeddings [n_support, embedding_dim]
        """
        # Basic encoding
        support_embeddings = self.encoder(support_data)  # [n_support, encoder_dim]
        
        if not self.use_full_context:
            return support_embeddings
            
        # Full context embeddings with LSTM
        if self.use_lstm:
            # Process through bidirectional LSTM
            # Add batch dimension for LSTM
            lstm_input = support_embeddings.unsqueeze(0)  # [1, n_support, encoder_dim]
            lstm_output = self.support_lstm(lstm_input)  # [1, n_support, 2*hidden_size]
            support_embeddings = lstm_output.squeeze(0)  # [n_support, 2*hidden_size]
            
        return support_embeddings
    
    def encode_query(self, query_data: torch.Tensor, support_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode query set with attention over support set.
        
        Args:
            query_data: Query set data [n_query, *input_shape]  
            support_embeddings: Support embeddings [n_support, embedding_dim]
            
        Returns:
            Query embeddings [n_query, embedding_dim]
        """
        # Basic encoding
        query_embeddings = self.encoder(query_data)  # [n_query, encoder_dim]
        
        if not self.use_full_context:
            return query_embeddings
            
        # Full context embeddings with attention-weighted LSTM
        if self.use_lstm:
            # Process each query with attention to support set
            enhanced_queries = []
            
            for i, query_emb in enumerate(query_embeddings):
                # Compute attention weights between query and all support examples
                if self.attention_type == "cosine":
                    similarities = F.cosine_similarity(
                        query_emb.unsqueeze(0).expand_as(support_embeddings),
                        support_embeddings,
                        dim=1
                    )
                elif self.attention_type == "learned":
                    # Concatenate query with each support example
                    query_expanded = query_emb.unsqueeze(0).expand(support_embeddings.size(0), -1)
                    concat_features = torch.cat([query_expanded, support_embeddings], dim=1)
                    similarities = self.attention_mlp(concat_features).squeeze(1)
                else:
                    raise ValueError(f"Unknown attention type: {self.attention_type}")
                
                attention_weights = F.softmax(similarities / self.temperature, dim=0)
                
                # Attention-weighted support context
                attended_context = torch.sum(
                    attention_weights.unsqueeze(1) * support_embeddings, 
                    dim=0
                )
                
                # Combine with query embedding
                if self.use_lstm:
                    # Process combined embedding through LSTM
                    combined = query_emb + attended_context
                    lstm_input = combined.unsqueeze(0).unsqueeze(0)  # [1, 1, encoder_dim]
                    lstm_output = self.query_lstm(lstm_input)  # [1, 1, 2*hidden_size]
                    enhanced_query = lstm_output.squeeze(0).squeeze(0)  # [2*hidden_size]
                else:
                    enhanced_query = query_emb + attended_context
                
                enhanced_queries.append(enhanced_query)
            
            query_embeddings = torch.stack(enhanced_queries)
            
        return query_embeddings
    
    def compute_similarities(self, query_embeddings: torch.Tensor, 
                           support_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarities between query and support embeddings.
        
        Args:
            query_embeddings: Query embeddings [n_query, embedding_dim]
            support_embeddings: Support embeddings [n_support, embedding_dim]
            
        Returns:
            Similarity matrix [n_query, n_support]
        """
        if self.attention_type == "cosine":
            # Normalize embeddings for cosine similarity
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            support_norm = F.normalize(support_embeddings, p=2, dim=1)
            
            # Compute cosine similarity matrix
            similarities = torch.mm(query_norm, support_norm.t())  # [n_query, n_support]
            
        elif self.attention_type == "learned":
            # Use learned similarity function
            n_query, n_support = query_embeddings.size(0), support_embeddings.size(0)
            similarities = torch.zeros(n_query, n_support, device=query_embeddings.device)
            
            for i in range(n_query):
                for j in range(n_support):
                    concat_features = torch.cat([query_embeddings[i], support_embeddings[j]])
                    similarity = self.attention_mlp(concat_features)
                    similarities[i, j] = similarity.squeeze()
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        return similarities
    
    def forward(self, episode: Episode) -> torch.Tensor:
        """
        Forward pass computing predictions for query set.
        
        Args:
            episode: Few-shot learning episode
            
        Returns:
            Logits for query predictions [n_query, n_classes]
        """
        # Extract data from episode
        support_data = episode.support_x
        support_labels = episode.support_y
        query_data = episode.query_x
        
        # Encode support and query sets
        support_embeddings = self.encode_support(support_data, support_labels)
        query_embeddings = self.encode_query(query_data, support_embeddings)
        
        # Compute similarity matrix
        similarities = self.compute_similarities(query_embeddings, support_embeddings)
        
        # Convert similarities to class predictions using support labels
        n_classes = len(torch.unique(support_labels))
        logits = torch.zeros(query_embeddings.size(0), n_classes, device=query_data.device)
        
        # Aggregate similarities by class
        for class_id in range(n_classes):
            class_mask = (support_labels == class_id)
            if class_mask.sum() > 0:
                # Average similarity to all support examples of this class
                class_similarities = similarities[:, class_mask]
                logits[:, class_id] = class_similarities.mean(dim=1)
        
        return logits
    
    def predict(self, episode: Episode) -> torch.Tensor:
        """
        Make predictions for episode query set.
        
        Args:
            episode: Few-shot learning episode
            
        Returns:
            Predicted class probabilities [n_query, n_classes]
        """
        with torch.no_grad():
            logits = self.forward(episode)
            return F.softmax(logits, dim=1)


def matching_networks_loss(model: MatchingNetworks, episode: Episode, 
                          reduction: str = "mean") -> torch.Tensor:
    """
    Compute Matching Networks loss for few-shot learning episode.
    
    Args:
        model: Matching Networks model
        episode: Few-shot learning episode
        reduction: Loss reduction method ('mean', 'sum', 'none')
        
    Returns:
        Cross-entropy loss between predictions and query labels
    """
    logits = model(episode)
    query_labels = episode.query_y
    
    loss = F.cross_entropy(logits, query_labels, reduction=reduction)
    return loss


def create_matching_networks(encoder: nn.Module, 
                           use_full_context: bool = True,
                           use_lstm: bool = True,
                           lstm_hidden_size: int = 256,
                           attention_type: str = "cosine") -> MatchingNetworks:
    """
    Factory function to create Matching Networks model with standard configuration.
    
    Args:
        encoder: Feature encoder network
        use_full_context: Whether to use full context embeddings
        use_lstm: Whether to use bidirectional LSTM
        lstm_hidden_size: Hidden size for LSTM layers
        attention_type: Type of attention mechanism
        
    Returns:
        Configured Matching Networks model
    """
    return MatchingNetworks(
        encoder=encoder,
        use_full_context=use_full_context,
        use_lstm=use_lstm,
        lstm_hidden_size=lstm_hidden_size,
        attention_type=attention_type
    )


def cosine_similarity_stable(x: torch.Tensor, y: torch.Tensor, 
                           eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable cosine similarity computation.
    
    Args:
        x: First tensor [n, d]
        y: Second tensor [m, d]  
        eps: Small value for numerical stability
        
    Returns:
        Cosine similarity matrix [n, m]
    """
    # Normalize vectors
    x_norm = F.normalize(x, p=2, dim=1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=1, eps=eps)
    
    # Compute cosine similarity
    similarity = torch.mm(x_norm, y_norm.t())
    
    # Clamp to avoid numerical issues
    similarity = torch.clamp(similarity, min=-1.0 + eps, max=1.0 - eps)
    
    return similarity


def attention_weights_softmax(similarities: torch.Tensor, 
                            temperature: float = 1.0,
                            dim: int = -1) -> torch.Tensor:
    """
    Compute attention weights with temperature-scaled softmax.
    
    Args:
        similarities: Similarity scores
        temperature: Temperature for scaling
        dim: Dimension for softmax
        
    Returns:
        Normalized attention weights
    """
    scaled_similarities = similarities / temperature
    attention_weights = F.softmax(scaled_similarities, dim=dim)
    
    return attention_weights


# Utility function for testing and validation
def validate_matching_networks(model: MatchingNetworks, episode: Episode) -> Dict[str, Any]:
    """
    Validate Matching Networks model on episode and return diagnostic information.
    
    Args:
        model: Matching Networks model
        episode: Test episode
        
    Returns:
        Dictionary with validation metrics and diagnostics
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits = model(episode)
        predictions = torch.argmax(logits, dim=1)
        
        # Compute accuracy
        correct = (predictions == episode.query_y).float()
        accuracy = correct.mean().item()
        
        # Compute support and query embeddings for analysis
        support_embeddings = model.encode_support(episode.support_x, episode.support_y)
        query_embeddings = model.encode_query(episode.query_x, support_embeddings)
        
        # Attention analysis (if using attention)
        similarities = model.compute_similarities(query_embeddings, support_embeddings)
        
        # Diagnostic information
        diagnostics = {
            'accuracy': accuracy,
            'n_correct': correct.sum().item(),
            'n_total': len(episode.query_y),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'similarities_mean': similarities.mean().item(),
            'similarities_std': similarities.std().item(),
            'support_embedding_norm': support_embeddings.norm(dim=1).mean().item(),
            'query_embedding_norm': query_embeddings.norm(dim=1).mean().item(),
            'temperature': model.temperature.item() if hasattr(model.temperature, 'item') else model.temperature
        }
        
        # Check for potential issues
        if torch.isnan(logits).any():
            diagnostics['warning'] = "NaN values detected in logits"
        elif torch.isinf(logits).any():
            diagnostics['warning'] = "Infinite values detected in logits"
        elif similarities.max() > 10:
            diagnostics['warning'] = "Very large similarity values detected"
        
    return diagnostics