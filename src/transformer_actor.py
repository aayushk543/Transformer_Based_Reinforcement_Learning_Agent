"""
Transformer-based actor for TD3.
Uses causal self-attention over sliding window of observation-action pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TransformerActor(nn.Module):
    """Causal Transformer actor for TD3."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, window_size: int = 8,
                 max_action: float = 1.0, dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.max_action = max_action
        
        # Input embedding: obs + action -> hidden_dim
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Embedding(2 * window_size, hidden_dim)
        
        # Transformer layers with pre-LN
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs: torch.Tensor, prev_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: (batch, window_size, obs_dim) - sliding window of observations
            prev_actions: (batch, window_size-1, action_dim) - previous actions
                         If None, use zero actions
        
        Returns:
            action: (batch, action_dim) - action for current observation
        """
        batch_size, window_size, _ = obs.shape
        
        if prev_actions is None:
            prev_actions = torch.zeros(batch_size, window_size - 1, self.action_dim,
                                      device=obs.device)
        
        # Embed observations
        obs_embed = self.obs_embed(obs)  # (batch, window, hidden)
        
        # Embed previous actions and pad
        action_embed = self.action_embed(prev_actions)  # (batch, window-1, hidden)
        action_embed = F.pad(action_embed, (0, 0, 1, 0))  # (batch, window, hidden)
        
        # Combine obs and action embeddings
        x = obs_embed + action_embed  # (batch, window, hidden)
        
        # Add positional encoding
        positions = torch.arange(window_size, device=obs.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        
        # Causal mask: allow attention to current and past, not future
        causal_mask = torch.triu(torch.ones(window_size, window_size, device=obs.device) * float('-inf'), 
                                 diagonal=1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask)
        
        # Use final timestep representation for action prediction
        x = x[:, -1, :]  # (batch, hidden)
        
        action = self.output_head(x)  # (batch, action_dim)
        return self.max_action * action


class TransformerCritic(nn.Module):
    """Transformer-based critic for TD3."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, window_size: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        # Input embedding
        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Embedding(2 * window_size, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Q1 and Q2 heads
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor, 
                prev_actions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (batch, window_size, obs_dim)
            actions: (batch, window_size, action_dim) - actions including current
            prev_actions: (batch, window_size-1, action_dim) - previous actions for embedding
        
        Returns:
            q1, q2: (batch, 1) - Q-values
        """
        batch_size, window_size, _ = obs.shape
        
        if prev_actions is None:
            prev_actions = torch.zeros(batch_size, window_size - 1, self.action_dim,
                                      device=obs.device)
        
        # Embed observations
        obs_embed = self.obs_embed(obs)
        
        # Embed all actions
        action_embed = self.action_embed(actions)
        
        # Interleave or add embeddings
        x = obs_embed + action_embed[:, :window_size, :]
        
        # Add positional encoding
        positions = torch.arange(window_size, device=obs.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(window_size, window_size, device=obs.device) * float('-inf'),
                                 diagonal=1)
        
        # Apply transformer
        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask)
        
        # Use final timestep
        x = x[:, -1, :]
        
        q1 = self.q1_head(x)
        q2 = self.q2_head(x)
        
        return q1, q2
