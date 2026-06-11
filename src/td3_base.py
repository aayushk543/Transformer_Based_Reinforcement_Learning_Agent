"""
Base TD3 implementation with MLP actor and critic.
Twin Delayed DDPG (Fujimoto et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from collections import deque


class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""
    
    def __init__(self, max_size: int = 1e6, obs_dim: int = 17, action_dim: int = 3):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.observations = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.next_observations = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add transition to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor]:
        """Sample random batch from buffer."""
        indices = np.random.randint(0, self.size, batch_size)
        
        obs = torch.FloatTensor(self.observations[indices])
        actions = torch.FloatTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1)
        next_obs = torch.FloatTensor(self.next_observations[indices])
        dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        
        return obs, actions, rewards, next_obs, dones


class MLPActor(nn.Module):
    """MLP actor network for TD3."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, 
                 max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(obs)


class MLPCritic(nn.Module):
    """Dual MLP critic network for TD3 (Q1 and Q2)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class TD3Agent:
    """TD3 (Twin Delayed DDPG) agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, max_action: float = 1.0,
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4, 
                 hidden_dim: int = 256, tau: float = 0.005, policy_delay: int = 2,
                 device: str = "cpu"):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.tau = tau
        self.policy_delay = policy_delay
        self.device = torch.device(device)
        self.total_steps = 0
        
        # Actor networks
        self.actor = MLPActor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target = MLPActor(obs_dim, action_dim, hidden_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic networks
        self.critic = MLPCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = MLPCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using actor network."""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        
        # Add exploration noise during training
        if training:
            noise = np.random.normal(0, 0.1, self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update(self, batch_size: int, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """Update actor and critic networks."""
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            noise = torch.randn_like(next_actions) * 0.2
            noise = torch.clamp(noise, -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -self.max_action, self.max_action)
            
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = rewards + (1 - dones) * 0.99 * q_target
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy update
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.policy_delay == 0:
            actions_pred = self.actor(obs)
            q1, _ = self.critic(obs, actions_pred)
            actor_loss = -q1.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.total_steps += 1
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
