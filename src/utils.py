"""Utility functions for training and evaluation."""

import numpy as np
import torch
from collections import deque
from typing import Optional


class RunningNormalization:
    """Running mean and standard deviation normalization."""
    
    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize input."""
        return x * np.sqrt(self.var + 1e-8) + self.mean


class ObservationBuffer:
    """Sliding window buffer for observations and actions."""
    
    def __init__(self, window_size: int, obs_dim: int, action_dim: int):
        self.window_size = window_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.obs_buffer = deque(maxlen=window_size)
        self.action_buffer = deque(maxlen=window_size - 1)
    
    def reset(self):
        """Clear buffers."""
        self.obs_buffer.clear()
        self.action_buffer.clear()
    
    def add(self, obs: np.ndarray, action: Optional[np.ndarray] = None):
        """Add observation and action to buffer."""
        self.obs_buffer.append(obs)
        if action is not None:
            self.action_buffer.append(action)
    
    def get_state(self) -> tuple:
        """Get current state as tensors."""
        if len(self.obs_buffer) < self.window_size:
            # Pad with zeros
            obs = list(self.obs_buffer) + [np.zeros(self.obs_dim)] * (self.window_size - len(self.obs_buffer))
        else:
            obs = list(self.obs_buffer)
        
        obs = torch.FloatTensor(np.array(obs)).unsqueeze(0)  # (1, window, obs_dim)
        
        # Actions are always window_size - 1
        actions = torch.FloatTensor(np.array(list(self.action_buffer))).unsqueeze(0)  # (1, window-1, action_dim)
        
        return obs, actions
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.obs_buffer) == self.window_size


def compute_episode_return(episode_rewards: list) -> float:
    """Compute discounted episode return."""
    gamma = 0.99
    return_val = 0.0
    for i, r in enumerate(reversed(episode_rewards)):
        return_val += (gamma ** i) * r
    return return_val


def eval_policy(agent, env, num_episodes: int = 5, window_size: int = 8,
                use_transformer: bool = False) -> tuple:
    """Evaluate policy."""
    episode_returns = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        
        if use_transformer:
            obs_buffer = ObservationBuffer(window_size, obs.shape[0], agent.action_dim)
            obs_buffer.add(obs)
        
        while not done and ep_length < 1000:
            if use_transformer:
                obs_tensor, actions_tensor = obs_buffer.get_state()
                obs_tensor = obs_tensor.to(agent.device)
                with torch.no_grad():
                    action = agent.actor(obs_tensor).cpu().numpy()[0]
                obs_buffer.add(action)
            else:
                with torch.no_grad():
                    action = agent.select_action(obs, training=False)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1
            
            if use_transformer:
                obs_buffer.add(obs)
        
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
    
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    return mean_return, std_return
