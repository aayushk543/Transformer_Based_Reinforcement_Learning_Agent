"""
Training script for TD3 baseline on Hopper-v5.
Trains MLP actor and critic for 1M steps over 3 seeds.
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))
from src.td3_base import TD3Agent, ReplayBuffer
from src.utils import RunningNormalization, eval_policy


def train_baseline_td3(seed: int = 0, total_steps: int = 1_000_000, 
                       eval_freq: int = 5000, save_dir: str = "checkpoints"):
    """Train TD3 baseline."""
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(save_dir).mkdir(exist_ok=True)
    
    # Environment
    env = gym.make("Hopper-v5")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"[Seed {seed}] Environment: Hopper-v5 | Obs: {obs_dim} | Action: {action_dim}")
    print(f"Device: {device}")
    
    # Agent
    agent = TD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=3e-4,
        critic_lr=3e-4,
        hidden_dim=256,
        tau=0.005,
        policy_delay=2,
        device=device
    )
    
    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=1e6, obs_dim=obs_dim, action_dim=action_dim)
    
    # Running normalization
    obs_normalizer = RunningNormalization(shape=(obs_dim,))
    
    # Training
    eval_returns = []
    step = 0
    episode = 0
    
    obs, info = env.reset(seed=seed)
    obs_normalizer.update(obs[None])
    
    pbar = tqdm(total=total_steps, desc=f"Seed {seed}")
    
    while step < total_steps:
        # Collect experience
        for _ in range(min(1000, total_steps - step)):
            # Select action
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in replay buffer
            replay_buffer.add(obs, action, reward, next_obs, done)
            obs_normalizer.update(next_obs[None])
            
            step += 1
            pbar.update(1)
            
            if done:
                obs, info = env.reset()
                obs_normalizer.update(obs[None])
                episode += 1
            else:
                obs = next_obs
        
        # Update networks
        if replay_buffer.size > 256:
            for _ in range(1000):
                losses = agent.update(batch_size=256, replay_buffer=replay_buffer)
        
        # Evaluate
        if step % eval_freq == 0:
            mean_return, std_return = eval_policy(agent, env, num_episodes=5, 
                                                  use_transformer=False)
            eval_returns.append({
                "step": step,
                "mean": mean_return,
                "std": std_return
            })
            pbar.set_postfix({"eval_return": f"{mean_return:.2f} ± {std_return:.2f}"})
    
    pbar.close()
    
    # Save results
    results_file = f"{save_dir}/baseline_td3_seed{seed}.json"
    with open(results_file, "w") as f:
        json.dump(eval_returns, f, indent=2)
    
    # Save model
    model_file = f"{save_dir}/baseline_td3_seed{seed}.pt"
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }, model_file)
    
    print(f"\nTraining complete! Final return: {eval_returns[-1]['mean']:.2f}")
    print(f"Saved to {results_file} and {model_file}")
    
    return eval_returns


if __name__ == "__main__":
    # Train over 3 seeds
    all_results = {}
    for seed in range(3):
        results = train_baseline_td3(seed=seed, total_steps=1_000_000)
        all_results[f"seed_{seed}"] = results
    
    # Save summary
    with open("checkpoints/baseline_td3_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("All training complete!")
    print("="*60)
