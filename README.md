Transformers for RL

Exploring whether Transformers' temporal memory gives them an advantage over memoryless policies in online RL, under both full and partial observability, using TD3 on Hopper-v5.

---

## Repository Structure

```
SAiDL-Summer-Assignment-2026/
├── README.md
├── LICENSE
├── requirements.txt
├── train_baseline.py          # TD3 with MLP actor/critic (baseline)
├── train_transformer.py       # TD3 with Transformer actor
├── src/
│   ├── __init__.py
│   ├── td3_base.py           # TD3 implementation, MLP networks, ReplayBuffer
│   ├── transformer_actor.py  # Transformer actor & critic architectures
│   └── utils.py              # Utilities: normalization, buffers, eval
├── checkpoints/              # Trained models & results

```

---

## Setup

### Installation
```bash
pip install -r requirements.txt
```

**Dependencies:**
- torch >= 2.0.0
- gymnasium >= 0.28.0
- numpy, matplotlib, scipy, tqdm, wandb

### Quick Start

#### 1. Baseline TD3 (MLP Actor)
```bash
python train_baseline.py
```
Trains 1M steps over 3 seeds. Results saved to `checkpoints/baseline_td3_*.json`.

#### 2. Transformer TD3 (Window Size Sweep)
```bash
python train_transformer.py
```
Tests L ∈ {4, 8, 16, 32} over 3 seeds each. Results saved to `checkpoints/transformer_td3_L*.json`.

---

## Key Components

### Core Implementations

**TD3 Agent (`src/td3_base.py`)**
- Twin Delayed DDPG with MLP actor & dual critics
- Experience replay buffer
- Target network updates via soft policy update (τ=0.005)
- Policy delay of 2 steps

**Transformer Actor (`src/transformer_actor.py`)**
- Causal self-attention over sliding windows
- Interleaved obs-action embeddings
- Configurable: 2 layers, 4 heads, hidden_dim=128
- Pre-LN architecture
- Output: Tanh-scaled actions

**Utilities (`src/utils.py`)**
- Running mean/std normalization
- Observation buffering for sliding windows
- Policy evaluation loop

---

## Experiments

### 1. Baseline: Full Observability
- Environment: Hopper-v5 (17 obs dims, 3 action dims)
- Training: 1M environment steps × 3 seeds
- Config: MLP (256→256), τ=0.005, batch=256, buffer=1M
- Metric: Mean ± std return over 5 test episodes

### 2. Transformer Sweep
- Window sizes: L ∈ {4, 8, 16, 32}
- Same train/test protocol
- Measures: Return vs. window size; attention patterns

### 3. Partial Observability (Planned)
- Hidden velocities
- Observation noise (σ ∈ {0.1, 0.3})
- Delayed rewards (K ∈ {10, 30})
- RLHF with learned reward model
- Attention attribution analysis (Chefer et al.)

---

## Configuration

**Hyperparameters** :
```python
# TD3
actor_lr = 3e-4
critic_lr = 3e-4
tau = 0.005              # soft update rate
policy_delay = 2         # actor update frequency
batch_size = 256
buffer_size = 1e6
gamma = 0.99

# Transformer
hidden_dim = 128
num_layers = 2
num_heads = 4
window_sizes = [4, 8, 16, 32]
positional_encoding = 'learned'  # or sinusoidal/RoPE
```

---

## Results

Results are saved in JSON format with structure:
```json
{
  "step": 5000,
  "mean": 1234.5,
  "std": 45.2
}
```



---

## Author

**Aayush Kushwaha**  

