<h1 align="center">Deep RL Showdown: DQN vs PPO on LunarLander</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg"/></a>
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg"/>
  <img src="https://img.shields.io/badge/Gym-LunarLander‑v2-purple"/>
</p>

---

## 🎯 Overview

This project provides a **comprehensive comparison** between two fundamental Deep RL paradigms applied to the challenging **LunarLander-v2** environment:

* **DQN (Deep Q-Network)**: Value-based approach with experience replay and target networks
* **PPO (Proximal Policy Optimization)**: Policy gradient method with clipped surrogate objective
* **Empirical analysis** of convergence speed, sample efficiency, stability, and hyperparameter sensitivity

Our implementation explores the theoretical and empirical trade-offs between these approaches, providing insights into when to use each algorithm.

<p align="center">
  <img src="assets/images/lunarlander_demo.gif" alt="LunarLander environment" width="40%">
</p>

---

## 📑 Table of Contents

1. [Environment](#environment)
2. [Theoretical Background](#theoretical-background)
3. [Methodology](#methodology)
4. [Experiments & Results](#experiments--results)
5. [Quick Start](#quick-start)
6. [Analysis & Insights](#analysis--insights)
7. [References](#references)

---

## Environment

**LunarLander-v2** is a classic control task from OpenAI Gym where an agent must safely land a lunar module on a landing pad.

**State Space** (8-dimensional continuous):
* Position (x, y)
* Velocity (vₓ, vᵧ)
* Angle, angular velocity
* Left/right leg ground contact (binary)

**Action Space** (4 discrete actions):
* 0: Do nothing
* 1: Fire left engine
* 2: Fire main engine
* 3: Fire right engine

**Reward Structure**:
* Moving toward/away from landing pad: +/- reward
* Crash: -100
* Successful landing: +100
* Leg ground contact: +10 each
* Fuel consumption: -0.3 per firing

**Success Criterion**: Average reward ≥ 200 over 100 consecutive episodes

---

## Theoretical Background

### DQN (Deep Q-Network)

DQN approximates the optimal action-value function Q*(s,a) using a deep neural network.

**Bellman Optimality Equation**:

$$Q^*(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a') | s, a]$$

**Loss Function**:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

**Key Components**:
* **Experience Replay**: Break temporal correlations by sampling from replay buffer $\mathcal{D}$
* **Target Network** $\theta^-$: Stabilize learning by fixing targets for C steps
* **ε-greedy exploration**: Balance exploration/exploitation

<p align="center">
  <img src="assets/images/dqn_architecture.png" alt="DQN architecture" width="45%">
</p>

---

### PPO (Proximal Policy Optimization)

PPO directly optimizes a stochastic policy $\pi_\theta(a|s)$ using policy gradients with conservative updates.

**Clipped Surrogate Objective**:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where:
* $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ (probability ratio)
* $\hat{A}_t$ = advantage estimate (Q - V baseline)
* $\epsilon$ = clip range (typically 0.2)

**Key Components**:
* **Advantage estimation**: GAE (Generalized Advantage Estimation) for variance reduction
* **Multiple epochs**: Reuse collected data K times (sample efficiency)
* **Entropy bonus**: Encourage exploration via $-\beta \mathcal{H}(\pi)$

<p align="center">
  <img src="assets/images/ppo_clip.png" alt="PPO clipping mechanism" width="45%">
</p>

---

## Methodology

### DQN Implementation

**Network Architecture**:
```
Input (8) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(4)
```

**Hyperparameters**:
* Replay buffer size: 100,000
* Batch size: 256
* Learning rate: 5e-4 (Adam)
* Target network update: Every 1,000 steps
* γ (discount): 0.99
* ε decay: 1.0 → 0.01 over 10,000 steps

**Training Procedure**:
1. Collect transition (s, a, r, s') using ε-greedy
2. Store in replay buffer
3. Sample random minibatch
4. Compute TD target using target network
5. Update Q-network via gradient descent
6. Periodically copy weights to target network

---

### PPO Implementation

**Actor-Critic Architecture**:
```
Shared trunk: Input (8) → Dense(128, ReLU) → Dense(128, ReLU)
Actor head:   → Dense(4, Softmax)
Critic head:  → Dense(1)
```

**Hyperparameters**:
* Rollout length: 2,048 steps
* Epochs per rollout: 10
* Minibatch size: 64
* Learning rate: 3e-4 (Adam)
* γ (discount): 0.99
* GAE λ: 0.95
* Clip range ε: 0.2
* Entropy coefficient: 0.01

**Training Procedure**:
1. Collect N-step rollout using current policy
2. Compute advantages via GAE
3. For K epochs:
   * Sample minibatches from rollout
   * Update actor via clipped objective
   * Update critic via MSE loss
4. Repeat

---

## Experiments & Results

All experiments run for **1M timesteps** with 3 random seeds. Evaluation performed every 10k steps over 10 episodes.

### Convergence Speed

| Algorithm | Steps to 200 | Steps to Optimal (>240) | Figure |
|-----------|-------------:|------------------------:|:------:|
| **DQN**   | ~450k | ~700k | <img src="assets/images/dqn_training.png" alt="DQN training curve" width="45%"/> |
| **PPO**   | **~300k** | **~500k** | <img src="assets/images/ppo_training.png" alt="PPO training curve" width="45%"/> |

<p align="center">
  <img src="assets/images/convergence_comparison.png" alt="Training curves comparison" width="60%">
</p>

**Key Observations**:
* PPO reaches success threshold ~33% faster
* DQN shows more variance early in training
* PPO exhibits smoother, more monotonic improvement

---

### Sample Efficiency

| Metric | DQN | PPO |
|--------|----:|----:|
| Timesteps to success | 450k | **300k** |
| Final performance | 248 ± 12 | **256 ± 8** |
| Training stability (σ) | 45.2 | **28.7** |

<p align="center">
  <img src="assets/images/sample_efficiency.png" alt="Sample efficiency comparison" width="50%">
</p>

_Key takeaway_: PPO achieves superior sample efficiency due to multi-epoch updates and advantage normalization.

---

### Training Stability

<p align="center">
  <img src="assets/images/stability_comparison.png" alt="Training stability" width="50%">
</p>

**Standard deviation of returns (across 3 seeds)**:
* DQN: σ = 45.2
* PPO: σ = **28.7**

PPO's clipping mechanism prevents catastrophic policy updates, leading to more consistent training dynamics.

---

### Hyperparameter Sensitivity

<p align="center">
  <img src="assets/images/hyperparam_sensitivity.png" alt="Hyperparameter sensitivity" width="70%">
</p>

**DQN sensitive to**:
* Target network update frequency
* Replay buffer size
* Learning rate

**PPO robust to**:
* Clip range ε (works well in [0.1, 0.3])
* Learning rate variations
* Rollout length

---

## Quick Start

### Installation
```bash
# Create conda environment
conda create -n lunar-rl python=3.11
conda activate lunar-rl

# Install dependencies
pip install -r requirements.txt
```

### Training

**Train DQN**:
```bash
python train_dqn.py \
  --env LunarLander-v2 \
  --total-timesteps 1000000 \
  --buffer-size 100000 \
  --learning-rate 5e-4 \
  --seed 42
```

**Train PPO**:
```bash
python train_ppo.py \
  --env LunarLander-v2 \
  
  --total-timesteps 1000000 \
  --n-steps 2048 \
  --learning-rate 3e-4 \
  --seed 42
```

### Evaluation
```bash
# Evaluate trained model
python evaluate.py \
  --model-path checkpoints/ppo_best.pt \
  --n-episodes 100 \
  --render
```

### Reproduce All Results
```bash
# Run complete experimental suite
bash scripts/run_all_experiments.sh

# Generate comparison plots
python scripts/plot_results.py
```

---

## Analysis & Insights

### When to Use DQN?

✅ **Discrete action spaces**  
✅ **Off-policy learning required** (learn from demonstrations)  
✅ **High replay ratio desired** (maximize data reuse)

❌ Continuous actions (requires modifications)  
❌ Need for fast convergence  
❌ Limited hyperparameter tuning budget

---

### When to Use PPO?

✅ **Continuous OR discrete actions**  
✅ **Need stability and robustness**  
✅ **Quick prototyping** (works out-of-the-box)  
✅ **Stochastic policies beneficial**

❌ Sample efficiency absolutely critical  
❌ Cannot collect rollouts in parallel

---

### Key Takeaways

1. **PPO dominates for LunarLander**: Faster convergence, better stability, less tuning required
2. **DQN competitive but harder**: Requires careful hyperparameter selection
3. **Sample efficiency**: PPO's multi-epoch updates beat DQN's single-sample approach
4. **Variance reduction matters**: GAE + value baseline crucial for PPO's performance
5. **Target networks essential**: DQN unstable without them

---

## Project Structure
```
.
├── agents/
│   ├── dqn.py              # DQN implementation
│   ├── ppo.py              # PPO implementation
│   └── networks.py         # Neural network architectures
├── utils/
│   ├── replay_buffer.py    # Experience replay for DQN
│   ├── gae.py              # GAE computation for PPO
│   └── logger.py           # Training metrics logging
├── scripts/
│   ├── run_all_experiments.sh
│   └── plot_results.py
├── train_dqn.py            # DQN training script
├── train_ppo.py            # PPO training script
├── evaluate.py             # Model evaluation
└── requirements.txt
```

---

## Report & Poster

For the full theoretical background, implementation details, and extended results, see:

| Resource | Link |
|----------|------|
| 📑 **Project Report (PDF)** | [assets/report/report.pdf](assets/report/report.pdf) |
| 🖼️ **Summary Poster** | [assets/poster/poster.png](assets/poster/poster.png) |

---

## References

1. **Volodymyr Mnih**, Koray Kavukcuoglu, David Silver, et al.  
   *Human-level control through deep reinforcement learning.*  
   Nature 518, 529–533 (2015).

2. **John Schulman**, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.  
   *Proximal Policy Optimization Algorithms.*  
   arXiv:1707.06347 (2017).

3. **Greg Brockman**, Vicki Cheung, Ludwig Pettersson, et al.  
   *OpenAI Gym.*  
   arXiv:1606.01540 (2016).

4. **Richard S. Sutton**, **Andrew G. Barto**.  
   *Reinforcement Learning: An Introduction (2nd ed.).*  
   MIT Press (2018).

---

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{lunarlander-dqn-ppo,
  author = {Zaid LK},
  title = {Deep RL Showdown: DQN vs PPO on LunarLander},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/lunarlander-dqn-ppo}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

* OpenAI Gym for the LunarLander-v2 environment
* Stable-Baselines3 for reference implementations
* The Deep RL community for valuable discussions
