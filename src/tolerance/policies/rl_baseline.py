"""Simple DQN baseline for Tolerance Law comparison.

Shows that the non-monotone capacity effect holds across methods.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """Simple Q-network for discrete actions."""
    
    def __init__(self, obs_dim: int, n_actions: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_actions),
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleDQN:
    """DQN with experience replay and target network."""
    
    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 9,  # 3x3 grid of (dx, dy) commands
        width: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update: int = 100,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Action mapping: 9 discrete actions → (dx, dy) velocity commands
        self.action_map = np.array([
            [-0.3, 0.0],   # 0: retract
            [-0.1, 0.0],   # 1: slow retract
            [0.0, 0.0],    # 2: hold
            [0.1, 0.0],    # 3: slow approach
            [0.3, 0.0],    # 4: fast approach
            [0.0, -0.1],   # 5: left
            [0.0, 0.1],    # 6: right
            [0.15, -0.1],  # 7: approach + left
            [0.15, 0.1],   # 8: approach + right
        ])
        
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_net = QNetwork(obs_dim, n_actions, width)
        self.target_net = QNetwork(obs_dim, n_actions, width)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.steps = 0
    
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if not greedy and random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_net(obs_t)
                action_idx = q_values.argmax(dim=1).item()
        
        return self.action_map[action_idx], action_idx
    
    def store_transition(self, obs, action_idx, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.buffer.append((obs, action_idx, reward, next_obs, done))
    
    def update(self):
        """Update Q-network from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_obs = torch.as_tensor(np.array(next_obs), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        
        # Current Q values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.steps += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.steps / self.epsilon_decay
        )
        
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def convert_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to velocity command."""
        return self.action_map[action_idx].copy()


def train_dqn(
    env,
    width: int = 64,
    n_episodes: int = 200,
    max_steps: int = 1200,
    seed: int = 0,
    progress: bool = False,
) -> dict:
    """Train DQN and return results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    agent = SimpleDQN(obs_dim=8, width=width)
    episode_rewards = []
    successes = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action, action_idx = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.store_transition(obs, action_idx, reward, next_obs, done)
            agent.update()
            
            obs = next_obs
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        successes.append(1.0 if info.get('success', False) else 0.0)
        
        if progress and (ep + 1) % 50 == 0:
            recent = successes[-50:]
            print(f"  ep {ep+1}/{n_episodes} | "
                  f"success={sum(recent)/len(recent):.2f} | "
                  f"eps={agent.epsilon:.3f}")
    
    return {
        'success_rate': sum(successes[-50:]) / len(successes[-50:]),
        'mean_reward': np.mean(episode_rewards[-50:]),
        'episode_rewards': episode_rewards,
        'successes': successes,
    }


if __name__ == "__main__":
    """Quick smoke test."""
    import sys
    sys.path.insert(0, 'src')
    from tolerance.envs.planar_insertion import PlanarInsertion
    from tolerance.policies.expert import DitherExpert
    
    for c in [0.0005, 0.002, 0.004]:
        for w in [32, 128, 256]:
            env = PlanarInsertion(clearance=c, seed=0)
            result = train_dqn(env, width=w, n_episodes=100, seed=0)
            print(f"c={c*1000:.1f}mm w={w:3d} | "
                  f"success={result['success_rate']:.2f} | "
                  f"reward={result['mean_reward']:.2f}")
