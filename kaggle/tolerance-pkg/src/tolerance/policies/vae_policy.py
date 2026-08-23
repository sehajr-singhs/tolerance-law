"""VAE-based policy with latent tolerance conditioning.

Uses a Variational Autoencoder to learn a latent space of insertion strategies,
then conditions the decoder on tolerance specifications. This allows:
1. Interpolation between strategies (smooth tolerance transitions)
2. Latent space analysis (what does the model learn?)
3. Sample-efficient learning (latent space is lower-dimensional)

Key insight: the VAE's latent space encodes the *strategy* (sweep frequency,
amplitude, retry pattern) while the tolerance conditioning controls *which*
strategy to use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class TrajectoryEncoder(nn.Module):
    """Encodes a trajectory (obs, actions) into a latent distribution.

    Uses a temporal encoder (TCN) followed by mean/variance projection.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        seq_len: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Temporal encoder (1D conv over time)
        input_dim = obs_dim + action_dim
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, T, obs_dim)
            actions: (B, T, action_dim)
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = torch.cat([obs, actions], dim=-1)  # (B, T, obs+act)
        x = x.permute(0, 2, 1)  # (B, obs+act, T)
        h = self.temporal(x).squeeze(-1)  # (B, hidden)
        return self.fc_mu(h), self.fc_logvar(h)


class ToleranceDecoder(nn.Module):
    """Decodes latent + tolerance into action sequences.

    The tolerance conditioning controls which strategy to use:
    - Tight tolerance → small sweep amplitude, high frequency
    - Loose tolerance → large sweep amplitude, low frequency
    """

    def __init__(
        self,
        latent_dim: int = 32,
        tolerance_dim: int = 2,
        action_dim: int = 2,
        hidden_dim: int = 128,
        action_horizon: int = 50,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # Combine latent + tolerance
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + tolerance_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * action_horizon),
        )

    def forward(self, z: torch.Tensor, tolerance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent code
            tolerance: (B, tolerance_dim) tolerance spec
        Returns:
            (B, action_horizon, action_dim) action sequence
        """
        h = torch.cat([z, tolerance], dim=-1)
        actions = self.decoder(h)
        return actions.view(-1, self.action_horizon, self.action_dim)


class VAEPolicy(nn.Module):
    """VAE-based policy with tolerance conditioning.

    Training:
    1. Encode trajectory → latent distribution
    2. Sample latent
    3. Decode latent + tolerance → action sequence
    4. Optimize reconstruction + KL divergence

    Inference:
    1. Encode current observation history
    2. Sample latent (or use mean)
    3. Decode with tolerance → action sequence
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 2,
        tolerance_dim: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        obs_horizon: int = 2,
        action_horizon: int = 50,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.tolerance_dim = tolerance_dim
        self.latent_dim = latent_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.kl_weight = kl_weight

        # Encoder
        self.encoder = TrajectoryEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # Decoder
        self.decoder = ToleranceDecoder(
            latent_dim=latent_dim,
            tolerance_dim=tolerance_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_horizon=action_horizon,
        )

        # Observation history buffer
        self.obs_buffer = []

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory into latent distribution."""
        return self.encoder(obs, actions)

    def decode(self, z: torch.Tensor, tolerance: torch.Tensor) -> torch.Tensor:
        """Decode latent + tolerance into action sequence."""
        return self.decoder(z, tolerance)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, tolerance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Returns:
            recon_loss: reconstruction loss
            kl_loss: KL divergence
            total_loss: weighted sum
        """
        mu, logvar = self.encode(obs, actions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, tolerance)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, actions)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.kl_weight * kl_loss

        return recon_loss, kl_loss, total_loss

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, tolerance: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Generate action from observation and tolerance.

        Args:
            obs: (obs_dim,) current observation
            tolerance: (tolerance_dim,) tolerance specification
            deterministic: if True, use mean latent; if False, sample

        Returns:
            (action_dim,) action
        """
        # Update observation buffer
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > self.obs_horizon:
            self.obs_buffer = self.obs_buffer[-self.obs_horizon:]

        # Pad if needed
        while len(self.obs_buffer) < self.obs_horizon:
            self.obs_buffer.insert(0, np.zeros_like(obs))

        # Convert to tensor
        obs_t = torch.as_tensor(np.stack(self.obs_buffer), dtype=torch.float32).unsqueeze(0)
        tol_t = torch.as_tensor(tolerance, dtype=torch.float32).unsqueeze(0)

        # Encode (use zeros for actions since we don't have them yet)
        T = obs_t.shape[1]
        dummy_actions = torch.zeros(1, T, self.action_dim)
        mu, logvar = self.encode(obs_t, dummy_actions)

        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        # Decode
        actions = self.decode(z, tol_t)  # (1, action_horizon, action_dim)

        # Return first action
        return actions[0, 0].numpy()

    def reset(self):
        """Reset observation buffer."""
        self.obs_buffer = []


class ToleranceVAETrainer:
    """Trainer for the VAE policy."""

    def __init__(
        self,
        policy: VAEPolicy,
        lr: float = 1e-3,
        kl_anneal_steps: int = 1000,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.kl_anneal_steps = kl_anneal_steps
        self.step_count = 0

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        tolerance: torch.Tensor,
    ) -> dict:
        """Single training step.

        Args:
            obs: (B, T, obs_dim) observation sequences
            actions: (B, T, action_dim) action sequences
            tolerance: (B, tolerance_dim) tolerance specs

        Returns:
            dict with loss components
        """
        self.policy.train()
        self.optimizer.zero_grad()

        # Anneal KL weight
        self.step_count += 1
        self.policy.kl_weight = min(1.0, self.step_count / self.kl_anneal_steps) * 0.01

        recon_loss, kl_loss, total_loss = self.policy(obs, actions, tolerance)
        total_loss.backward()
        self.optimizer.step()

        return {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "kl_weight": self.policy.kl_weight,
        }


if __name__ == "__main__":
    """Smoke test."""
    batch_size = 4
    obs_dim = 8
    action_dim = 2
    tolerance_dim = 2
    seq_len = 50

    policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        tolerance_dim=tolerance_dim,
        action_horizon=seq_len,
    )

    trainer = ToleranceVAETrainer(policy)

    # Generate dummy data
    obs = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    tolerance = torch.tensor([[0.001, 10.0], [0.004, 5.0], [0.002, 8.0], [0.001, 15.0]])

    # Train
    for i in range(10):
        losses = trainer.train_step(obs, actions, tolerance)
        if (i + 1) % 5 == 0:
            print(f"Step {i+1}: {losses}")

    # Inference
    policy.eval()
    obs_np = np.random.randn(obs_dim).astype(np.float32)
    tol_np = np.array([0.002, 8.0], dtype=np.float32)
    action = policy.act(obs_np, tol_np)
    print(f"Action shape: {action.shape}")
