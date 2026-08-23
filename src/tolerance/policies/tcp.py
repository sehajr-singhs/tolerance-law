"""Tolerance-Conditioned Policy (TCP) — Novel Method.

The key insight: manufacturing policies should be conditioned on the
*tolerance specification*, not just the task. A single TCP can handle
multiple tolerance levels by learning to adjust its behavior:

- Tight tolerance → conservative, high-frequency sweep, slow approach
- Loose tolerance → aggressive, low-frequency sweep, fast approach
- Medium tolerance → interpolated behavior

Architecture:
1. Tolerance Encoder: maps tolerance spec → embedding
2. Policy Network: MLP conditioned on tolerance embedding
3. Adaptive Action Selection: adjusts action scale based on tolerance

This is the method that makes the Tolerance Law *actionable* — instead of
just measuring the boundary, TCP operates optimally within it.

Key novelty: the tolerance conditioning is *not* just a bias — it modulates
the policy's action distribution via FiLM (Feature-wise Linear Modulation),
allowing the same network to produce qualitatively different behaviors
depending on the tolerance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ToleranceEncoder(nn.Module):
    """Encodes tolerance specification into a conditioning embedding.

    The tolerance spec includes:
    - clearance (mm): lateral gap between peg and channel
    - force_envelope (N): acceptable force range
    - position_tolerance (mm): acceptable position error

    The encoder maps this to a latent embedding that modulates the policy.
    """

    def __init__(
        self,
        tolerance_dim: int = 3,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(tolerance_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, tolerance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tolerance: (B, tolerance_dim) tolerance specification
        Returns:
            (B, embedding_dim) tolerance embedding
        """
        return self.encoder(tolerance)


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Modulates features based on conditioning:
    h_out = gamma * h_in + beta

    where gamma and beta are learned functions of the conditioning.
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, ..., feature_dim)
            condition: (B, condition_dim)
        Returns:
            modulated features
        """
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)

        # Broadcast for different sequence lengths
        if features.dim() == 3:
            gamma = gamma[:, None, :]
            beta = beta[:, None, :]

        return gamma * features + beta


class ToleranceConditionedMLP(nn.Module):
    """MLP policy with FiLM conditioning on tolerance.

    Architecture:
    1. Observation encoder (shared)
    2. Tolerance encoder (FiLM)
    3. Action decoder (modulated by tolerance)

    The FiLM modulation allows the same network to produce qualitatively
    different actions depending on the tolerance — this is the key
    mechanism that makes TCP work.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 2,
        tolerance_dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.tolerance_dim = tolerance_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
        )

        # Tolerance encoder
        self.tolerance_encoder = ToleranceEncoder(
            tolerance_dim=tolerance_dim,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # FiLM layers
        self.film_layers = nn.ModuleList([
            FiLMModulation(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Action decoder with FiLM
        self.action_decoder = nn.ModuleList()
        for i in range(n_layers):
            self.action_decoder.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ))
        self.action_decoder.append(nn.Linear(hidden_dim, action_dim))

        # Adaptive action scale (learned from tolerance)
        self.action_scale = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),  # Always positive
        )

    def forward(
        self,
        obs: torch.Tensor,
        tolerance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim) observation
            tolerance: (B, tolerance_dim) tolerance specification
        Returns:
            (B, action_dim) action with adaptive scale
        """
        # Encode
        h = self.obs_encoder(obs)
        tol_emb = self.tolerance_encoder(tolerance)

        # FiLM modulation through layers
        for film, layer in zip(self.film_layers, self.action_decoder):
            h = layer(h)
            h = film(h, tol_emb)
            h = F.silu(h)

        # Final action
        action_mean = self.action_decoder[-1](h)

        # Adaptive scale based on tolerance
        scale = self.action_scale(tol_emb)
        action = action_mean * scale

        return action

    def act(
        self,
        obs: np.ndarray,
        tolerance: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Generate action for a single timestep.

        Args:
            obs: (obs_dim,) observation
            tolerance: (tolerance_dim,) tolerance spec
            deterministic: if True, use mean; if False, add noise

        Returns:
            (action_dim,) action
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            tol_t = torch.as_tensor(tolerance, dtype=torch.float32).unsqueeze(0)
            action = self.forward(obs_t, tol_t)
            return action.squeeze(0).numpy()


class TCPTrainer:
    """Trainer for Tolerance-Conditioned Policy.

    Training objective:
    1. BC loss: MSE between predicted and expert actions
    2. Tolerance consistency: actions should change smoothly with tolerance
    3. Smoothness: consecutive actions should be smooth
    """

    def __init__(
        self,
        policy: ToleranceConditionedMLP,
        lr: float = 1e-3,
        smooth_weight: float = 0.1,
        tolerance_consistency_weight: float = 0.05,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.smooth_weight = smooth_weight
        self.tolerance_consistency_weight = tolerance_consistency_weight

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        tolerance: torch.Tensor,
        next_tolerance: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute training loss.

        Args:
            obs: (B, obs_dim) observations
            actions: (B, action_dim) expert actions
            tolerance: (B, tolerance_dim) tolerance specs
            next_tolerance: (B, tolerance_dim) next tolerance (for consistency)

        Returns:
            dict with loss components
        """
        # BC loss
        pred_actions = self.policy(obs, tolerance)
        bc_loss = F.mse_loss(pred_actions, actions)

        # Smoothness loss (penalize large action changes)
        smooth_loss = torch.mean(torch.abs(pred_actions[:, 1:] - pred_actions[:, :-1])) if pred_actions.dim() > 2 else torch.tensor(0.0)

        # Tolerance consistency (actions should be similar for similar tolerances)
        tol_loss = torch.tensor(0.0)
        if next_tolerance is not None:
            # Small tolerance change should produce small action change
            tol_diff = torch.norm(tolerance - next_tolerance, dim=1)
            with torch.no_grad():
                act_diff = torch.norm(pred_actions - self.policy(obs, next_tolerance), dim=1)
            # Encourage proportionality
            tol_loss = F.mse_loss(act_diff, tol_diff)

        total_loss = bc_loss + self.smooth_weight * smooth_loss + self.tolerance_consistency_weight * tol_loss

        return {
            "bc_loss": bc_loss.item(),
            "smooth_loss": smooth_loss.item(),
            "tolerance_loss": tol_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        tolerance: torch.Tensor,
        next_tolerance: Optional[torch.Tensor] = None,
    ) -> dict:
        """Single training step."""
        self.policy.train()
        self.optimizer.zero_grad()

        losses = self.compute_loss(obs, actions, tolerance, next_tolerance)
        torch.tensor(losses["total_loss"], requires_grad=True).backward()
        self.optimizer.step()

        return losses


if __name__ == "__main__":
    """Smoke test."""
    batch_size = 32
    obs_dim = 8
    action_dim = 2
    tolerance_dim = 3

    # Create policy
    policy = ToleranceConditionedMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        tolerance_dim=tolerance_dim,
    )

    # Create trainer
    trainer = TCPTrainer(policy)

    # Generate dummy data
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    tolerance = torch.rand(batch_size, tolerance_dim) * torch.tensor([0.01, 20.0, 5.0])

    # Train
    for i in range(10):
        losses = trainer.train_step(obs, actions, tolerance)
        if (i + 1) % 5 == 0:
            print(f"Step {i+1}: {losses}")

    # Inference
    obs_np = np.random.randn(obs_dim).astype(np.float32)
    tol_np = np.array([0.002, 8.0, 2.0], dtype=np.float32)
    action = policy.act(obs_np, tol_np)
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
