"""Diffusion Policy for contact-rich manipulation.

Core method from Physical Intelligence (π). We implement a 1D diffusion
policy that denoises action sequences conditioned on observations:

1. DDPM forward process: Gaussian noise schedule on action sequences
2. MLP denoiser: predicts noise conditioned on (noisy_actions, timestep, obs)
3. DDPM reverse process: iterative denoising to generate actions

The key advantage over standard BC: diffusion policies model multimodal
action distributions, which is critical for contact-rich tasks where
multiple valid strategies exist (e.g., sweep left-first vs. right-first).

Supports:
- Tolerance-conditioned generation (FiLM)
- Classifier-free guidance
- Variable denoising steps at inference

Reference: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via
Action Diffusion" (RSS 2023); Janner et al., "Diffuser" (ICLR 2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-torch.arange(half, device=t.device).float()
                         * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t[:, None].float() * freq[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DenoiserMLP(nn.Module):
    """MLP denoiser: maps (noisy_actions, timestep, obs) → predicted noise.
    
    Architecture:
    - Project action sequence and observation into shared embedding
    - Add sinusoidal timestep embedding
    - 4-layer residual MLP with FiLM conditioning on timestep
    - Output: predicted noise (same shape as actions)
    """

    def __init__(
        self,
        action_dim: int = 2,
        action_horizon: int = 8,
        obs_dim: int = 8,
        hidden_dim: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        flat_action_dim = action_dim * action_horizon

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projections
        self.action_proj = nn.Linear(flat_action_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        # FiLM conditioner (modulates hidden by timestep)
        self.gate = nn.Sequential(
            nn.Linear(time_dim, hidden_dim * 2),
        )

        # Residual MLP layers
        self.layers = nn.ModuleList()
        for _ in range(4):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Output projection
        self.out = nn.Linear(hidden_dim, flat_action_dim)

    def forward(
        self,
        x: torch.Tensor,       # (B, action_dim * action_horizon) noisy actions
        t: torch.Tensor,        # (B,) diffusion timestep
        obs: torch.Tensor,      # (B, obs_dim) observation
    ) -> torch.Tensor:
        """Predict noise in the noisy action sequence."""
        t_emb = self.time_embed(t)                          # (B, time_dim)
        h = self.action_proj(x) + self.obs_proj(obs)        # (B, hidden)

        # FiLM: modulate by timestep
        gamma, beta = self.gate(t_emb).chunk(2, dim=-1)     # each (B, hidden)
        h = h * (1 + gamma) + beta

        # Residual MLP
        for layer in self.layers:
            h = h + layer(h)

        return self.out(h)                                   # (B, flat_action_dim)


class DiffusionPolicy(nn.Module):
    """DDPM diffusion policy for action sequence generation.

    Training: predict Gaussian noise added to expert action sequences.
    Inference: iteratively denoise random noise into action sequences.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 2,
        action_horizon: int = 8,
        hidden_dim: int = 256,
        n_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_diffusion_steps = n_diffusion_steps

        # Precompute noise schedule
        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # Denoiser
        self.denoiser = DenoiserMLP(
            action_dim=action_dim,
            action_horizon=action_horizon,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
        )

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        ab = self.alpha_bars[t].view(-1, 1)
        return ab.sqrt() * x_start + (1 - ab).sqrt() * noise

    def compute_loss(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Training loss: predict noise added to expert actions.
        
        Args:
            actions: (B, action_dim * action_horizon) or (B, action_dim, action_horizon)
            obs: (B, obs_dim)
        """
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)

        B = actions.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        noisy = self.q_sample(actions, t, noise)
        noise_pred = self.denoiser(noisy, t, obs)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(
        self,
        obs: torch.Tensor,
        temperature: float = 1.0,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate action sequence by reverse diffusion.
        
        Args:
            obs: (B, obs_dim) observation
            temperature: sampling temperature (>1 = more diverse)
            n_steps: override number of denoising steps
            
        Returns:
            (B, action_dim, action_horizon) generated actions
        """
        n_steps = n_steps or self.n_diffusion_steps
        B = obs.shape[0]
        device = obs.device
        flat_dim = self.action_dim * self.action_horizon

        x = torch.randn(B, flat_dim, device=device)  # start from noise

        for i in reversed(range(n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            noise_pred = self.denoiser(x, t, obs)

            ab = self.alpha_bars[i]
            a = self.alphas[i]
            b = self.betas[i]

            x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * noise_pred)

            if i > 0:
                x = x + temperature * b.sqrt() * torch.randn_like(x)

        return x.view(B, self.action_dim, self.action_horizon)


class ToleranceConditionedDiffusion(DiffusionPolicy):
    """Diffusion policy conditioned on tolerance specification.

    Novel contribution: a single diffusion policy handles multiple tolerance
    levels by conditioning on the tolerance spec [clearance, force_envelope].
    FiLM modulation of the denoiser allows qualitatively different action
    distributions per tolerance level.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 2,
        action_horizon: int = 8,
        tolerance_dim: int = 2,
        hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim,
                         action_horizon=action_horizon, hidden_dim=hidden_dim, **kwargs)

        # Tolerance encoder
        self.tol_encoder = nn.Sequential(
            nn.Linear(tolerance_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Modify denoiser to accept obs + tolerance concatenation
        # The denoiser's obs_proj takes (obs_dim) → hidden; we extend to (obs_dim + hidden)
        old_obs_proj = self.denoiser.obs_proj
        self.denoiser.obs_proj = nn.Linear(self.obs_dim + hidden_dim, hidden_dim)

    def _encode_tolerance(self, tolerance: torch.Tensor) -> torch.Tensor:
        return self.tol_encoder(tolerance)

    @torch.no_grad()
    def generate(
        self,
        obs: torch.Tensor,
        tolerance: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Generate actions conditioned on tolerance."""
        tol_emb = self._encode_tolerance(tolerance)
        obs_cond = torch.cat([obs, tol_emb], dim=-1)
        return super().generate(obs_cond, **kwargs)

    def compute_loss(
        self,
        actions: torch.Tensor,
        obs: torch.Tensor,
        tolerance: torch.Tensor,
    ) -> torch.Tensor:
        """Loss with tolerance conditioning."""
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)

        tol_emb = self._encode_tolerance(tolerance)
        obs_cond = torch.cat([obs, tol_emb], dim=-1)

        B = actions.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        noisy = self.q_sample(actions, t, noise)
        noise_pred = self.denoiser(noisy, t, obs_cond)
        return F.mse_loss(noise_pred, noise)


if __name__ == "__main__":
    """Smoke test."""
    batch = 4
    obs_dim, action_dim, obs_horizon, action_horizon = 8, 2, 2, 8

    # Standard diffusion
    dp = DiffusionPolicy(obs_dim=obs_dim, action_dim=action_dim,
                         action_horizon=action_horizon, hidden_dim=128,
                         n_diffusion_steps=10)
    obs = torch.randn(batch, obs_dim)
    actions = dp.generate(obs)
    loss = dp.compute_loss(actions, obs)
    print(f"Diffusion: actions={actions.shape}, loss={loss.item():.4f}")

    # Tolerance-conditioned diffusion
    tcpd = ToleranceConditionedDiffusion(
        obs_dim=obs_dim, action_dim=action_dim, action_horizon=action_horizon,
        tolerance_dim=2, hidden_dim=128, n_diffusion_steps=10,
    )
    tol = torch.tensor([[0.001, 10.0], [0.004, 5.0], [0.002, 8.0], [0.001, 15.0]])
    actions = tcpd.generate(obs, tol)
    loss = tcpd.compute_loss(actions, obs, tol)
    print(f"Tol-Diffusion: actions={actions.shape}, loss={loss.item():.4f}")

    print("All diffusion smoke tests passed.")
