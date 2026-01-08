import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConsistencyDownscalingModel(nn.Module):
    """
    Consistency Model für Climate Downscaling
    Kompatibel mit Emely's Dataloaders und CorrDiff's HRRRMiniDataset
    
    Architecture:
        Low-res Input (8×8) → Encoder → Upsample → U-Net → High-res Output (64×64)
    """
    
    def __init__(
        self,
        # Input/Output
        n_input_features: int = 28,  # ERA5 features
        n_output_vars: int = 4,      # HRRR variables
        
        # Architecture
        encoding_dim: int = 128,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        
        # Resolution
        low_res_size: Tuple[int, int] = (8, 8),
        high_res_size: Tuple[int, int] = (64, 64),
        
        # Consistency Model specific (EDM-style)
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        num_steps: int = 18,
    ):
        super().__init__()
        
        self.n_output_vars = n_output_vars
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        
        # EDM parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.num_steps = num_steps
        
        # 1. Encoder: Process low-res features
        # Ähnlich wie Emely's encoder, aber für Consistency
        self.encoder = nn.Sequential(
            nn.Linear(n_input_features, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU()
        )
        
        # 2. Upsampler: Low-res (8×8) → High-res (64×64)
        # Ersetzt Emely's graph downscaler mit standard Upsampling
        self.upsampler = nn.Sequential(
            nn.Conv2d(encoding_dim, encoding_dim, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(encoding_dim, model_channels, 3, padding=1),
        )
        
        # 3. U-Net Backbone (für Consistency)
        self.unet = ConsistencyUNet(
            in_channels=model_channels,
            out_channels=n_output_vars,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
        
        # Precompute discretization
        self.register_buffer('timesteps', self._get_timesteps())
    
    def _get_timesteps(self) -> torch.Tensor:
        """EDM-style discretization"""
        rho_inv = 1.0 / self.rho
        steps = torch.arange(self.num_steps)
        t = (self.sigma_max ** rho_inv + 
             steps / (self.num_steps - 1) * 
             (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv))
        return t ** self.rho
    
    # EDM Preconditioning functions
    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
    
    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
    
    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
    
    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma.clamp(min=1e-20))
    
    def forward(
        self,
        x_low: torch.Tensor,          # (B, N_low, features) or (B, features, H, W)
        sigma: Optional[torch.Tensor] = None,  # (B,) noise levels
        inference: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x_low: Low-res input
                - If graph format: (B, N_low, features) where N_low = 64 (8×8)
                - If image format: (B, features, 8, 8)
            sigma: Noise levels for consistency training
            inference: If True, skip EDM preconditioning
            
        Returns:
            x_high: (B, n_output_vars, 64, 64)
        """
        B = x_low.shape[0]
        
        # Convert graph format to image format if needed
        if x_low.dim() == 3:  # (B, N_low, features)
            # Reshape to spatial grid
            x_low = x_low.view(B, self.low_res_size[0], self.low_res_size[1], -1)
            x_low = x_low.permute(0, 3, 1, 2)  # (B, features, H, W)
        
        # 1. Encode low-res features
        # Reshape to (B*H*W, features)
        B, C, H, W = x_low.shape
        x_flat = x_low.permute(0, 2, 3, 1).reshape(-1, C)
        encoded = self.encoder(x_flat)  # (B*H*W, encoding_dim)
        encoded = encoded.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, encoding_dim, H, W)
        
        # 2. Upsample to high-res
        x_upsampled = self.upsampler(encoded)  # (B, model_channels, 64, 64)
        
        # 3. U-Net processing
        if sigma is not None and not inference:
            # Training with EDM preconditioning
            c_skip = self.c_skip(sigma)[:, None, None, None]
            c_out = self.c_out(sigma)[:, None, None, None]
            c_in = self.c_in(sigma)[:, None, None, None]
            c_noise = self.c_noise(sigma)
            
            # Scale input
            x_in = x_upsampled * c_in
            
            # U-Net forward
            F_theta = self.unet(x_in, c_noise)
            
            # Skip connection and output scaling
            x_high = c_skip * x_upsampled + c_out * F_theta
        else:
            # Inference (no noise conditioning)
            x_high = self.unet(x_upsampled, None)
        
        return x_high
    
    def sample(
        self,
        x_low: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Sample from model (inference)
        
        Args:
            x_low: (B, N_low, features) or (B, features, H, W)
            num_steps: Number of sampling steps (1 for one-step generation)
            
        Returns:
            x_high: (B, n_output_vars, 64, 64)
        """
        device = x_low.device
        B = x_low.shape[0]
        
        # Get initial prediction
        x_high = self.forward(x_low, sigma=None, inference=True)
        
        # Add noise and denoise (consistency sampling)
        if num_steps > 1:
            noise = torch.randn_like(x_high) * self.sigma_max
            x_high = x_high + noise
            
            timesteps = self.timesteps[-num_steps:].to(device)
            for t in timesteps:
                sigma = torch.full((B,), t.item(), device=device)
                x_high = self.forward(x_low, sigma=sigma, inference=False)
        
        return x_high


class ConsistencyUNet(nn.Module):
    """
    Simplified U-Net for Consistency Model
    Basierend auf CorrDiff's U-Net aber vereinfacht
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        channel_mult: Tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        
        # Time embedding (for noise level σ)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input conv
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(ch, out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            
            if i < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
        
        # Middle
        self.middle_block = ResBlock(ch, ch, time_embed_dim, dropout)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(ch, out_ch, time_embed_dim, dropout)
                )
                ch = out_ch
            
            if i > 0:
                self.up_blocks.append(Upsample(ch))
        
        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, sigma: Optional[torch.Tensor]) -> torch.Tensor:
        if sigma is not None:
            t_emb = self.time_embed(sigma)
        else:
            t_emb = torch.zeros(x.shape[0], self.time_embed[1].out_features, device=x.device)
        
        h = self.input_conv(x)
        
        # Down
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # Middle
        h = self.middle_block(h, t_emb)
        
        # Up
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        return self.output_conv(h)


# Helper modules (von CorrDiff übernommen)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device) * -emb)
        emb = sigma[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# Training Loss (kompatibel mit Emely's Trainer)

class ConsistencyLoss(nn.Module):
    """
    Consistency Training Loss
    Kompatibel mit Emely's Training Loop
    """
    
    def __init__(self, model: ConsistencyDownscalingModel):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        x_low: torch.Tensor,
        x_high_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_low: (B, N_low, features) low-res input
            x_high_target: (B, n_output_vars, 64, 64) high-res target
            
        Returns:
            loss: scalar
        """
        B = x_low.shape[0]
        device = x_low.device
        
        # Sample noise pair
        n = self.model.num_steps
        idx = torch.randint(0, n - 1, (B,), device=device)
        sigma_t = self.model.timesteps[idx]
        sigma_t_next = self.model.timesteps[idx + 1]
        
        # Add noise to target
        noise = torch.randn_like(x_high_target)
        x_t = x_high_target + sigma_t[:, None, None, None] * noise
        x_t_next = x_high_target + sigma_t_next[:, None, None, None] * noise
        
        # Predictions
        with torch.no_grad():
            target = self.model(x_low, sigma_t_next, inference=False)
        
        pred = self.model(x_low, sigma_t, inference=False)
        
        # Loss
        loss = F.mse_loss(pred, target)
        
        return loss