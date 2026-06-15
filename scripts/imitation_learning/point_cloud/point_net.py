import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP

class PointNet(nn.Module):
    def __init__(self, encoder_hidden_dims, action_hidden_dims, proprio_dim, action_dim, predict_std):
        super().__init__()
        self.predict_std = predict_std
        self.set_encoder = MLP(
            in_channels=3,
            hidden_channels=encoder_hidden_dims,
            norm_layer=nn.LayerNorm,
        )
        self.point_norm = torch.nn.LayerNorm(encoder_hidden_dims[-1])
        self.proprio_encoder = torch.nn.Sequential(torch.nn.Linear(proprio_dim, encoder_hidden_dims[-1]), torch.nn.LayerNorm(encoder_hidden_dims[-1]))
        out_dim = action_dim * 2 if predict_std else action_dim
        self.action_head = MLP(
            in_channels=encoder_hidden_dims[-1]*2,
            hidden_channels=[*action_hidden_dims, out_dim],
        )

    def forward(self, x, proprio):                     # x: (B, N, 3)
        assert x.ndim == 3 and x.shape[-1] == 3
        assert proprio.ndim == 2
        feats = self.set_encoder(x)
        pooled = self.point_norm(feats.amax(dim=1))
        proprio_feats = self.proprio_encoder(proprio)
        feats = torch.cat([pooled, proprio_feats], dim=-1)
        out = self.action_head(feats)
        if self.predict_std:
            mean, log_std = out.chunk(2, dim=-1)
            return mean, log_std
        return out

    def calculate_loss(self, points, proprio, targets):
        if self.predict_std:
            mean, log_std = self.forward(points, proprio)
            var = torch.exp(2 * log_std)      # std = exp(log_std) > 0, so var > 0
            return F.gaussian_nll_loss(mean, targets, var)
        return F.mse_loss(self.forward(points, proprio), targets)