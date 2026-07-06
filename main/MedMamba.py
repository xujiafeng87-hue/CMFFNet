import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


class MambaEncoder(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm(x)
        x = self.act(x)
        return residual + x


class ConvMamba(nn.Module):
    def __init__(self, d_model, depth, grid_size, d_state=16, d_conv=4, expand=2, drop_rate=0.1):
        super().__init__()
        self.depth = depth
        self.grid_size = grid_size
        token_count = grid_size * grid_size

        self.inner_conv = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.token_pool = nn.Conv1d(in_channels=token_count, out_channels=1, kernel_size=1)
        self.layers = nn.ModuleList(
            [MambaEncoder(d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(depth)]
        )
        self.dropout = nn.Dropout(drop_rate)
        self.fc_center = nn.Linear(d_model, d_model)
        self.fc_output = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        batch_size, token_count, dim = x.shape
        grid_size = int(math.sqrt(token_count))
        if grid_size * grid_size != token_count:
            raise ValueError(f"ConvMamba expects a square token grid, got {token_count} tokens.")
        if grid_size != self.grid_size:
            raise ValueError(
                f"ConvMamba was initialized for {self.grid_size}x{self.grid_size} tokens, "
                f"but received {grid_size}x{grid_size}."
            )

        for layer in self.layers:
            residual = x
            x1 = layer(x)
            x1 = rearrange(x1, "b (h w) d -> b d h w", h=grid_size, w=grid_size)
            x1 = F.relu(self.inner_conv(x1), inplace=True)
            x1 = rearrange(x1, "b d h w -> b (h w) d")
            x = self.dropout(x1 + residual)

        center_pixel = x[:, x.shape[1] // 2, :]
        center_pixel = F.relu(self.fc_center(center_pixel), inplace=True)
        pooled_tokens = self.token_pool(x).squeeze(dim=1)
        output = torch.cat([pooled_tokens, center_pixel], dim=1)
        output = self.fc_output(output)
        return x, output


class VSSM(nn.Module):
    """
    ConvMamba image classifier with the same construction style used by the
    original CNN_Mamba scripts: VSSM(depths=..., dims=..., num_classes=...).
    """

    def __init__(
        self,
        patch_size=8,
        in_chans=3,
        num_classes=1000,
        depths=None,
        dims=None,
        d_state=24,
        d_conv=4,
        expand=1,
        drop_rate=0.1,
        input_size=128,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 4, 2]
        dims = dims or [96, 192, 384, 768]

        if isinstance(dims, int):
            dim = dims
        else:
            dim = dims[0]
        depth = max(depths) if isinstance(depths, (list, tuple)) else int(depths)

        if input_size % patch_size != 0:
            raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch_size}.")
        grid_size = input_size // patch_size

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.input_size = input_size
        self.grid_size = grid_size
        self.embed_dim = dim

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.local_trans_pixel = ConvMamba(
            d_model=dim,
            depth=depth,
            grid_size=grid_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            drop_rate=drop_rate,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim * 2, num_classes),
        )

        # Compatibility for scripts that inspect model.layers.
        self.layers = self.local_trans_pixel.layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward_features(self, x):
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = self.patch_embed(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        _, features = self.local_trans_pixel(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        return self.head(features)


if __name__ == "__main__":
    model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=4)
    data = torch.randn(1, 3, 128, 128)
    print(model(data).shape)
