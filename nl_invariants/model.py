
import math
import logging

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class Orthogonal(nn.Module):
    
    def __init__(self, n: int, bias: bool = True) -> None:
        super().__init__()
        self.q_params = nn.Parameter(torch.Tensor(n, n))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    @torch.no_grad()
    def reset_parameters(self) -> None:
        
        # Init rotation parameters
        nn.init.uniform_(self.q_params, -math.pi, math.pi)
        
        self.q_params[:] = torch.triu(self.q_params, diagonal=1)
        
        # Init bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.q_params)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    @property
    def log_rotation(self):
        triu = torch.triu(self.q_params, diagonal=1)
        return triu - triu.T
    
    @property
    def q(self):
        return torch.matrix_exp(self.log_rotation)
    
    def log_det(self, x):
        return torch.full(x.shape[0], 1.0)
    
    def forward(self, x):
        x = nn.functional.linear(x, self.q, self.bias)
        return x
    
    def reverse(self, x):
        if self.bias is not None:
            x = x - self.bias
        x = nn.functional.linear(x, self.q.T)
        return x


class MLP(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, channel_mults: list) -> None:
        super().__init__()

        channels = [in_features] + [int(in_features*i) for i in channel_mults]
        channel_pairs = list(zip(channels[:-1], channels[1:]))

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch_i, ch_j),
                nn.ReLU()
            )
            for (ch_i, ch_j) in channel_pairs[:-1]
        ])

        self.final_layer = nn.Linear(channel_pairs[-1][0], out_features)
    
    def forward(self, x):
        for (block) in self.layers:
            x = block(x)

        return self.final_layer(x)


class CouplingLayer(nn.Module):
    
    def __init__(self, dim: int, channel_mults: list) -> None:
        super().__init__()
        
        self.transformed_features = dim//2
        self.num_params = dim - self.transformed_features
        self.mlp = MLP(self.num_params, self.transformed_features, channel_mults)
    
    def forward(self, x):
        x1, x2 = x[:, :self.transformed_features], x[:, self.transformed_features:]
        x1 = x1 + self.mlp(x2)
        return torch.cat([x1, x2], dim=1)
    
    def reverse(self, x):
        x1, x2 = x[:, :self.transformed_features], x[:, self.transformed_features:]
        x1 = x1 - self.mlp(x2)
        return torch.cat([x1, x2], dim=1)


class PermutationLayer(nn.Module):
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        
        self.indices = torch.randperm(dim)
        self.reverse_indices = torch.argsort(self.indices)
    
    def forward(self, x):
        
        return x[:, self.indices]
    
    def reverse(self, x):

        return x[:, self.reverse_indices]


class VolumePreservingNet(nn.Module):
    
    def __init__(self, dim: int, num_layers: int = 4, channel_mults: list = [1, 4, 4, 1]) -> None:
        super().__init__()
        layers = [i for _ in range(num_layers) for i in (Orthogonal(dim), CouplingLayer(dim, channel_mults))] + [Orthogonal(dim)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def reverse(self, x):
        for layer in reversed(self.layers):
            x = layer.reverse(x)
        return x


def build_model(dim: int, num_layers: int, channel_mults: list):
    model = VolumePreservingNet(dim, num_layers, channel_mults)
    
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("Trainable params: %d", num_of_parameters)

    return model