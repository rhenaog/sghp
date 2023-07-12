import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class sigmoid_gated_kernel(nn.Module):

    def __init__(self,
                 num_types=1, d_type=1):
        super().__init__()

        self.d_type = d_type
        self.num_types = num_types

        if num_types == 1:

            self.lengthscale = torch.nn.Parameter(torch.randn(1))
            self.sigma = torch.nn.Parameter(torch.randn(1))
            self.alpha = torch.nn.Parameter(torch.randn(1))
            self.s = torch.nn.Parameter(torch.randn(1))
            self.p = torch.nn.Parameter(torch.randn(1))

        else:
            self.lengthscale = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus(beta=0.4))
            self.sigma = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())
            self.s = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus(beta=0.3))
            self.alpha = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())
            self.p = nn.Sequential(nn.Linear(d_type * 2, 1, bias=False), nn.Softplus())

    def forward(self, time_diff, combined_embeddings=None):

        d = time_diff

        if self.num_types == 1:
            lengthscale = F.softplus(self.lengthscale, beta=0.4)
            sigma = F.softplus(self.sigma)
            s = F.softplus(self.s, beta=0.3)
            alpha = F.softplus(self.alpha)
            p = F.softplus(self.p)

        else:
            lengthscale = self.lengthscale(combined_embeddings).squeeze(-1)
            sigma = self.sigma(combined_embeddings).squeeze(-1)
            s = self.s(combined_embeddings).squeeze(-1)
            alpha = self.alpha(combined_embeddings).squeeze(-1)
            p = self.p(combined_embeddings).squeeze(-1)

        k1 = (1 + torch.exp(p - d)) ** (-s)
        k2 = (1 + (d ** 2) / (2 * alpha * lengthscale ** 2)) ** (-alpha)
        scores = (sigma) * (k1) * (k2)

        return scores

