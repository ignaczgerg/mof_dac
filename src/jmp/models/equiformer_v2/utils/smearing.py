"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

# taken from fairchem
@torch.jit.script
def _smooth_envelope_cos(r: torch.Tensor, r_cut: float) -> torch.Tensor:
    x = (r / r_cut).clamp(min=0.0, max=1.0)
    return 0.5 * (torch.cos(math.pi * x) + 1.0) * (r <= r_cut).to(r.dtype)

@torch.jit.script
def _cos_env(r: torch.Tensor, r_cut: float) -> torch.Tensor:
    x = (r / r_cut).clamp(0.0, 1.0)
    return 0.5 * (torch.cos(math.pi * x) + 1.0)

@torch.jit.script
def _safe_sinc(x: torch.Tensor) -> torch.Tensor:
    # sinc(x) = sin(x)/x with lim=1 at 0
    out = torch.empty_like(x)
    eps = 1e-12
    mask = x.abs() < eps
    out[mask] = 1.0
    x_nz = x[~mask]
    out[~mask] = torch.sin(x_nz) / x_nz
    return out

# spherical Bessel j0(x) = sin(x)/x
@torch.jit.script
def _sph_j0(x: torch.Tensor) -> torch.Tensor:
    return _safe_sinc(x)

# Different encodings for the atom distance embeddings
class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(torch.nn.Module):
    def __init__(
        self, start=-5.0, stop=5.0, num_sigmoid=50, basis_width_scalar=1.0
    ) -> None:
        super().__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist)


class LinearSigmoidSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_sigmoid: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_sigmoid
        offset = torch.linspace(start, stop, num_sigmoid)
        self.coeff = (basis_width_scalar / (offset[1] - offset[0])).item()
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        exp_dist = self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))
        return torch.sigmoid(exp_dist) + 0.001 * exp_dist


class SiLUSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_output: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_output
        self.fc1 = nn.Linear(2, num_output)
        self.act = nn.SiLU()

    def forward(self, dist):
        x_dist = dist.view(-1, 1)
        x_dist = torch.cat([x_dist, torch.ones_like(x_dist)], dim=1)
        return self.act(self.fc1(x_dist))


class CoulombSturmianSmearing(nn.Module):
    def __init__(
        self,
        num_radial: int,
        r_cut: float,
        p: float = 1.0,
        a: float | None = None,
        alpha_min: float | None = None,  # default: 1/r_cut
        alpha_max: float | None = None,  # default: 16/r_cut
        learn_alpha: bool = True,
        normalize: bool = True,
        envelope: str = "cos",
    ) -> None:
        super().__init__()
        assert num_radial >= 1
        self.num_output = num_radial
        self.r_cut = float(r_cut)
        self.p = float(p)
        self.a_fixed = None if a is None else float(a)
        self.normalize = normalize
        self.envelope = envelope

        a_min = (alpha_min if alpha_min is not None else 1.0 / self.r_cut)
        a_max = (alpha_max if alpha_max is not None else 16.0 / self.r_cut)
        grid = torch.logspace(math.log10(a_min), math.log10(a_max), steps=num_radial)
        self.log_alpha = nn.Parameter(grid.log(), requires_grad=learn_alpha)

        self.post = nn.Linear(num_radial, num_radial, bias=True)
        nn.init.eye_(self.post.weight)
        nn.init.zeros_(self.post.bias)

        with torch.no_grad():
            r = torch.linspace(0.0, self.r_cut, steps=1024)
            B = self._eval_basis(r)
            W = _smooth_envelope_cos(r, self.r_cut).unsqueeze(-1)
            Phi = B * W
            w = (r ** 2).unsqueeze(-1)
            G_diag = (Phi * Phi * w).sum(0) + 1e-8
            scale = (1.0 / torch.sqrt(G_diag)).clamp(max=1e3)
            self.register_buffer("whiten_scale", scale)

    def _eval_basis(self, r: torch.Tensor, a_override: float | None = None) -> torch.Tensor:
        s = 2.0 * torch.exp(self.log_alpha)
        sr = r.unsqueeze(-1) * s

        a = self.a_fixed if a_override is not None else self.a_fixed
        if a is None:
            a = 2.0 * self.p - 1.0
        a = float(a)

        R, C = sr.shape
        Lmm = torch.ones_like(sr)
        diag = torch.empty_like(sr)
        diag[:, 0] = Lmm[:, 0]
        if C > 1:
            Lm = 1.0 + a - sr 
            diag[:, 1] = Lm[:, 1]
            for k in range(1, C - 1):
                Lk = ((2.0 * k + 1.0 + a - sr) * Lm - (k + a) * Lmm) / (k + 1.0)
                diag[:, k + 1] = Lk[:, k + 1]
                Lmm, Lm = Lm, Lk

        pref = (sr ** self.p) * torch.exp(-0.5 * sr)
        return pref * diag

    def _envelope(self, r: torch.Tensor) -> torch.Tensor:
        if self.envelope == "cos":
            return _smooth_envelope_cos(r, self.r_cut)
        elif self.envelope == "none":
            return torch.ones_like(r)
        else:
            raise ValueError(f"Unknown envelope {self.envelope}")

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        B = self._eval_basis(dist)
        Phi = B * self._envelope(dist).unsqueeze(-1)
        if self.normalize:
            Phi = Phi * self.whiten_scale
        return self.post(Phi)


# spherical Bessel j1(x) = (sin(x)/x^2) - (cos(x)/x)
class SphericalBesselSmearing(nn.Module):
    """
    Basis: [ j0(k_m * r) ]_{m=1..M} with learnable k_m (optional).
    """
    def __init__(
        self,
        num_basis: int,
        r_cut: float,
        k_min: float | None = None,
        k_max: float | None = None,
        learn_k: bool = False,
        envelope: bool = True,
    ):
        super().__init__()
        self.num_output = int(num_basis)
        self.r_cut = float(r_cut)
        k_min = k_min or (math.pi / r_cut)
        k_max = k_max or (64.0 * math.pi / r_cut)
        k_grid = torch.logspace(math.log10(k_min), math.log10(k_max), self.num_output)
        if learn_k:
            self.log_k = nn.Parameter(k_grid.log())
        else:
            self.register_buffer("k", k_grid)
            self.log_k = None
        self.envelope = envelope
        self.post = nn.Identity()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            d = dist.float().clamp_min(0.0).unsqueeze(-1)
            k = (self.log_k.exp() if self.log_k is not None else self.k).unsqueeze(0)
            Phi = _sph_j0(d * k)
            if self.envelope:
                Phi = Phi * _cos_env(d.squeeze(-1), self.r_cut).unsqueeze(-1)
        return self.post(Phi).to(dist.dtype)

# hankel j0 basis with learnable spectral weights
class HankelSpectralSmearing(nn.Module):
    """
    Learn spectral weights A (C x M) over fixed j0(k_m r) modes.
    """
    def __init__(self, num_output: int, r_cut: float, num_modes: int = 64,
                 k_min: float | None = None, k_max: float | None = None):
        super().__init__()
        self.num_output = int(num_output)
        self.r_cut = float(r_cut)
        k_min = k_min or (math.pi / r_cut)
        k_max = k_max or (64.0 * math.pi / r_cut)
        k_grid = torch.logspace(math.log10(k_min), math.log10(k_max), num_modes)
        self.register_buffer("k", k_grid)
        self.A = nn.Parameter(torch.zeros(self.num_output, num_modes))
        nn.init.normal_(self.A, std=0.02)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            d = dist.float().clamp_min(0.0).unsqueeze(-1)
            j0 = _sph_j0(d * self.k.unsqueeze(0))
            Phi = j0 @ self.A.t()
            Phi = Phi * _cos_env(d.squeeze(-1), self.r_cut).unsqueeze(-1)
        return Phi.to(dist.dtype)

# laplace basis with learnable mix weights
class LaplaceMixSmearing(nn.Module):
    """
    g_c(r) = r^p * sum_j w_{c,j} * exp(-alpha_{c,j} r) * envelope(r)
    """
    def __init__(self, num_output: int, r_cut: float, num_mix: int = 8,
                 p: float = 1.0, learn_alpha: bool = True):
        super().__init__()
        self.num_output = int(num_output)
        self.r_cut = float(r_cut)
        self.p = float(p)
        self.w = nn.Parameter(torch.zeros(self.num_output, num_mix))
        a0 = torch.log(torch.linspace(1.0/r_cut, 16.0/r_cut, num_mix))
        self.log_alpha = nn.Parameter(a0.expand(self.num_output, num_mix).clone(),
                                      requires_grad=learn_alpha)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            r = dist.float().clamp_min(0.0).unsqueeze(-1).unsqueeze(-1)
            alpha = self.log_alpha.exp().unsqueeze(0)
            w = self.w.unsqueeze(0)
            Phi = (r ** self.p) * torch.exp(-alpha * r)
            Phi = (Phi * w).sum(-1)
            env = _cos_env(dist.float(), self.r_cut).unsqueeze(-1)
            Phi = Phi * env
        return Phi.to(dist.dtype)


# RadialMLP over a seed featurizer (default: Gaussians)
class GaussianSmearing(nn.Module):
    def __init__(self, start: float, stop: float, num_gaussians: int, basis_width_scalar: float = 1.0):
        super().__init__()
        self.num_output = int(num_gaussians)
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1]-offset[0]).item())**2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        d = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * d.pow(2))

class RadialMLP(nn.Module):
    """
    Wrap any seed featurizer (default: Gaussians) with a small MLP.
    Output dim = num_output (set equal to seed.num_output unless you pass proj_out).
    """
    def __init__(self,
                 seed: nn.Module | None,
                 r_cut: float,
                 num_output: int | None = None,
                 hidden: int = 128,
                 layers: int = 2,
                 proj_out: int | None = None,
                 use_layernorm: bool = True):
        super().__init__()
        self.seed = seed or GaussianSmearing(0.0, r_cut, 128, 2.0)
        in_dim = self.seed.num_output
        out_dim = proj_out if proj_out is not None else (num_output or in_dim)
        self.num_output = int(out_dim)
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mlp = []
        for i in range(len(dims)-2):
            mlp += [nn.Linear(dims[i], dims[i+1]), nn.SiLU()]
            if use_layernorm:
                mlp += [nn.LayerNorm(dims[i+1])]
        mlp += [nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*mlp)
        self.r_cut = float(r_cut)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        Phi0 = self.seed(dist)
        Phi = self.mlp(Phi0)
        Phi = Phi * _cos_env(dist, self.r_cut).unsqueeze(-1)
        return Phi

# Chebyshev/DCT 1D spectral operator (operator learning)
# idea taken from 10.48550/arxiv.2108.08481
class ChebyshevRadialOperator(nn.Module):
    """
    Build C channels as IDCT of learnable spectral weights on a fixed grid,
    then linearly interpolate at edge distances.
    """
    def __init__(self, num_output: int, r_cut: float, grid_size: int = 128, modes: int = 64,
                 layers: int = 1, channel_mix: bool = True):
        super().__init__()
        assert modes <= grid_size, "modes must be <= grid_size"
        self.num_output = int(num_output)
        self.r_cut = float(r_cut)
        self.N = int(grid_size)
        self.K = int(modes)
        r = torch.linspace(0.0, self.r_cut, self.N)
        self.register_buffer("r_grid", r)
        n = torch.arange(self.N, dtype=torch.float32).unsqueeze(1) + 0.5
        k = torch.arange(self.K, dtype=torch.float32).unsqueeze(0)
        B = torch.cos(math.pi / self.N * (n @ k))
        s = torch.ones(self.K); s[0] = 1.0 / math.sqrt(2.0)
        self.register_buffer("B", B * s)
        self.spec = nn.ParameterList([nn.Parameter(torch.zeros(self.num_output, self.K)) for _ in range(layers)])
        for w in self.spec:
            nn.init.normal_(w, std=0.02)
        self.mix = nn.Linear(self.num_output, self.num_output, bias=False) if channel_mix else nn.Identity()

    @staticmethod
    def _interp1(r_q: torch.Tensor, r_grid: torch.Tensor, f_grid: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(r_grid, r_q.clamp(0, r_grid[-1]-1e-12))
        idx1 = idx.clamp(min=1)
        idx0 = idx1 - 1
        r0 = r_grid[idx0]; r1 = r_grid[idx1]
        t = ((r_q - r0) / (r1 - r0 + 1e-12)).unsqueeze(-1)
        f0 = f_grid[idx0]; f1 = f_grid[idx1]
        return f0 + t * (f1 - f0)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            G = (self.B @ self.spec[0].t())
            for i in range(1, len(self.spec)):
                G = G + (self.B @ self.spec[i].t())
            G = self.mix(G)
            env = _cos_env(self.r_grid, self.r_cut).unsqueeze(-1)
            G = G * env
            Phi = self._interp1(dist.float(), self.r_grid, G)
        return Phi.to(dist.dtype)
