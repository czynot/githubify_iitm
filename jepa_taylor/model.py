"""Invariant message-passing encoder, energy head, direct (non-conservative) force
head, and an action-conditioned latent predictor whose "action" is an atomic
displacement field dr.

All tensors are dense and batched: pos (B, N, 3), latents h (B, N, F).
"""
import copy
import math
import torch
import torch.nn as nn

RC = 3.0       # cutoff (sigma); LJ13 fits inside so the graph is effectively complete
N_RBF = 24


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def geometry(pos):
    """Pairwise geometry. Returns unit vectors u_ij=(r_i-r_j)/d, distances, rbf, cutoff*mask."""
    n = pos.shape[1]
    rij = pos[:, :, None, :] - pos[:, None, :, :]
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    d = torch.sqrt((rij ** 2).sum(-1) + eye)                      # diag -> 1, masked anyway
    u = rij / d[..., None]
    mu = torch.linspace(0.7, RC, N_RBF, dtype=pos.dtype, device=pos.device)
    gamma = 1.0 / (mu[1] - mu[0]) ** 2
    rbf = torch.exp(-gamma * (d[..., None] - mu) ** 2)
    fc = 0.5 * (torch.cos(math.pi * d / RC) + 1.0) * (d < RC).to(pos.dtype) * (1 - eye)
    return u, d, rbf, fc


class Interaction(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.filt = mlp(N_RBF, f, f)
        self.pair = mlp(2 * f, f, f)
        self.upd = mlp(f, f, f)
        self.norm = nn.LayerNorm(f)

    def forward(self, h, rbf, fc):
        n = h.shape[1]
        hi = h[:, :, None, :].expand(-1, -1, n, -1)
        hj = h[:, None, :, :].expand(-1, n, -1, -1)
        m = self.pair(torch.cat([hi, hj], -1)) * self.filt(rbf) * fc[..., None]
        return h + self.upd(self.norm(m.sum(2)))


class Encoder(nn.Module):
    def __init__(self, f=64, layers=3):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(f) * 0.1)
        self.layers = nn.ModuleList(Interaction(f) for _ in range(layers))

    def forward(self, pos, geo=None):
        u, d, rbf, fc = geo if geo is not None else geometry(pos)
        h = self.emb.expand(pos.shape[0], pos.shape[1], -1)
        for layer in self.layers:
            h = layer(h, rbf, fc)
        return h


class EnergyHead(nn.Module):
    """E = sum_i e(h_i); works on encoder latents and on predicted latents alike."""
    def __init__(self, f=64):
        super().__init__()
        self.net = mlp(f, f, 1, n=3)

    def forward(self, h):
        return self.net(h).squeeze(-1).sum(-1)


class DirectForceHead(nn.Module):
    """F_i = sum_j c(h_i, h_j, d_ij) u_ij with c symmetric in (i, j).

    Rotation-equivariant and momentum-conserving, but NOT the gradient of any
    energy: c depends on the many-body environment through h, so curl F != 0.
    """
    def __init__(self, f=64):
        super().__init__()
        self.net = mlp(2 * f + N_RBF, f, 1, n=3)

    def forward(self, h, geo):
        u, d, rbf, fc = geo
        n = h.shape[1]
        hi = h[:, :, None, :].expand(-1, -1, n, -1)
        hj = h[:, None, :, :].expand(-1, n, -1, -1)
        c = self.net(torch.cat([hi + hj, hi * hj, rbf], -1)).squeeze(-1) * fc
        return (c[..., None] * u).sum(2)


class DisplacementPredictor(nn.Module):
    """Action-conditioned latent predictor  h' = P(h, S, dr).

    The action dr (B,N,3) enters through invariant edge features
        a1 = (dr_i - dr_j) . u_ij           (first-order change of d_ij)
        a2 = |dr_i - dr_j|^2 - a1^2          (transverse, second order)
    Per structure we compute once a learned *response tensor*
        C_ij = coef(h_i, h_j, rbf_ij)  in R^{3 x F}
    (`prepare`).  Every displacement then costs only a contraction plus
    per-atom MLPs (`apply`):
        m_i  = sum_j fc_ij [a1, a1^2, a2]_ij . C_ij
        m_i += sum_j fc_ij g(m_j)                      (one cheap extra hop)
        h'_i = h_i + s * (out(h_i, m_i) - out(h_i, 0))
    so P(h, S, 0) = h exactly: a latent Taylor expansion in the displacement,
    and K displacements of one structure share a single encoder pass and a
    single response-tensor evaluation.
    """
    def __init__(self, f=64, scale=0.02):
        super().__init__()
        self.scale = scale
        self.f = f
        self.coef = mlp(2 * f + N_RBF, f, 3 * f)
        self.hop = mlp(f, f, f)
        self.out = mlp(2 * f, f, f)

    def prepare(self, h, geo):
        u, d, rbf, fc = geo
        n = h.shape[1]
        hi = h[:, :, None, :].expand(-1, -1, n, -1)
        hj = h[:, None, :, :].expand(-1, n, -1, -1)
        C = self.coef(torch.cat([hi + hj, hi * hj, rbf], -1)).view(*fc.shape, 3, self.f)
        base = self.out(torch.cat([h, torch.zeros_like(h)], -1))
        return h, u, fc, C, base

    def apply(self, ctx, dr):
        h, u, fc, C, base = ctx
        dij = dr[:, :, None, :] - dr[:, None, :, :]
        a1 = (dij * u).sum(-1) / self.scale
        a2 = (dij ** 2).sum(-1) / self.scale ** 2 - a1 ** 2
        act = torch.stack([a1, a1 ** 2, a2], -1) * fc[..., None]        # (B,N,N,3)
        m = torch.einsum("bijk,bijkf->bif", act, C)
        g = self.hop(m) - self.hop(torch.zeros_like(m[:, :1]))           # g(0) = 0
        m = m + torch.einsum("bij,bjf->bif", fc, g)
        return h + self.scale * (self.out(torch.cat([h, m], -1)) - base)

    def forward(self, h, geo, dr):
        return self.apply(self.prepare(h, geo), dr)


class Model(nn.Module):
    """kind in {direct, piwsl, jepa, conservative}."""
    def __init__(self, kind, f=64, layers=3, e_mean=0.0, e_std=1.0, f_std=1.0):
        super().__init__()
        self.kind = kind
        self.enc = Encoder(f, layers)
        self.ehead = EnergyHead(f)
        self.fhead = DirectForceHead(f) if kind != "conservative" else None
        self.pred = DisplacementPredictor(f) if kind == "jepa" else None
        self.register_buffer("e_mean", torch.tensor(float(e_mean)))   # per atom
        self.register_buffer("e_std", torch.tensor(float(e_std)))
        self.register_buffer("f_std", torch.tensor(float(f_std)))
        self.target_enc = None
        if kind == "jepa":
            self.target_enc = copy.deepcopy(self.enc).requires_grad_(False)

    # physical-unit heads -------------------------------------------------
    def energy_from_latent(self, h):
        return self.ehead(h) * self.e_std + self.e_mean * h.shape[1]

    def direct_forces(self, h, geo):
        return self.fhead(h, geo) * self.f_std

    def energy(self, pos):
        return self.energy_from_latent(self.enc(pos))

    def forces(self, pos):
        """Inference-time force: direct head (no backprop) unless conservative."""
        if self.kind == "conservative":
            return self.conservative_forces(pos)[1]
        geo = geometry(pos)
        return self.direct_forces(self.enc(pos, geo), geo)

    def conservative_forces(self, pos):
        with torch.enable_grad():
            p = pos if pos.requires_grad else pos.detach().requires_grad_(True)
            E = self.energy(p)
            (g,) = torch.autograd.grad(E.sum(), p,
                                     create_graph=self.training or pos.requires_grad)
        return E, -g

    @torch.no_grad()
    def ema_update(self, decay):
        for pt, po in zip(self.target_enc.parameters(), self.enc.parameters()):
            pt.mul_(decay).add_(po.detach(), alpha=1 - decay)
