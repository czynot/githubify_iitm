"""Ground-truth system: a Lennard-Jones cluster (LJ13) in reduced units.

Gives exact energies and forces so that we can measure force errors, curl, and
energy drift in MD against a known conservative reference.
"""
import math
import torch

N_ATOMS = 13


def lj_energy_forces(pos):
    """pos: (B, N, 3) float64 -> E (B,), F (B, N, 3). Untruncated 12-6 LJ, eps=sigma=1."""
    rij = pos[:, :, None, :] - pos[:, None, :, :]           # r_i - r_j
    n = pos.shape[1]
    eye = torch.eye(n, dtype=pos.dtype, device=pos.device)
    d2 = (rij ** 2).sum(-1) + eye                            # avoid 0 on diagonal
    inv6 = (1.0 / d2) ** 3
    inv12 = inv6 ** 2
    mask = 1.0 - eye
    E = 0.5 * (4.0 * (inv12 - inv6) * mask).sum((1, 2))
    # dE/dd2 per pair -> F_i = -sum_j dE/dr_i = sum_j 24 (2 inv12 - inv6)/d2 * r_ij
    coef = 24.0 * (2.0 * inv12 - inv6) / d2 * mask
    F = (coef[..., None] * rij).sum(2)
    return E, F


def random_cluster(b, n=N_ATOMS, gen=None):
    """Rough random starting clusters (rejection sampling to avoid overlaps)."""
    out = []
    while len(out) < b:
        p = []
        while len(p) < n:
            x = (torch.rand(3, generator=gen, dtype=torch.float64) * 2 - 1) * 1.6
            if all(((x - q) ** 2).sum() > 0.9 ** 2 for q in p):
                p.append(x)
        out.append(torch.stack(p))
    return torch.stack(out)


def minimize(pos, steps=2000, lr=1e-3):
    pos = pos.clone()
    for _ in range(steps):
        _, F = lj_energy_forces(pos)
        pos = pos + lr * F.clamp(-50, 50)
    return pos


def langevin(pos, T, nsteps, dt=0.005, gamma=1.0, gen=None, sample_every=None):
    """BAOAB Langevin, unit masses. Returns final state and optional samples."""
    v = torch.randn(pos.shape, generator=gen, dtype=pos.dtype) * math.sqrt(T)
    _, F = lj_energy_forces(pos)
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt((1 - c1 ** 2) * T)
    samples = []
    for t in range(nsteps):
        v = v + 0.5 * dt * F
        pos = pos + 0.5 * dt * v
        v = c1 * v + c2 * torch.randn(pos.shape, generator=gen, dtype=pos.dtype)
        pos = pos + 0.5 * dt * v
        _, F = lj_energy_forces(pos)
        v = v + 0.5 * dt * F
        if sample_every and (t + 1) % sample_every == 0:
            samples.append(pos.clone())
    return pos, samples


def evaporated(pos, rmax=2.6):
    com = pos.mean(1, keepdim=True)
    return ((pos - com).norm(dim=-1).max(-1).values > rmax)


def make_dataset(n_rep=64, temps=(0.15, 0.25, 0.35), equil=4000, nsamp=16,
                 sample_every=250, seed=0):
    """Returns dict(pos, E, F) of decorrelated, bound LJ13 configurations."""
    gen = torch.Generator().manual_seed(seed)
    start = minimize(random_cluster(n_rep, gen=gen))
    allpos = []
    for T in temps:
        pos, _ = langevin(start.clone(), T, equil, gen=gen)
        _, samples = langevin(pos, T, nsamp * sample_every, gen=gen,
                              sample_every=sample_every)
        s = torch.cat(samples)
        allpos.append(s[~evaporated(s)])
    pos = torch.cat(allpos)
    pos = pos - pos.mean(1, keepdim=True)
    pos = pos[torch.randperm(len(pos), generator=gen)]
    E, F = lj_energy_forces(pos)
    return dict(pos=pos, E=E, F=F)
