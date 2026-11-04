"""Evaluation: accuracy, curl, predictor-as-latent-response-model, cost, NVE MD, MTS.

Everything runs in float64 on a copy of the model.
"""
import copy
import math
import time
import torch

import lj
from model import geometry


def f64(model):
    return copy.deepcopy(model).double().eval()


@torch.no_grad()
def accuracy(model, te, bs=250):
    eE, eF, eFc = [], [], []
    for i in range(0, len(te["pos"]), bs):
        pos, E, F = te["pos"][i:i + bs], te["E"][i:i + bs], te["F"][i:i + bs]
        Ec, Fc = model.conservative_forces(pos)
        Fd = model.forces(pos)
        eE.append((Ec - E).abs())
        eF.append((Fd - F).abs().flatten(1))
        eFc.append((Fc - F).abs().flatten(1))
    return {"E_mae_per_atom": torch.cat(eE).mean().item() / te["pos"].shape[1],
            "F_mae": torch.cat(eF).mean().item(),               # inference force
            "F_mae_gradE": torch.cat(eFc).mean().item()}        # -grad of energy head


def curl(model, te, n=48):
    """Jacobian asymmetry  ||J - J^T||_F / ||J||_F  of the inference force field
    (J_ab = dF_a/dr_b).  Zero for any conservative field."""
    vals = []
    for p in te["pos"][:n]:
        J = torch.autograd.functional.jacobian(
            lambda x: model.forces(x.view(1, -1, 3)).flatten(), p.flatten())
        vals.append(((J - J.T).norm() / (2 * J.norm())).item())
    return {"curl_asym": sum(vals) / len(vals)}


def latent_response(model, te, deltas=(0.01, 0.03, 0.1), n=200, seed=0):
    """How well does the predictor act as a cheap model of the response to dr?
    Compares predicted dE = E(P(h,dr)) - E(h) with the true LJ dE, and with
    re-encoding E(enc(S+dr)) - E(h)."""
    if model.pred is None:
        return {}
    g = torch.Generator().manual_seed(seed)
    pos = te["pos"][:n]
    geo = geometry(pos)
    out = {}
    with torch.no_grad():
        h = model.enc(pos, geo)
        E0 = model.energy_from_latent(h)
        Et0 = lj.lj_energy_forces(pos)[0]
        for d in deltas:
            dr = d * torch.randn(pos.shape, generator=g, dtype=pos.dtype)
            true = lj.lj_energy_forces(pos + dr)[0] - Et0
            pred = model.energy_from_latent(model.pred(h, geo, dr)) - E0
            reenc = model.energy(pos + dr) - E0
            tgt = model.enc(pos + dr) - h
            lat = ((model.pred(h, geo, dr) - h - tgt).norm(dim=-1) / tgt.norm(dim=-1).clamp_min(1e-12)).mean()
            out[f"dE_mae_pred@{d}"] = (pred - true).abs().mean().item()
            out[f"dE_mae_reenc@{d}"] = (reenc - true).abs().mean().item()
            out[f"dE_true_mean_abs@{d}"] = true.abs().mean().item()
            out[f"latent_rel_err@{d}"] = lat.item()
    # "latent force": -d/d(dr) E(P(h, dr)) at dr = 0 -- backprop through predictor only
    with torch.enable_grad():
        dr = torch.zeros_like(pos, requires_grad=True)
        e = model.energy_from_latent(model.pred(h, geo, dr)).sum()
        (g_,) = torch.autograd.grad(e, dr)
    out["F_mae_latent"] = (-g_ - te["F"][:n]).abs().mean().item()
    return out


def timing(model, te, bs=32, reps=50, k=8):
    pos = te["pos"][:bs]
    def t(fn):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return 1e3 * (time.perf_counter() - t0) / reps
    out = {}
    with torch.no_grad():
        if model.kind != "conservative":
            out["ms_direct_force"] = t(lambda: model.forces(pos))
        out["ms_encode"] = t(lambda: model.enc(pos))
        if model.pred is not None:
            geo = geometry(pos)
            h = model.enc(pos, geo)
            drs = 0.01 * torch.randn(k, *pos.shape, dtype=pos.dtype)
            def via_pred():
                ctx = model.pred.prepare(model.enc(pos, geo), geo)
                return [model.energy_from_latent(model.pred.apply(ctx, d)) for d in drs]
            out[f"ms_{k}_displacements_predictor"] = t(via_pred)
            out[f"ms_{k}_displacements_reencode"] = t(lambda: [model.energy(pos + d) for d in drs])
    out["ms_conservative_force"] = t(lambda: model.conservative_forces(pos))
    return out


# --------------------------------------------------------------------------- MD
def _init_v(pos, T, g):
    v = torch.randn(pos.shape, generator=g, dtype=pos.dtype) * math.sqrt(T)
    return v - v.mean(1, keepdim=True)


def _dmin(pos):
    d = (pos[:, :, None] - pos[:, None]).norm(dim=-1)
    d = d + 1e3 * torch.eye(pos.shape[1], dtype=pos.dtype)
    return torch.nan_to_num(d, nan=0.0).amin((1, 2))


def _record(pos, v):
    E, _ = lj.lj_energy_forces(pos)
    ke = 0.5 * (v ** 2).sum((1, 2))
    return E + ke, ke


def run_md(force_fn, pos, T=0.2, dt=0.005, nsteps=20000, every=100, seed=0,
           slow_fn=None, mts=1):
    """Velocity Verlet NVE (mts=1) or RESPA multiple time stepping:
    fast = force_fn (direct head) every dt; slow correction = slow_fn - force_fn
    applied as an impulse every `mts` steps."""
    g = torch.Generator().manual_seed(seed)
    v = _init_v(pos, T, g)
    pos = pos.clone()
    alive = torch.ones(len(pos), dtype=torch.bool)
    H, KE = [], []
    with torch.no_grad():
        F = force_fn(pos)
        Fs = (slow_fn(pos) - F) if slow_fn else None
        for t in range(nsteps):
            if slow_fn and t % mts == 0:
                v = v + 0.5 * mts * dt * Fs
            v = v + 0.5 * dt * F
            pos = pos + dt * v
            F = force_fn(pos)
            v = v + 0.5 * dt * F
            if slow_fn and (t + 1) % mts == 0:
                Fs = slow_fn(pos) - F
                v = v + 0.5 * mts * dt * Fs
            bad = ~torch.isfinite(pos).all(-1).all(-1) | lj.evaporated(pos, 10.0) | (v.abs().amax((1, 2)) > 50) | (_dmin(pos) < 0.6)
            alive &= ~bad
            if (t + 1) % every == 0:
                h, ke = _record(torch.nan_to_num(pos), torch.nan_to_num(v))
                H.append(h); KE.append(ke)
    H, KE = torch.stack(H, 1), torch.stack(KE, 1)
    n = pos.shape[1]
    time_axis = torch.arange(1, H.shape[1] + 1, dtype=H.dtype) * every * dt
    out = {"alive_frac": alive.float().mean().item()}
    if alive.any():
        Ha = H[alive] / n
        tc = time_axis - time_axis.mean()
        slope = ((Ha - Ha.mean(1, keepdim=True)) * tc).sum(1) / (tc ** 2).sum()
        dof = 3 * n - 3
        Tk = 2 * KE[alive] / dof
        out.update({
            "drift_per_atom_per_tau": slope.mean().item(),       # true-H drift
            "abs_drift_per_atom_per_tau": slope.abs().mean().item(),
            "T_start": Tk[:, :10].mean().item(),
            "T_end": Tk[:, -10:].mean().item(),
            "H_std_per_atom": Ha.std(1).mean().item(),
        })
    return out


def md_suite(model, te, n_traj=32, nsteps=20000, dt=0.005, mts=8, T=0.2):
    pos = te["pos"][-n_traj:]
    res = {"nve": run_md(model.forces, pos, T=T, dt=dt, nsteps=nsteps)}
    if model.kind != "conservative":
        slow = lambda p: model.conservative_forces(p)[1]
        res[f"mts{mts}"] = run_md(model.forces, pos, T=T, dt=dt, nsteps=nsteps,
                                  slow_fn=slow, mts=mts)
    return res


def reference_md(te, n_traj=32, nsteps=20000, dt=0.005, T=0.2):
    pos = te["pos"][-n_traj:]
    return run_md(lambda p: lj.lj_energy_forces(p)[1], pos, T=T, dt=dt, nsteps=nsteps)
