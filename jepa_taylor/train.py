"""Training for the four variants.

  direct        : E + direct-F supervision
  piwsl         : + Taylor consistency on *re-encoded* displaced structures
                  E(enc(S+dr)) - E(enc(S-dr)) ~ -2 F(S).dr            (PIWSL-style)
  jepa          : + JEPA latent prediction  P(h_S, dr) - h_S ~ sg(h̄_{S+dr} - h̄_S)
                  + Taylor consistency through the *predictor*
                  E(P(h_S, +dr)) - E(P(h_S, -dr)) ~ -2 F(S).dr
  conservative  : F = -dE/dr reference (needs backprop at inference)

`taylor_mode="forward"` uses the one-sided E(S+dr) - E(S) ~ -F.dr instead of the
central difference (bias O(dr^2) instead of O(dr^3)).
"""
import copy
import math
import time
import torch

from model import Model, geometry


def taylor_residual(model, pos, geo, h, F, dr, mode):
    """Residual of the Taylor relation in energy units, shape (B,)."""
    if model.kind == "jepa":  # one shared response tensor for +dr and -dr
        ctx = model.pred.prepare(h, geo)
        e = lambda s: model.energy_from_latent(model.pred.apply(ctx, s * dr))
    else:  # piwsl: re-encode the displaced structure (2 extra full encoder passes)
        e = lambda s: model.energy(pos + s * dr)
    work = (F * dr).sum((1, 2))
    if mode == "central":
        return e(1.0) - e(-1.0) + 2.0 * work, 2.0
    return e(1.0) - model.energy_from_latent(h) + work, 1.0


def train(kind, data, seed=0, epochs=300, bs=32, lr=1e-3, f=64,
          w_e=1.0, w_f=1.0, w_taylor=1.0, w_jepa=1.0, taylor_delta=0.01,
          taylor_mode="central", jepa_range=(0.005, 0.05), ema=0.995, log_every=50,
          verbose=True):
    torch.manual_seed(seed)
    tr, va = data["train"], data["val"]
    n = tr["pos"].shape[1]
    e_mean = (tr["E"] / n).mean().item()
    e_std = tr["E"].std().item()
    f_std = tr["F"].pow(2).mean().sqrt().item()
    model = Model(kind, f=f, e_mean=e_mean, e_std=e_std, f_std=f_std)
    params = [p for name, p in model.named_parameters() if not name.startswith("target_enc")]
    opt = torch.optim.Adam(params, lr=lr)
    ntr = len(tr["pos"])
    steps = epochs * math.ceil(ntr / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    use_taylor = kind in ("piwsl", "jepa") and w_taylor > 0
    use_jepa = kind == "jepa" and w_jepa > 0
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr)
        agg = {}
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            pos, E, F = tr["pos"][idx], tr["E"][idx], tr["F"][idx]
            geo = geometry(pos)
            h = model.enc(pos, geo)
            Ep = model.energy_from_latent(h)
            if kind == "conservative":
                Ep, Fp = model.conservative_forces(pos)
            else:
                Fp = model.direct_forces(h, geo)
            loss_e = ((Ep - E) / e_std).pow(2).mean()
            loss_f = ((Fp - F) / f_std).pow(2).mean()
            loss = w_e * loss_e + w_f * loss_f
            terms = {"e": loss_e.item(), "f": loss_f.item()}
            if use_taylor:
                dr = taylor_delta * torch.randn_like(pos)
                res, k = taylor_residual(model, pos, geo, h, Fp, dr, taylor_mode)
                # normalise by the size of the first-order term k*|F||dr|
                loss_t = (res / (k * f_std * taylor_delta * math.sqrt(3 * n))).pow(2).mean()
                loss = loss + w_taylor * loss_t
                terms["taylor"] = loss_t.item()
            if use_jepa:
                lo, hi = math.log(jepa_range[0]), math.log(jepa_range[1])
                s = torch.exp(torch.empty(len(pos), 1, 1).uniform_(lo, hi))
                drj = s * torch.randn_like(pos)
                with torch.no_grad():
                    tgt = model.target_enc(pos + drj) - model.target_enc(pos, geo)
                # As in V-JEPA 2-AC the predictor is trained *on top of* the encoder:
                # h is detached here (the encoder is shaped by E/F and, through the
                # predictor, by the Taylor loss). The loss is relative so that it is
                # invariant to the (unconstrained) scale of the latent response.
                hd = h.detach()
                pred = model.pred(hd, geo, drj) - hd
                loss_j = (pred - tgt).pow(2).sum((1, 2)) / tgt.pow(2).sum((1, 2)).clamp_min(1e-12)
                loss_j = loss_j.mean()
                loss = loss + w_jepa * loss_j
                terms["jepa"] = loss_j.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            if use_jepa:
                model.ema_update(ema)
            for k_, v_ in terms.items():
                agg[k_] = agg.get(k_, 0.0) + v_ * len(idx) / ntr
        val = quick_val(model, va)
        score = val["e_mae"] / e_std + val["f_mae"] / f_std
        if score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
        hist.append({"epoch": ep, **agg, **val})
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            tl = " ".join(f"{k_}={v_:.4f}" for k_, v_ in agg.items())
            print(f"[{kind} s{seed}] ep {ep:4d} {tl} | val E {val['e_mae']:.4f} "
                  f"F {val['f_mae']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    model.train_time = time.time() - t0
    return model.eval(), hist


def quick_val(model, va, bs=256):
    model.eval()
    e_err, f_err = [], []
    for i in range(0, len(va["pos"]), bs):
        pos = va["pos"][i:i + bs]
        with torch.no_grad():
            if model.kind == "conservative":
                Ep, Fp = model.conservative_forces(pos)
            else:
                geo = geometry(pos)
                h = model.enc(pos, geo)
                Ep, Fp = model.energy_from_latent(h), model.direct_forces(h, geo)
        e_err.append((Ep - va["E"][i:i + bs]).abs())
        f_err.append((Fp - va["F"][i:i + bs]).abs().flatten(1))
    model.train()
    return {"e_mae": torch.cat(e_err).mean().item(), "f_mae": torch.cat(f_err).mean().item()}
