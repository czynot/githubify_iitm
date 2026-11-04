# Displacement-conditioned JEPA with a Taylor-consistency loss

A small, self-contained test of this idea: **take the action-conditioned predictor
from V-JEPA 2, make the "action" an atomic displacement δr, and train a direct
(non-conservative) force head so that E(S+δr) − E(S) ≈ −F·δr, where the energies
come from predicted latents.** Inference uses only the direct force head, with
no backprop.

The ground truth is a 13-atom Lennard-Jones cluster (LJ13, reduced units). Exact
energies and forces are available, so we can measure force error, curl, and the
*true* energy drift in MD. The system runs on a CPU in a few hours.

## Method

```
          S ──enc──► h_S ──E-head──► E(S)                      (energy label)
                      │  └─F-head──► F_direct(S)               (force label; used at inference)
                      │
   δr ──► P(h_S, S, δr) ──► ĥ_{S+δr} ──E-head──► Ê(S+δr)
                      ▲
          S+δr ──EMA target enc──► h̄_{S+δr}   (JEPA target, stop-grad)
```

* **Encoder.** An invariant message-passing net (3 layers, 64 features, RBF distances).
* **Energy head.** E = Σᵢ e(hᵢ). It works on both encoder latents and predicted latents.
* **Direct force head.** Fᵢ = Σⱼ c(hᵢ, hⱼ, dᵢⱼ) ûᵢⱼ, with c symmetric in (i, j). The head is
  equivariant and conserves momentum, but it is not a gradient, so curl ≠ 0.
* **Displacement predictor** (`DisplacementPredictor`). The action δr is turned into invariant
  edge features a₁ = (δrᵢ−δrⱼ)·ûᵢⱼ and a₂ = |δrᵢ−δrⱼ|² − a₁², which are first- and second-order
  changes in pair geometry. A learned *response tensor* Cᵢⱼ = coef(hᵢ, hⱼ, rbfᵢⱼ) is computed
  **once per structure**. Each displacement then costs only a contraction and per-atom
  MLPs, and P(h, S, 0) = h exactly. The predictor is a latent Taylor expansion, so
  K displacements of one structure share one encoder pass.
* **Losses** for the `jepa` model:
  * energy MSE + direct-force MSE (the same for every model);
  * **JEPA**: ‖(P(h_S, δr) − h_S) − sg(h̄_{S+δr} − h̄_S)‖² / s². The EMA target encoder
    follows V-JEPA. δr has per-coordinate std s ~ logU[0.005, 0.05]σ. The predictor
    learns latent *displacement responses*, so EMA lag between online and target
    encoders cancels;
  * **Taylor consistency through the predictor**:
    E(P(h_S, +δr)) − E(P(h_S, −δr)) ≈ −2 F_direct(S)·δr, with δ = 0.01σ.

### Baselines and ablations

| config | what it is |
|---|---|
| `direct` | energy + direct-force supervision only |
| `piwsl` | PIWSL-style Taylor consistency on **re-encoded** structures: E(enc(S+δr)) − E(enc(S−δr)) ≈ −2F·δr (two extra full encoder passes per step) |
| `jepa` | this proposal |
| `conservative` | F = −∇E reference. Needs backprop at inference. |
| `jepa_noJEPA` | Taylor loss through a predictor with no JEPA anchor |
| `jepa_noTaylor` | JEPA latent loss only |
| `*_fwd` | one-sided Taylor form E(S+δr) − E(S) ≈ −F·δr, as in the original proposal |

A note on the one-sided vs central form: on LJ13 the second-order term ½δrᵀHδr is
about as large as F·δr at δ ≈ 0.02σ. The one-sided relation therefore has an O(δ²)
bias, which the model would be pushed to absorb. The central form has an O(δ³)
bias instead, and with the predictor it costs almost nothing: two `apply` calls on
one response tensor. The central form is the default, and the one-sided form is run
as an ablation.

## Evaluation

* **Accuracy.** Energy MAE per atom, MAE of the inference force (the direct head), and
  MAE of −∇E from the energy head. The last tells you whether the energy head is
  usable as a conservative potential, for example in MTS.
* **Curl.** The antisymmetry ‖J − Jᵀ‖_F / 2‖J‖_F of the force Jacobian Jₐᵦ = ∂Fₐ/∂rᵦ,
  averaged over 48 test structures. It is exactly 0 for a conservative field; the test
  harness gives 1e-16 for the `conservative` model.
* **NVE MD.** 32 trajectories, velocity Verlet, dt = 0.005τ, 20k steps (100τ),
  T₀ = 0.2. The metric is the linear drift of the **true** LJ total energy per atom per τ.
  Direct forces with curl pump energy in or out.
* **MTS** (RESPA, as in Bigi et al. 2024, *The dark side of the forces*). Direct forces
  are applied every step, and the conservative correction −∇E − F_direct every 8 steps.
  This only works well if ∇E and F_direct agree, which is exactly what the Taylor loss
  enforces.
* **The predictor as a latent response model.** Predicted ΔE for random δr
  (δ = 0.01, 0.03, 0.1) is compared with the true ΔE and with re-encoding. "Latent
  force" −∂E(P(h, δr))/∂δr at 0 is computed by backprop through the predictor only.
  Timings compare evaluating 8 displacements through the predictor with re-encoding them.

## Running

```bash
pip install torch numpy
python run.py --configs direct piwsl jepa conservative --seeds 0 1 2 --epochs 200
python run.py --configs jepa_noJEPA jepa_noTaylor jepa_fwd piwsl_fwd --seeds 0 --epochs 200
python summarize.py            # -> results.md
```

Files: `lj.py` (ground truth and data), `model.py`, `train.py`, `evaluate.py`,
`run.py`, `summarize.py`. Per-run JSON with training curves is in `results/`.

## Results

RESULTS_PLACEHOLDER
