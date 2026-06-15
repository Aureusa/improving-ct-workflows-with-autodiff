# Improving CT Workflows with Autodiff

Differentiable beam-hardening correction for X-ray CT. A polychromatic
forward model is fit to a measured sinogram by gradient descent (PyTorch
autograd), then the object is re-projected *monochromatically* to produce a
beam-hardening-free reconstruction. Implemented in both 2-D (slice) and 3-D
(volume).

---

## 1. The problem (physics)

A real X-ray source is polychromatic - photons span many energies and the
linear attenuation `μ` depends on energy. Low-energy photons are absorbed
faster, so the beam "hardens" as it passes through matter. The reconstructed
attenuation then cups (a uniform material looks brighter at the rim than the
centre) and material values are wrong.

The forward model, per detector ray (`e`=energy bin, `m`=material, `a`=angle, `p`=pixel):

```
I_sim(a,p) = Σ_e  I_e · exp( − Σ_m μ(e,m) · ℓ_m(a,p) )      # polychromatic Beer-Lambert
A_sim(a,p) = − log( I_sim / Σ_e I_e )                       # log-attenuation sinogram
```

- `I_e`     — source fluence in energy bin `e`        (learnable)
- `μ(e,m)`  — linear attenuation of material `m` at energy `e`  (learnable)
- `ℓ_m`     — path length through material `m` (forward-projected soft mask × voxel size)
- `t`       — segmentation thresholds, Otsu-initialised, then learned

The pipeline fits `I`, `μ`, `t` so `A_sim ≈ A_meas` (MSE loss, Adam), then builds
a **monochromatic-equivalent** sinogram and reconstructs it. Material path
lengths come from a soft `tanh` segmentation of the reconstruction.

**Correction** (`correction_mode`):
- `residual` (default): `y_corrected = y_meas + (y_mono − y_poly)` — corrects the
  measured sinogram by the modelled hardening, keeping real detail.
- `replace`: reconstruct the fully synthetic monochromatic sinogram `y_mono`.

**Effective attenuation** `μ_eff` per material (`mu_eff_mode`):
- `lstsq` (default): least-squares regression of the polychromatic sinogram onto
  the per-material path-length sinograms — immune to spectral soft-tail inflation.
- `fluence`: spectrum-weighted average `Σ_e I_e·μ(e,m) / Σ_e I_e` — simpler but
  degrades badly under noise.

---

## 2. Repository layout

```
experiments.py          ← experiment runner (sweeps + figures) — START HERE
main_2d.py              ← single 2-D run (prints report, writes PNGs)
main.py                 ← single 3-D run (CLI) → interactive HTML viewer
plot_AL.py              ← A(L) mono-vs-poly attenuation figure
plot_clean_results.py   ← 2-D figure renderer (matplotlib, run in a subprocess)
plot_3d_interactive.py  ← 3-D interactive plotly viewer

ct_autodiff/engine/     ← tiny autodiff-workflow framework
  block.py · parameter.py · workflow.py

workflows/
  beam_hardening_correction_2d/   ← 2-D pipeline
    workflow.py · blocks.py · isp_2d.py · barba_2d_phantom.py
    objective_func.py · plotting.py · utils.py
  beam_hardening_correction/      ← 3-D pipeline (mirrors 2-D)
    workflow.py · blocks.py · isp.py · barba_3D_phantom_1.py
    objective_func.py · plotting.py

results/                ← experiment outputs (2-D PNGs, 3-D HTMLs, metric JSONs)
```

---

## 3. Setup

Requires an **NVIDIA GPU** (reconstruction uses ASTRA-CUDA).

```bash
pip install -r requirments.txt        # torch, spekpy, xraylib, scikit-image, numpy, matplotlib, plotly, tqdm
conda install -c astra-toolbox -c nvidia astra-toolbox   # might be issues on Windows!
```
`KMP_DUPLICATE_LIB_OK=TRUE` to avoid an OpenMP clash common on conda/Windows.

---

## 4. Running experiments

`experiments.py` is the main entry point. It runs the standardised sweeps and
writes results to `results/`.

```bash
python experiments.py 2d       # 2-D sweep: metrics → results/_exp_2d.json
python experiments.py fig2d    # 2-D figures → results/2d/*.png
python experiments.py 3d       # 3-D sweep: metrics + figures (HTML) in one pass
```

What each sweep covers:
- **2-D `2d`** — a `dk` hardening sweep (dk = 2, 5, 10, 20, no noise) and a noise
  sweep (0/2/5 % noise, with and without pre-smoothing) at dk=2, seed-averaged
  over 3 seeds. `fig2d` renders one representative figure per config.
- **3-D `3d`** — a `dk` sweep (dk = 5, 20, no noise, single-seed) and a noise
  sweep at dk=5 (0.02/0.05 noise, with and without pre-smoothing) seed-averaged
  over 3 seeds. Metrics and the interactive HTML are produced in the same run
  (HTML rendered for seed 0).

### Single runs

```bash
python main_2d.py              # one 2-D run: prints a report, writes PNGs to repo root
python main.py                 # one 3-D run: builds reconstruction_3d.html
python main.py --dk 2 --noise 0.05 --steps 300 --mu-eff-mode lstsq --correction-mode residual
```

### Supporting figures

```bash
python plot_AL.py                      # monochromatic vs polychromatic A(L) curves
```

---

## 5. Key parameters

| Parameter | Meaning |
|---|---|
| `dk` | Spectrum bin width [keV]. **Smaller dk → more bins → stronger hardening** (resolves the steep μ-spread). dk=2 is strong hardening, dk=20 is mild. Both pipelines run at 120 kVp. |
| `add_gaussian_noise` | Fractional Gaussian noise σ (relative to peak intensity) added to the sinogram in the **intensity** domain. `noise_seed` makes it reproducible. |
| `smooth_sigma` | Gaussian pre-smoothing [px/vox] of the reconstruction before segmentation. Protects the segmentation against noise; 0 = off. |
| `mu_eff_mode` | `lstsq` (default, robust) or `fluence`. |
| `correction_mode` | `residual` (default) or `replace`. |

---

## 6. Outputs 

- **2-D** → `results/2d/<config>.png` (comparison), `_cupping.png` (PMMA radial
  profile), `_loss.png` (optimisation loss). Metrics in `results/_exp_2d.json`.
- **3-D** → `results/3d/<config>.html` (interactive plotly viewer: volumes,
  slice slider, central cuts, profiles, histograms, loss, metrics table).
  Metrics in `results/_exp_3d.json`.


## 7. Reproducibility

All randomness is seeded: the phantom layout (fixed seed) and the simulated
sinogram noise (`noise_seed`). 
