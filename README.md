# Improving CT Workflows with Autodiff

Differentiable **beam-hardening correction** for X-ray CT. A polychromatic
forward model is fit to a measured sinogram by gradient descent (PyTorch
autograd), then the object is re-projected *monochromatically* to produce a
beam-hardening-free reconstruction.

> This file is the canonical project summary — read it first to re-prime context.

---

## 1. What the project does (physics)

A real X-ray source is **polychromatic**: photons span many energies, and the
linear attenuation μ depends on energy. Low-energy photons are absorbed faster,
so the beam "hardens" as it passes through matter → the reconstructed attenuation
*cups* (edges brighter than the centre) and materials are mis-valued.

The forward model (per detector ray, indices: `e`=energy bin, `m`=material,
`a`=angle, `p`=pixel):

```
I_sim(a,p) = Σ_e  I_e · exp( − Σ_m μ(e,m) · ℓ_m(a,p) )      # polychromatic Beer–Lambert
A_sim(a,p) = − log( I_sim / Σ_e I_e )                       # measured-style log-attenuation
```

- `I_e`  — source fluence in energy bin `e`  (learnable)
- `μ(e,m)` — linear attenuation [cm⁻¹] of material `m` at energy `e`  (learnable)
- `ℓ_m`  — path length [cm] through material `m` (forward-projected soft mask × voxel size)
- The sum over materials happens **inside** the exponent; the sum over energy bins
  happens **outside** (spectrum integration). `I_0 = Σ_e I_e`.

**Correction** uses the fluence-weighted effective attenuation (the thin-object
initial slope), which is the monochromatic-equivalent:

```
μ_eff(m) = Σ_e I_e · μ(e,m) / Σ_e I_e
A_mono(a,p) = Σ_m μ_eff(m) · ℓ_m(a,p)        # linear → reconstructs without cupping
```

Material path lengths come from a **soft segmentation** of the reconstruction
(`_s`): cumulative `tanh` thresholds (steepness `gamma`) split the image into
material masks. Thresholds `t` are initialised by multi-Otsu and then learned.

---

## 2. Repository layout

```
main_2d.py                         ← 2-D entry point (PRIMARY focus)
main.py                            ← 3-D entry point (analogous)
ct_autodiff/engine/                ← tiny autodiff-workflow framework
  block.py        Block     : holds Parameters; exposes them as attributes
  parameter.py    Parameter : torch.Tensor wrapper (name, trainable flag)
  workflow.py     Workflow  : ordered dict of Blocks; .parameters(), .to()
workflows/
  beam_hardening_correction_2d/    ← 2-D pipeline (validation target)
    workflow.py        BeamHardeningCorrectionWorkflow2D (orchestrates everything)
    blocks.py          ProjectionData2D, Reconstruct2D, CorrectProjection, SpectralProjection2D
    isp_2d.py          ISP2D — the differentiable polychromatic model (_compute_I_sim, _s, mono)
    barba_2d_phantom.py  phantom + ASTRA FP/FBP + calculate_I_2d (data generator)
    objective_func.py  PhiLoss (MSE between A_sim and A_meas)
    plotting.py        result figures
    utils.py           tanh_thresholding
  beam_hardening_correction/       ← 3-D pipeline (same design, volumes instead of slices)
```

**Engine note:** `SpectralProjection2D(Block, ISP2D)` mixes the engine `Block`
with the `torch.nn.Module` model. Learnable tensors are accessed as `self.I`,
`self.mu`, `self.t`, `self.gamma` → resolved by `Block.__getattr__` to the
`_params` dict (the tensors the optimizer actually updates). The underscore
`nn.Parameter`s (`self._I` …) are only used to seed `_params` and are otherwise
inert.

---

## 3. 2-D data flow (`BeamHardeningCorrectionWorkflow2D.run`)

```
ProjectionData2D.execute()
    → measured sinogram A_meas  (n_angles, size)         [polychromatic, beam-hardened]
    → also writes energy_bins.npy / fluence.npy / mu_values.npy  (seed ISP2D)

Reconstruct2D.execute(A_meas)
    → original_reconstruction  (size, size)              [FBP, shows cupping]

repeat outer_iters times (default 3), starting from original_reconstruction:
    _optim_loop:  Adam fits ISP2D params (I, mu, t) so A_sim ≈ A_meas   (PhiLoss = MSE)
                  (path lengths segmented from the *current* recon)
    ISP2D.compute_monochromatic_sinogram(current_recon)
        → mono sinogram (n_angles, size)                 [linearised, no hardening]
    Reconstruct2D.execute(mono sinogram)
        → corrected recon  (size, size)  → becomes current_recon for next pass

final_reconstruction = last pass's corrected recon
```

**Iterative correction** (`outer_iters`, default 3): each pass re-segments the
latest (de-cupped) recon, so the path lengths driving the fit keep improving.
Set `outer_iters=1` for the old single-pass behaviour. Only `I`, `mu`, `t` are
optimised; the per-pass recon is otherwise treated as a fixed input.

---

## 4. How to run

```bash
# from the repo root
python main_2d.py        # 2-D (primary)
python main.py           # 3-D
```

Outputs are written next to the script (repo root). Paths are
`__file__`-relative — no machine-specific paths.

### Dependencies
`requirments.txt` (note the spelling) now lists the full pip-installable set:

```
torch        # autograd; CUDA build recommended (cu118 confirmed working)
spekpy       # X-ray source spectrum
xraylib      # mass attenuation coefficients
scikit-image # threshold_multiotsu
numpy, matplotlib, plotly, tqdm
```

`astra-toolbox` (parallel-beam FP_CUDA / FBP_CUDA, **REQUIRES an NVIDIA GPU**) is
*not* in the pip file — it has no Windows wheels.

ASTRA has no Windows pip wheels — install via conda:
`conda install -c astra-toolbox -c nvidia astra-toolbox`.
`main_2d.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` to dodge an OpenMP clash common on
conda/Windows.

**Without ASTRA the pipeline cannot run** (FP/FBP are ASTRA-CUDA). The pure-physics
math and the plotting functions can still be exercised in isolation.

---

## 5. Output figures (what each PNG shows)

Produced by `main_2d.py`:

| File | Function | Panels / meaning |
|------|----------|------------------|
| `optimization_history_2d.png` | inline | Loss (MSE of A_sim vs A_meas) per Adam step, log-y. Should descend & flatten = forward model fitted. |
| `original_reconstruction_2d.png` | `plot_reconstruction_2d` | (1) beam-hardened FBP image; (2) horizontal centre-line profile — **look for cupping** (dip in the middle of a uniform region). |
| `final_reconstruction_2d.png` | `plot_reconstruction_2d` | Same two panels for the **corrected** reconstruction — profile should be flatter, material values more uniform. |
| `comparison_2d.png` | `plot_comparison_2d` | 4 panels: (1) ground-truth phantom (air/PMMA/Al colour map); (2) original (beam-hardened) recon; (3) corrected recon — (2)&(3) share grayscale; (4) centre-line profiles, original (blue) vs corrected (orange dashed), with the phantom material steps overlaid on a right-hand axis. Shows cupping removal vs truth. |
| `comparison_threshold_2d.png` | `plot_segmentation_comparison_2d` | 4 panels: (1) ground-truth segmentation; (2) original recon hard-segmented with the **learned ISP thresholds**; (3) corrected recon segmented the same way; (4) centre-line material labels (truth / original / corrected). Reveals beam-hardening **mis-segmentation** (e.g. PMMA centre wrongly labelled air) and whether the correction fixes it. |

Running `barba_2d_phantom.py` directly (standalone demo) also produces
`phantom_2d.png` (material map + profile), `sinogram_2d.png` (the −log(I/I₀)
sinogram + a line profile), and `reconstruction_2d.png` (FBP + profile + phantom).

---

## 6. Key parameters & known caveats

- **`dk` (spectrum bin width, keV)** in `ProjectionData2D` — controls the number
  of energy bins. `dk=50` → only **2 bins** (~0.6% hardening, looks like nothing to
  correct); `dk≈1` → ~119 bins and realistic hardening (~69%). Hardening saturates
  fast: `dk=5` already gives ~64%. **Default is now `dk=10`.** The data generator and
  the model share the same bins (auto-detected from `fluence.npy`), so the fit stays
  self-consistent at any `dk`. Bin count barely affects runtime — ASTRA FP dominates,
  the energy sum is a cheap chunked einsum.
- **`gamma` (tanh steepness, default 100) now works in a normalised space.** `_s`
  rescales the recon to `[0,1]` (min/max) *before* thresholding and learns `t` in that
  same `[0,1]` space, so `gamma` is decoupled from the physical recon scale
  (~0.005–0.02) and `gamma=100` yields crisp masks (transition width `1/gamma=0.01`).
  The `[diag]` line now reports thresholds + transition width in `[0,1]` units and warns
  if the width still exceeds the smallest threshold gap.
- **Per-parameter learning rates:** `_build_optimizer` gives each parameter group
  (`I`, `mu`, `t`) an Adam `lr = base_lr · mean|param|`, i.e. a uniform *fractional*
  step. With one absolute lr, `I` (~1e5) was effectively frozen while `t` moved; now all
  three move proportionally. **Caveat:** because `I`/`mu` init at ground truth (below),
  letting them move can *drift* `μ_eff` away from truth on this rigged setup — for a
  clean test, freeze `I`/`mu` and learn only `t`, or init them away from truth.
- **Validation caveat:** ISP2D initialises `I` and `mu` from the *same* spectrum/μ
  used to generate the data, so optimisation **starts at ground truth** for the
  spectral parameters (only `t` is learned from scratch). "Loss goes down" therefore
  does not prove the method can recover an *unknown* spectrum. For an honest test,
  init `I`/`μ` away from truth or freeze them and learn only `t`.
- **Threshold ordering:** `_s` sorts `t` every forward pass so the exclusive-mask
  subtraction can't go negative if Adam reorders the thresholds.
- **Iterative correction (`outer_iters`, default 3):** each outer pass re-segments the
  latest corrected recon and refits, so cupping-induced mis-segmentation is reduced
  pass-over-pass. `outer_iters=1` restores the old single-pass behaviour. Total optimiser
  steps = `optim_steps · outer_iters`.
- **Reproducibility:** simulated-noise RNG is seeded (`ProjectionData2D.noise_seed`,
  default 0); the phantom is seeded at 69.

---

## 7. Verified facts (local, Windows)

- `torch 2.5.1+cu118`, CUDA available; `spekpy`, `xraylib`, `scikit-image` present;
  **`astra` NOT installed locally** — full pipeline only runs where ASTRA+CUDA exist.
- The polychromatic core was checked numerically: the data generator
  (`calculate_I_2d`) and the model (`_compute_I_sim`/`_compute_A_sim`) agree to
  floating-point zero for identical path lengths — the spectrum integration,
  multi-bin Beer–Lambert, and μ accumulation are correct.

---

## 8. Changes from the original implementation

Everything below was modified on top of the original codebase. The physics core
(`_compute_I_sim` / `_compute_A_sim` / `compute_monochromatic_sinogram`) was
verified correct and left unchanged. **All changes are reasoned + byte-compiled
but NOT yet run end-to-end locally — no ASTRA on the Windows box, so they need HPC
validation.**

### 8.1 Portability (hard-coded paths removed)
- **`main_2d.py` / `main.py`:** replaced a teammate's absolute path
  `sys.path.append("/home/s4861264/CIT_project/")` with `_HERE =
  os.path.dirname(os.path.abspath(__file__))`; all PNG outputs now save to `_HERE`.
- **3-D `blocks.py` / `isp.py`:** added `_DATA_DIR = os.path.dirname(abspath(__file__))`
  and routed every `.npy` save/load through it (was cwd-relative).
- **`main_2d.py`:** `KMP_DUPLICATE_LIB_OK=TRUE` set at import (OpenMP clash on conda/Windows).

### 8.2 Correctness / robustness fixes
- **Threshold ordering (`isp_2d.py:_s`):** `t` is `torch.sort`-ed every forward pass,
  so Adam reordering thresholds can no longer make the exclusive-mask subtraction
  `s_cum[:-1] − s_cum[1:]` go negative. (`torch.sort` is differentiable.)
- **Seeded noise (`barba_2d_phantom.py` / `blocks.py`):** `calculate_I_2d` takes a
  `seed`; `ProjectionData2D` exposes `noise_seed` (default 0) → reproducible sinograms.

### 8.3 Optimisation improvements (the "underwhelming results" work)
| Area | Original | Now |
|------|----------|-----|
| **Mask sharpness** | `tanh` thresholding on raw recon (~0.005–0.02); `gamma=100` never saturated → mushy masks | `_s` normalises recon to `[0,1]` and learns `t` there → `gamma=100` gives crisp masks (§6) |
| **Learning rate** | one Adam `lr` for `I`(~1e5), `mu`(~1), `t` → `I` effectively frozen | `_build_optimizer` per-group `lr = base_lr·mean\|param\|` (uniform fractional step) |
| **Correction passes** | single-pass: segment the beam-hardened recon once | iterative `outer_iters` (default 3): re-segment the corrected recon + refit each pass |
| **`dk` default** | `50` → 2 energy bins, ~0.6% hardening (nothing to correct) | `10` (user-set) → realistic hardening; `dk≈1`→~69% if you want more |
| **`[diag]` line** | — | warm-up diagnostic prints normalised thresholds + tanh transition width, warns if masks will be blurry |

> ⚠️ The per-parameter LR interacts with the ground-truth-init validation caveat
> (§6): now that `I`/`mu` can move, they may drift `μ_eff` away from truth on the
> rigged setup. For a clean test, freeze `I`/`mu` and learn only `t`.

### 8.4 Housekeeping
- **`requirments.txt`:** added `scikit-image`, `tqdm`; dropped unused `scipy`; kept
  `plotly` (used by 3-D phantom + notebook); documented `astra-toolbox` as conda-only.
- **`README.md`:** this file — created as the canonical re-priming summary.
