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
python main_2d.py        # 2-D PRIMARY results — lstsq mu_eff + residual correction (§8.8)
python main_2d_clean.py  # 2-D experimentation harness — CLI flags for every regime (§8.5–8.8)
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

> **Update (§8.8):** `main_2d.py` now writes `comparison_2d.png`, `cupping_2d.png`, and
> `optimization_history_2d.png` (lstsq + residual config). The per-reconstruction and
> threshold figures in the table below come from the earlier flow, still reachable via
> `main_2d_clean.py` / the standalone phantom script.

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

### 8.5 Clean-validation harness (`main_2d_clean.py`)
A separate 2-D entry point that runs the pipeline as an *honest* test, removing the
confounders flagged in §6 so "did the correction work?" gets a readable answer:
- `add_gaussian_noise=0.0` — noise no longer buries the cupping bowl.
- `dk=2.0` — strong, unambiguous beam hardening.
- `freeze_spectral=True` — `I`/`mu` held at ground truth; only the segmentation
  thresholds `t` are learned, so the per-parameter LR cannot drift `μ_eff`.

Supporting changes (all backward-compatible — `main_2d.py` is unchanged):
- **`workflow.py`:** `BeamHardeningCorrectionWorkflow2D` gained pass-through args
  `dk`, `add_gaussian_noise`, `noise_seed`, `freeze_spectral` (previously `dk` was
  fixed at the `ProjectionData2D` default and noise was hard-coded to `0.02`).
- **`blocks.py`:** `SpectralProjection2D` gained `freeze_spectral` — registers `I`/`mu`
  as **non-trainable** so `Block.parameters()` (and thus the optimizer) only sees `t`.
- **`plotting.py`:** new `plot_cupping_validation_2d` (PMMA-only **radial** profile +
  per-material mean±std) and `compute_validation_metrics` (PMMA **cupping %** =
  100·(rim−centre)/rim, per-material **CoV**). These replace the noisy
  through-the-bubbles centre line with a metric you can actually read.

Supporting changes also include `al_filter_mm` (added Al tube filtration, on both
`ProjectionData2D` and the workflow) — see the finding below for why it matters.

Outputs are suffixed `_clean` (`comparison_2d_clean.png`, `cupping_validation_2d_clean.png`,
`optimization_history_2d_clean.png`) plus a printed report (PMMA cupping %, per-material
mean/std/CoV).

**Plotting runs in a subprocess (`plot_clean_results.py`).** On the local conda env
(`Tomography`) matplotlib's native rendering *hard-crashes* (silent, no traceback,
exit `0xC06D7FFF`) once torch+astra+MKL OpenMP are loaded in the process —
`KMP_DUPLICATE_LIB_OK` does not help. So `main_2d_clean.py` computes + prints the
report, saves the arrays, then spawns `plot_clean_results.py` (numpy+matplotlib only)
to draw the PNGs. If that env's matplotlib is also broken, run the plotter with a
different interpreter: `python plot_clean_results.py --arrays _clean_arrays`.

#### Finding (ran locally, ASTRA-CUDA + CPU-torch): the `μ_eff` soft-tail pathology
The honest test surfaced **why the correction underwhelms**: the fluence-weighted
`μ_eff = Σ_e I_e·μ(e,m) / Σ_e I_e` (README §1) is dominated by the **<20 keV soft
spectral tail**, where μ is enormous (PMMA 146, **Al 2128 cm⁻¹ at 3 keV**) but the
photons are *fully absorbed* in the object, so they contribute nothing to the measured
sinogram yet dominate the correction target.

At `dk=2`, kvp=120, **no added filtration** (spekpy keeps bins down to 3 keV; <20 keV =
25% of fluence but **94–97% of the `μ_eff` numerator**):
- `μ_eff`: PMMA **3.5**, Al **53.5 cm⁻¹** (should be ~0.25 / ~0.7) → the corrected Al
  recon **explodes ~38×** (Al 0.021→1.0, PMMA CoV 0.16→9.8). Correction **broken**.

Adding realistic tube filtration (`al_filter_mm=2.0`, an available option — **not** the
default; §8.6's transmission fix is) drops <20 keV fluence to ~1% and restores `μ_eff`
PMMA **0.28** / Al **1.64 cm⁻¹**:
- Original PMMA cupping **4.1%**, corrected **−4.3%** (mild overcorrection); Al raised
  0.021→0.031; PMMA CoV 0.094→0.152. Now **well-behaved** but only mildly corrective —
  filtration also pre-hardens the beam, leaving little cupping to remove.

**Take-aways:** (1) the fluence-weighted "thin-object initial slope" `μ_eff` is the
*wrong* monochromatic target for thick objects / soft spectra — a
**transmission-weighted** effective μ (or an explicit low-energy cutoff) would be far
more robust; (2) there's a tension — filtration fixes `μ_eff` but reduces the hardening
there is to correct, so a convincing strong-yet-physical demo needs a thicker/denser
phantom or lighter filtration (e.g. `al_filter_mm=1.0`). The unfiltered figures are kept
as `*_clean_unfiltered.png` for comparison.

### 8.6 Fix — transmission-weighted `μ_eff` (`mu_eff_mode="transmission"`)
Implements take-away (1) of §8.5. `ISP2D._effective_mu` (and the 3-D `ISP` mirror) now
supports two weightings via `mu_eff_mode` (threaded through `SpectralProjection2D` and
the workflow; **default `"fluence"` = original behaviour, so `main_2d.py`/3-D are
unchanged**):

- **`"fluence"`** (original): `μ_eff[m] = Σ_e I_e·μ(e,m) / Σ_e I_e`.
- **`"transmission"`** (new): weight each bin by the photons that actually survive a
  representative object path, `w_e = I_e·exp(−Σ_m μ(e,m)·L_rep[m])`, then
  `μ_eff[m] = Σ_e w_e·μ(e,m) / Σ_e w_e`. `L_rep[m]` is the mean path through material
  `m` over the object-intersecting rays (data-driven — no reference energy/thickness).
  Absorbed soft photons get ~zero weight, so `μ_eff` is physical *without* filtration.

**Result (local run: `dk=2`, NO filtration, ~25% hardening, `freeze_spectral`, `outer_iters=1`):**

| metric | fluence (orig) | transmission (fix) |
|---|---|---|
| `μ_eff` Al | 53.5 cm⁻¹ | ~1.5 cm⁻¹ |
| corrected Al recon | 1.0 (explodes ×38) | 0.027 (physical) |
| PMMA cupping (orig→corr) | 25.3% → **977%** | 25.3% → **7.1%** |
| PMMA CoV (orig→corr) | 0.16 → **9.8** | 0.16 → **0.11** |

`main_2d_clean.py` defaults to `mu_eff_mode="transmission"`, `al_filter_mm=0.0`,
`freeze_spectral=True`, `outer_iters=1`: it removes most of the cupping **without**
sacrificing the hardening (cf. §8.5's filtration route, which fixed `μ_eff` but
pre-hardened the cupping away). The residual ~7% is edge ringing + soft-mask
segmentation (not `μ_eff`), and **more passes do not reliably reduce it** (§8.7
outer-iters sweep). Figures kept: `*_clean.png` (the transmission fix / default),
`*_clean_unfiltered.png` (broken fluence), `*_clean_filtered.png` (fluence + 2 mm Al),
`*_clean_recovery.png` (honest recovery, §8.7).

### 8.7 Honest recovery test (`--no-freeze --perturb`) + CLI harness
`main_2d_clean.py` is now a CLI (argparse) covering every regime; `ISP2D` gained
`spectral_perturb`/`spectral_perturb_seed` (threaded through the block + workflow) to
start `I`/`mu` *away* from truth — so unfreezing actually tests recovery rather than
sitting on the ground-truth init (README §6 caveat). The harness also reports the
transmission `μ_eff` of the *learned* spectrum vs ground truth.

| run (transmission, dk=2, **outer=3**) | freeze | perturb | corrected cupping | learned μ_eff err (PMMA / Al) |
|---|---|---|---|---|
| **recovery** | **no** | **±30%** | 25.3% → **2.1%** | **9.9% / 8.3%** |
| stability | no | 0 | 25.3% → 2.0% | 9.9% / 7.8% |
| frozen reference | yes | – | 25.3% → 9.2% | 0 (frozen at truth) |

Findings (transmission mode, dk=2, no filtration):
- **The transmission `μ_eff` is identifiable.** From a ±30% perturbed start the fit
  recovers `μ_eff` and converges to *essentially the same solution* as starting at
  truth (both ≈ PMMA 0.297 / Al 1.42) — because fitting `A_meas` constrains the
  *detected-spectrum-weighted* attenuation, which is exactly the transmission `μ_eff`.
  The individual `Iₑ`/`μₑ` are NOT uniquely recovered (118-bin degeneracy) but don't
  need to be. The residual ~8–10% vs the `_trans_mu_eff` yardstick is a fixed offset of
  the converged solution (identical for perturb 0 and 0.3), not a recovery failure.
- **Unfreezing is stable** in transmission mode — no explosion / runaway drift, the
  opposite of the fluence-mode hazard warned about in §8.3.

**Outer-iters sweep — do we need the correction loop? (`--no-freeze --perturb 0.3`):**

| `outer_iters` | corrected cupping | μ_eff err (PMMA / Al) |
|---|---|---|
| **1** | 25.3% → 4.3% | **1.2% / 1.2%** |
| 2 | 25.3% → 8.6% | 66.9% / 29.0% |
| 3 | 25.3% → 2.1% | 9.9% / 8.3% |

The loop is **non-monotonic** once the spectrum is learned: one pass already recovers
`μ_eff` (1.2%) and most cupping, pass 2 is *worse*, pass 3 trades `μ_eff` accuracy for
flatter cupping. Each pass re-optimises `I`/`μ` against a *moving* segmentation target,
so extra passes wander. **`main_2d_clean.py` now defaults to `outer_iters=1`** (faster,
best `μ_eff`, most stable); the frozen default likewise improves 9.2% → **7.1%** cupping
at a single pass. Use more passes only with `freeze_spectral` (segmentation-only
refinement — no spectral wandering).

CLI examples:
```bash
python main_2d_clean.py                                  # frozen + transmission (the fix)
python main_2d_clean.py --no-freeze --perturb 0.3 --suffix _recovery   # honest recovery test
python main_2d_clean.py --mu-eff-mode fluence            # the broken baseline
```

### 8.8 Closing the gap to the original authors (`og_work/`) — least-squares `μ_eff` + residual correction
`og_work/` is the original authors' `autodiffCT` codebase. Reading `og_work/autodiffCT/ISP.py`
showed the overall method matches ours, but **two of their choices are more robust** than our
re-implementation — now adopted (2-D):

- **`mu_eff_mode="lstsq"`** — instead of averaging the spectrum, solve a least-squares
  regression for the per-material effective attenuation that best reproduces the
  polychromatic simulation: `mu_eff = argmin_a ‖Σ_m a_m·As_m − y_poly‖² = pinv(B)·V`,
  `B[i,j]=⟨As_i,As_j⟩`, `V[i]=⟨As_i,y_poly⟩`. Measurement-weighted by construction →
  **structurally immune to the soft-tail inflation (§8.5)** and best per-material uniformity.
- **`correction_mode="residual"`** — instead of reconstructing a fully synthetic mono
  sinogram, correct the *measured* one by the modelled BH difference:
  **`y_corrected = y_meas + (y_mono − y_poly)`**. Preserves real measurement detail and
  removes almost all the cupping.

Both are threaded model → block → workflow in **both pipelines** (2-D `ISP2D` and 3-D
`ISP`); **defaults stay `"fluence"`/`"replace"`**, so `main_2d.py` / `main.py`'s old
behaviour is unchanged unless selected. `main.py` (3-D) now also runs
`mu_eff_mode="lstsq"`, `correction_mode="residual"`. The 3-D port is validated by
byte-compile + a unit test of the new lstsq/residual math (it recovers known linear
coefficients exactly) + energy-bin alignment (`dk=50` → 3 bins matches `energy_bins=3`);
a full 3-D run still needs HPC (ASTRA-CUDA memory, and a working-matplotlib env since
`render_phantom` plots in-process).

**Config comparison (frozen, dk=2, no filtration, outer=1):**

| `mu_eff_mode` | `correction_mode` | corrected cupping | PMMA CoV | Al CoV |
|---|---|---|---|---|
| transmission | replace | 7.1% | 0.105 | 0.072 |
| lstsq | replace | 7.1% | 0.077 | 0.064 |
| transmission | **residual** | **0.4%** | 0.099 | 0.074 |
| **lstsq** | **residual** | 2.4% | **0.069** | 0.066 |

The residual correction is the big cupping win; lstsq gives the best material uniformity.

**`main_2d.py` is now the primary results runner** — `mu_eff_mode="lstsq"`,
`correction_mode="residual"`, `freeze_spectral=False` (honest — learns the spectrum, lstsq
doesn't rely on it being exact), `dk=2`, noise off, `outer_iters=1`:

```
PMMA cupping 25.3% → 3.1%   |   PMMA CoV 0.159 → 0.066   |   Al mean preserved exactly
```

It writes `comparison_2d.png`, `cupping_2d.png`, `optimization_history_2d.png` (via the
`plot_clean_results.py` subprocess) and prints the report. `main_2d_clean.py` remains the
flag-driven harness (now with `--mu-eff-mode lstsq` and `--correction-mode residual`).

### 8.9 3-D fix — two bugs: ~0% hardening, *and* a broken segmentation/optimizer
`main.py` was producing `original_reconstruction.png` ≡ `final_reconstruction.png`
(correction does nothing) with a flat loss. **Two compounding bugs.** *(1)* The 3-D data
generator had almost no hardening to correct: `ProjectionData` used `dk=50`, which spekpy
resolves to only **3 energy bins** → an effectively monochromatic beam:

| `dk` | energy bins | hardening (0.5 cm PMMA) |
|---|---|---|
| **50 (old)** | **3** | **0.0%** |
| 5 (new) | 35 | 47% |
| 2 | 89 | 78% |

(`dk≈10` is a trap here — 17 bins still under-resolves → ~5%; `dk=5`/`8` jump to ~47%.)

*(2)* But `dk` alone was **not** sufficient: with hardening added, the correction *still*
did nothing (cupping 19.4% → 19.4%, per-material values unchanged, loss stuck at 1.77e-2).
The 3-D front end had **never received the 2-D §8.3 fixes**, so the fit could not converge:
`_s` thresholded the **raw** recon (~0.005) with `gamma=100` → mushy masks, and the
optimiser used a **single** lr over `I`(~1e5)/`mu`/`t` (so `I` was effectively frozen).

Fix (3-D, mirroring 2-D §8.3 / §8.5):
- **`ProjectionData` `dk` 50 → 5** (default), exposed through the workflow.
- **Workflow auto-detects `energy_bins`** from `fluence.npy` (drops hard-coded `energy_bins`).
- **`_s` normalises the recon to [0,1]** before tanh-thresholding (crisp masks) + sorts `t`.
- **`_build_optimizer`: per-parameter LR** scaled by each parameter's magnitude.
- **`ProjectionData` no longer plots the phantom in-pipeline** (`render_phantom(show_3d=False, …)`).
- **`main.py`** drops `energy_bins=3`, sets `dk=5`, `optim_steps=300`, keeps
  `mu_eff_mode="lstsq"`, `correction_mode="residual"`.

**Validated end-to-end (50 steps, dk=5, lstsq + residual, phantom-mask metrics):** the fit
now converges (loss 1.77e-2 → 6.7e-6) and the correction de-cups for real:

| metric | data fix only | + segmentation/optimizer fix |
|---|---|---|
| PMMA cupping (orig → corr) | 19.4% → **19.4%** | 19.4% → **0.9%** |
| PMMA CoV (corr) | 0.197 | **0.067** |
| Al CoV (corr) | 0.159 | **0.042** |

(Masks verified aligned: Air < PMMA < Al; means preserved.) Only the two *critical* 2-D
fixes were ported — the iterative outer loop and `freeze_spectral` are still 2-D-only (not
needed here). A full high-iteration `main.py` run is GPU-heavy (128³, ~35 bins, SIRT-1000
×2) and `plot_reconstruction` plots in-process — re-run it in a working-matplotlib env to
refresh `*_reconstruction.png`; the metrics above already confirm the correction works.
