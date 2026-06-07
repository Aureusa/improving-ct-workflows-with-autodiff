"""
workflow.py  (2-D beam-hardening correction)
--------------------------------------------
Mirrors BeamHardeningCorrectionWorkflow from the 3-D version but operates
entirely on 2-D data:

  sinogram shape     : (n_angles, n_pixels)          ← 3-D: (n_pixels, n_angles, n_pixels)
  reconstruction shape : (n_pixels, n_pixels)         ← 3-D: (n_pixels, n_pixels, n_pixels)

Pipeline
--------
  ProjectionData2D.execute()
      → measured sinogram  (n_angles, n_pixels)

  Reconstruct2D.execute(sinogram)
      → initial reconstruction  (n_pixels, n_pixels)   [FBP, beam-hardened]

  _optim_loop: fits ISP2D parameters so A_sim ≈ A_meas

  SpectralProjection2D.compute_monochromatic_sinogram(reconstruction)
      → mono sinogram  (n_angles, n_pixels)             [beam-hardening free]

  CorrectProjection.execute(mono_sinogram)
      → numpy array  (n_angles, n_pixels)

  Reconstruct2D.execute(corrected_sinogram)
      → final reconstruction  (n_pixels, n_pixels)      [corrected]
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from ct_autodiff.engine.workflow import Workflow

from .blocks import CorrectProjection, ProjectionData2D, Reconstruct2D, SpectralProjection2D
from .objective_func import PhiLoss


class BeamHardeningCorrectionWorkflow2D(Workflow):
    """
    End-to-end 2-D beam-hardening correction workflow.

    Parameters
    ----------
    optim_steps        : gradient-descent iterations
    lr                 : Adam learning rate
    n_angles           : number of projection angles
    number_of_materials: number of distinct materials in the phantom
    energy_bins        : number of spectral energy bins (must match dk/kvp settings)
    gamma              : steepness of soft thresholding in ISP2D
    size               : phantom pixel grid (size × size)
    scale              : cm per pixel  (physical_width_cm / size)
    outer_iters        : iterative-correction passes (total steps = optim_steps * outer_iters)
    dk                 : spectrum bin width [keV] (smaller → more bins → stronger hardening)
    add_gaussian_noise : fractional Gaussian noise on the simulated sinogram (0.0 = clean)
    noise_seed         : RNG seed for the simulated noise (reproducibility)
    freeze_spectral    : if True, hold I/mu at ground truth and learn only t (honest test)
    al_filter_mm       : added Al tube filtration [mm]; removes the soft spectral tail
                         that otherwise inflates mu_eff (0.0 = no added filtration)
    mu_eff_mode        : 'fluence' (original), 'transmission', or 'lstsq' (original
                         autodiffCT least-squares regression) — effective attenuation
                         weighting (see ISP2D._effective_mu)
    correction_mode    : 'replace' (reconstruct a synthetic mono sinogram) or 'residual'
                         (y_meas + y_mono − y_poly; preserves real detail — autodiffCT)
    spectral_perturb   : perturb the I/mu init away from ground truth by this fraction
                         (per-bin ±); makes recovery an honest test with freeze_spectral=False
    spectral_perturb_seed : RNG seed for the spectral perturbation
    device             : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        optim_steps: int = 500,
        lr: float = 0.01,
        n_angles: int = 360,
        number_of_materials: int = 2,
        gamma: float = 100.0,
        size: int = 256,
        scale: float = 5.0 / 256,
        outer_iters: int = 3,
        dk: float = 10.0,
        add_gaussian_noise: float = 0.02,
        noise_seed: int = 0,
        freeze_spectral: bool = False,
        al_filter_mm: float = 0.0,
        mu_eff_mode: str = "fluence",
        correction_mode: str = "replace",
        spectral_perturb: float = 0.0,
        spectral_perturb_seed: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        # 1. Simulate the polychromatic sinogram and save spectrum/mu npy files
        #    *before* SpectralProjection2D.__init__ loads them.
        self.add_block(ProjectionData2D(
            size=size,
            scale=scale,
            dk=dk,
            add_gaussian_noise=add_gaussian_noise,
            noise_seed=noise_seed,
            n_angles=n_angles,
            al_filter_mm=al_filter_mm,
        ))
        self._input_data = self.ProjectionData2D.execute()

        # Auto-detect energy bin count from the npy written by ProjectionData2D
        _data_dir = os.path.dirname(os.path.abspath(__file__))
        energy_bins = int(np.load(os.path.join(_data_dir, "fluence.npy")).shape[0])

        # 2. Reconstruction and projection correction blocks
        self.add_block(Reconstruct2D(n_angles=n_angles, device=device))
        self.add_block(CorrectProjection(device=device))

        # 3. Differentiable polychromatic model (loads npy files written above)
        self.add_block(SpectralProjection2D(
            n_angles=n_angles,
            number_of_materials=number_of_materials,
            energy_bins=energy_bins,
            gamma=gamma,
            voxel_size=scale,   # keep physical scale consistent with ProjectionData2D
            freeze_spectral=freeze_spectral,
            mu_eff_mode=mu_eff_mode,
            spectral_perturb=spectral_perturb,
            spectral_perturb_seed=spectral_perturb_seed,
            device=device,
        ))

        self._optim_steps = optim_steps
        self._loss_fn     = PhiLoss()
        self._lr          = lr
        self._outer_iters = outer_iters
        self._correction_mode = correction_mode
        self._device      = device

        # Build initial optimizer (t is not yet in _params — added on first forward)
        self._optim = self._build_optimizer()

        self.to(self._device)

    def _build_optimizer(self):
        """
        Adam with per-parameter learning rates scaled by each parameter's own
        magnitude, so every group takes ~`lr` *fractional* steps per iteration.

        I (~1e5 photons), mu (~1 cm⁻¹) and t (~0.1-0.5 in normalised space) span
        many orders of magnitude. A single absolute lr (Adam moves each param by
        ≈lr/step) leaves I effectively frozen while t moves fine. Scaling each
        group's lr by its mean magnitude makes the relative step size uniform.
        """
        groups = []
        for name, p in self.parameters():
            scale = float(p.detach().abs().mean().clamp_min(1e-8))
            groups.append({"params": [p], "lr": self._lr * scale, "name": name})
        if not groups:
            # No trainable params registered yet (t added on first forward pass).
            return torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=self._lr)
        return torch.optim.Adam(groups)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self):
        """
        Execute the full beam-hardening correction pipeline.

        Returns
        -------
        original_reconstruction : ndarray (n_pixels, n_pixels)
            FBP reconstruction from the polychromatic sinogram (beam-hardened).
        final_reconstruction    : ndarray (n_pixels, n_pixels)
            FBP reconstruction from the corrected monochromatic sinogram.
        history                 : list[float]
            Per-iteration loss values, concatenated across all outer passes.
        """
        # Initial (uncorrected) reconstruction
        input_data  = self._input_data
        measured_projection = torch.from_numpy(input_data).float().to(self._device)

        original_reconstruction = self.Reconstruct2D.execute(input_data)
        current_recon = (
            torch.from_numpy(original_reconstruction).float().to(self._device)
        )

        # ── Iterative correction ────────────────────────────────────────────────
        # Each outer pass: fit ISP2D against A_meas using path lengths segmented
        # from the *current* recon, synthesise a monochromatic sinogram, and
        # reconstruct it. The corrected (de-cupped) recon then feeds the next
        # pass, so the segmentation that drives the path lengths keeps improving
        # instead of being frozen on the beam-hardened recon (single-pass).
        history = []
        final_reconstruction = original_reconstruction
        for outer in range(self._outer_iters):
            print(f"\n[outer {outer + 1}/{self._outer_iters}]")
            history += self._optim_loop(measured_projection, current_recon)

            corrected_sino = self.SpectralProjection2D.compute_corrected_sinogram(
                current_recon,
                y_meas=measured_projection,
                correction_mode=self._correction_mode,
            )
            correct_np           = self.CorrectProjection.execute(corrected_sino)
            final_reconstruction = self.Reconstruct2D.execute(correct_np)
            current_recon = (
                torch.from_numpy(final_reconstruction).float().to(self._device)
            )

        return original_reconstruction, final_reconstruction, history

    # ── Private helpers ───────────────────────────────────────────────────────

    def _optim_loop(
        self,
        A_meas: torch.Tensor,           # (n_angles, n_pixels)
        reconstructed_data: torch.Tensor,  # (n_pixels, n_pixels)
    ):
        history = []

        # Warmup forward pass: triggers Otsu initialisation of t and registers
        # it with _params so the optimizer can update it.
        with torch.no_grad():
            self.SpectralProjection2D.execute(reconstructed_data)

        # Rebuild optimizer to include t now that it has been added
        self._optim = self._build_optimizer()

        # ── Diagnostic: tanh-threshold steepness in the normalised [0,1] space ──
        # _s normalises the recon to [0,1] before thresholding, so the mask
        # transition width is 1/gamma in that space. Compare it against the
        # spacing between the (normalised) thresholds; if the width is wider the
        # masks never become crisp and the forward model is mis-fit.
        with torch.no_grad():
            t_init    = torch.sort(self.SpectralProjection2D.t)[0]
            gamma_val = float(self.SpectralProjection2D.gamma)
            width     = 1.0 / max(gamma_val, 1e-12)
            edges     = torch.cat(
                [torch.zeros(1).to(t_init), t_init.flatten(), torch.ones(1).to(t_init)]
            )
            min_gap = float(torch.diff(edges).min())
            print(
                f"\n[diag] normalised thresholds {t_init.cpu().numpy().round(4)}; "
                f"gamma {gamma_val:g} -> tanh transition width {width:.4g} (in [0,1] space)"
            )
            if width > min_gap:
                print(
                    f"[diag] WARNING: transition width ({width:.4g}) exceeds the smallest "
                    f"threshold gap ({min_gap:.4g}); soft masks will be blurry. "
                    f"Consider gamma >~ {5.0 / max(min_gap, 1e-12):.0f}."
                )

        for _ in tqdm(
            range(self._optim_steps),
            desc="Optimising Spectral Projection 2D",
        ):
            A_sim = self.SpectralProjection2D.execute(reconstructed_data)
            loss  = self._loss_fn(A_sim, A_meas)

            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            history.append(loss.item())

        return history
