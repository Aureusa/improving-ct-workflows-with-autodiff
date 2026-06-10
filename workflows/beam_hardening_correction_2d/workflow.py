"""
workflow.py -- 2-D beam-hardening correction (mirrors the 3-D workflow, one fewer axis).

Pipeline: ProjectionData2D -> measured sinogram -> Reconstruct2D (FBP, beam-hardened)
-> _optim_loop fits ISP2D so A_sim ~= A_meas -> compute_corrected_sinogram ->
CorrectProjection -> Reconstruct2D -> corrected reconstruction.
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

    Key parameters:
      dk                 : spectrum bin width [keV]; smaller -> more bins -> stronger hardening
      add_gaussian_noise : fractional Gaussian noise on the sinogram (0 = clean); noise_seed for repro
      freeze_spectral    : hold I/mu at ground truth, learn only t (honest test)
      al_filter_mm       : added Al filtration [mm]; removes the soft tail that inflates mu_eff
      mu_eff_mode        : 'fluence' | 'transmission' | 'lstsq'  (see ISP2D._effective_mu)
      correction_mode    : 'replace' (synthetic mono sinogram) | 'residual' (keeps real detail)
      spectral_perturb   : perturb I/mu init off ground truth (per-bin +/-); honest recovery test
      smooth_sigma       : Gaussian sigma [px] to denoise recon before segmentation (0 = off)
    Plus optim_steps, lr, n_angles, number_of_materials, gamma, size, scale, device.
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
        dk: float = 10.0,
        add_gaussian_noise: float = 0.02,
        noise_seed: int = 0,
        freeze_spectral: bool = False,
        al_filter_mm: float = 0.0,
        mu_eff_mode: str = "fluence",
        correction_mode: str = "replace",
        spectral_perturb: float = 3.0,
        spectral_perturb_seed: int = 0,
        spectral_bins: int = 0,
        smooth_sigma: float = 0.0,
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
            spectral_bins=spectral_bins,
            smooth_sigma=smooth_sigma,
            device=device,
        ))

        self._optim_steps = optim_steps
        self._loss_fn     = PhiLoss()
        self._lr          = lr
        self._correction_mode = correction_mode
        self._device      = device

        # Build initial optimizer (t is not yet in _params -- added on first forward)
        self._optim = self._build_optimizer()

        self.to(self._device)

    def _build_optimizer(self):
        """
        Adam with per-parameter LR scaled by each param's magnitude. I (~1e5), mu (~1)
        and t (~0.1-0.5) span orders of magnitude; a single absolute lr would leave I
        frozen while t moves. Per-magnitude scaling makes the fractional step uniform.
        """
        groups = []
        for name, p in self.parameters():
            scale = float(p.detach().abs().mean().clamp_min(1e-8))
            groups.append({"params": [p], "lr": self._lr * scale, "name": name})
        if not groups:
            # No trainable params registered yet (t added on first forward pass).
            return torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=self._lr)
        return torch.optim.Adam(groups)

    # -- Public API ------------------------------------------------------------

    def run(self):
        """
        Run the full pipeline. Returns (original_reconstruction, final_reconstruction,
        history): FBP of the polychromatic (beam-hardened) sinogram, FBP of the corrected
        mono sinogram, and per-iteration loss.
        """
        # Initial (uncorrected) reconstruction
        input_data  = self._input_data
        measured_projection = torch.from_numpy(input_data).float().to(self._device)

        original_reconstruction = self.Reconstruct2D.execute(input_data)
        current_recon = (
            torch.from_numpy(original_reconstruction).float().to(self._device)
        )

        # -- Single-pass correction ----------------------------------------------
        # Fit ISP2D against A_meas using path lengths segmented from the (beam-
        # hardened) recon, then synthesise the corrected sinogram and reconstruct it.
        history = self._optim_loop(measured_projection, current_recon)

        corrected_sino = self.SpectralProjection2D.compute_corrected_sinogram(
            current_recon,
            y_meas=measured_projection,
            correction_mode=self._correction_mode,
        )
        correct_np           = self.CorrectProjection.execute(corrected_sino)
        final_reconstruction = self.Reconstruct2D.execute(correct_np)

        return original_reconstruction, final_reconstruction, history

    # -- Private helpers -------------------------------------------------------

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

        # Diagnostic: warn if the tanh transition width (1/gamma in [0,1] space) exceeds
        # the smallest threshold gap -> masks won't be crisp and the forward model mis-fits.
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
