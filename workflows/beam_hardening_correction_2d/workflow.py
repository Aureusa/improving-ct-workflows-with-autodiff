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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        # 1. Simulate the polychromatic sinogram and save spectrum/mu npy files
        #    *before* SpectralProjection2D.__init__ loads them.
        self.add_block(ProjectionData2D(
            size=size,
            scale=scale,
            add_gaussian_noise=0.02,
            n_angles=n_angles,
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
            device=device,
        ))

        self._optim_steps = optim_steps
        self._loss_fn     = PhiLoss()
        self._lr          = lr
        self._device      = device

        # Build initial optimizer (t is not yet in _params — added on first forward)
        self._optim = torch.optim.Adam(
            [p for _, p in self.parameters()], lr=lr
        )

        self.to(self._device)

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
            Per-iteration loss values from the optimisation loop.
        """
        # Initial (uncorrected) reconstruction
        input_data  = self._input_data
        measured_projection = torch.from_numpy(input_data).float().to(self._device)

        original_reconstruction = self.Reconstruct2D.execute(input_data)
        original_reconstruction_tensor = (
            torch.from_numpy(original_reconstruction).float().to(self._device)
        )

        # Optimise ISP2D to match the measured sinogram
        history = self._optim_loop(measured_projection, original_reconstruction_tensor)

        # Synthesise beam-hardening-free sinogram with learned parameters
        mono_sinogram    = self.SpectralProjection2D.compute_monochromatic_sinogram(
            original_reconstruction_tensor
        )
        correct_np       = self.CorrectProjection.execute(mono_sinogram)
        final_reconstruction = self.Reconstruct2D.execute(correct_np)

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
        self._optim = torch.optim.Adam(
            [p for _, p in self.parameters()], lr=self._lr
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
