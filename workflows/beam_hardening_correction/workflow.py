import os

import numpy as np
import torch
from tqdm import tqdm

from ct_autodiff.engine.workflow import Workflow

from .blocks import Reconstruct, CorrectProjection, SpectralProjection, ProjectionData
from .objective_func import PhiLoss


class BeamHardeningCorrectionWorkflow(Workflow):
    def __init__(
            self,
            optim_steps: int = 500,
            lr: float = 0.01,
            n_angles=360,
            number_of_materials=2,
            dk=5,
            gamma=100,
            mu_eff_mode="fluence",
            correction_mode="replace",
            spectral_perturb=0.0,
            spectral_perturb_seed=0,
            add_gaussian_noise=0.0,
            noise_seed=0,
            smooth_sigma=0.0,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()
        # dk sets the spectral resolution. dk=50 -> ~2 bins -> ~0% hardening (nothing to
        # correct, original==final). dk=5 -> ~24 bins (at 120 kVp) -> real hardening.
        self.add_block(ProjectionData(dk=dk, add_gaussian_noise=add_gaussian_noise, noise_seed=noise_seed))
        # Execute the ProjectionData block to get the measured sinogram; it also writes
        # energy_bins.npy / fluence.npy / mu_values.npy used to seed the ISP block.
        self._input_data = self.ProjectionData.execute()

        # Auto-detect energy bin count from the npy ProjectionData just wrote, so dk can
        # change freely without a brittle hard-coded energy_bins (mirrors the 2-D workflow).
        _data_dir = os.path.dirname(os.path.abspath(__file__))
        energy_bins = int(np.load(os.path.join(_data_dir, "fluence.npy")).shape[0])

        self.add_block(Reconstruct(n_angles=n_angles, device=device))
        self.add_block(CorrectProjection(device=device))
        self.add_block(SpectralProjection(
            n_angles=n_angles,
            number_of_materials=number_of_materials,
            energy_bins=energy_bins,
            gamma=gamma,
            mu_eff_mode=mu_eff_mode,
            spectral_perturb=spectral_perturb,
            spectral_perturb_seed=spectral_perturb_seed,
            smooth_sigma=smooth_sigma,
            device=device)
        )

        self._optim_steps = optim_steps
        self._correction_mode = correction_mode
        self._loss_fn = PhiLoss()

        self._lr = lr
        self._optim = self._build_optimizer()

        self._device = device
        self.to(self._device)

    def _build_optimizer(self):
        """
        Adam with per-parameter learning rates scaled by each parameter's own
        magnitude, so every group takes `lr` *fractional* steps per iteration.
        """
        groups = []
        for name, p in self.parameters():
            scale = float(p.detach().abs().mean().clamp_min(1e-8))
            groups.append({"params": [p], "lr": self._lr * scale, "name": name})
        if not groups:
            return torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=self._lr)
        return torch.optim.Adam(groups)

    def run(self):
        """
        Executes the workflow by sequentially running each block on the output of the previous block.
        
        :param input_data: The initial input data for the workflow, which will be passed to the first block.
        :type input_data: Any
        :return: The output of the final block in the workflow after processing the input data through
                 all blocks.
        :rtype: Any
        """
        input_data = self._input_data
        measured_projection = torch.from_numpy(input_data).float().to(self._device)
        original_reconstruction = self.Reconstruct.execute(input_data)

        # Convert original_reconstruction to torch tensor for optimization
        original_reconstruction_tensor = torch.from_numpy(original_reconstruction).float().to(self._device)
        
        # Optimization loop - fits ISP parameters so the model matches the measured sinogram
        history = self._optim_loop(measured_projection, original_reconstruction_tensor)

        # After optimisation, compute a monochromatic-equivalent sinogram using the
        # learned material decomposition.  Reconstructing from this removes beam hardening.
        corrected_sino = self.SpectralProjection.compute_corrected_sinogram(
            original_reconstruction_tensor,
            y_meas=measured_projection,
            correction_mode=self._correction_mode,
        )
        correct_projection = self.CorrectProjection.execute(corrected_sino)

        final_reconstruction = self.Reconstruct.execute(correct_projection)
        return original_reconstruction, final_reconstruction, history
            
    def _optim_loop(self, input_data, reconstructed_data):
        history = []
        A_meas = input_data # (n_pixels, n_angles, n_pixels)  -- ASTRA parallel3d layout

        # Warmup forward pass: triggers Otsu initialization of t so it is added
        # to _params before the optimizer is (re)built.
        with torch.no_grad():
            self.SpectralProjection.execute(reconstructed_data)
        self._optim = self._build_optimizer()

        for _ in tqdm(range(self._optim_steps), desc="Optimizing Spectral Projection"):
            A_sim = self.SpectralProjection.execute(reconstructed_data) # (n_pixels, n_angles, n_pixels)
            loss = self._loss_fn(A_sim, A_meas)

            # Step
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            history.append(loss.item())
        return history
