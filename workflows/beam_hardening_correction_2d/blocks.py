import os

import numpy as np
import spekpy as sp
import torch

from ct_autodiff.engine.block import Block

from .barba_2d_phantom import (
    astra_back_projection_2d,
    calculate_I_2d,
    generate_linear_attenuation_params,
    render_phantom_2d,
)
from .isp_2d import ISP2D

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# -- Input block ---------------------------------------------------------------

class ProjectionData2D(Block):
    """
    Simulate a polychromatic CT acquisition of the 2-D Barba phantom.

    Side-effects (written to the module directory for ISP2D to load):
        energy_bins.npy  -- (E,) energy bin centres [keV]
        fluence.npy      -- (E,) spectral photon fluence per bin
        mu_values.npy    -- (2, E) linear attenuation [cm^-1] for PMMA and Al
    """

    def __init__(
        self,
        kvp: float = 120.0,
        th: float = 12.0,
        dk: float = 10.0,
        physics: str = "spekcalc",
        size: int = 256,
        scale: float = 5.0 / 256,
        add_gaussian_noise: float = 0.0,
        n_angles: int = 360,
        noise_seed: int = 0,
        al_filter_mm: float = 0.0,
    ):
        super().__init__()
        self.kvp               = kvp
        self.th                = th
        self.dk                = dk
        self.physics           = physics
        self.size              = size
        self.scale             = scale
        self.add_gaussian_noise = add_gaussian_noise
        self.n_angles          = n_angles
        self.noise_seed        = noise_seed
        self.al_filter_mm      = al_filter_mm

    def execute(self) -> np.ndarray:
        """Return the polychromatic attenuation sinogram -log(I/I0), (n_angles, size)."""
        r = sp.Spek(kvp=self.kvp, th=self.th, dk=self.dk, physics=self.physics)

        # Optional tube filtration
        if self.al_filter_mm > 0:
            r.filter("Al", self.al_filter_mm)

        pmma_mu = generate_linear_attenuation_params(r, "C5H8O2")
        al_mu   = generate_linear_attenuation_params(r, "Al")

        phantom = render_phantom_2d(size=self.size, show=False)

        sinogram = calculate_I_2d(
            r, pmma_mu, al_mu, phantom,
            scale=self.scale,
            add_gaussian_noise=self.add_gaussian_noise,
            n_angles=self.n_angles,
            seed=self.noise_seed,
        )

        # Persist for ISP2D.__init__
        np.save(os.path.join(_DATA_DIR, "energy_bins.npy"), r.get_k())
        np.save(os.path.join(_DATA_DIR, "fluence.npy"),     r.get_spk())
        np.save(
            os.path.join(_DATA_DIR, "mu_values.npy"),
            np.stack([pmma_mu, al_mu], axis=0),   # (2, E)
        )

        return sinogram


# -- Non-optimisable blocks ----------------------------------------------------

class Reconstruct2D(Block):
    """Reconstruct a 2-D slice from a sinogram via ASTRA FBP (or SIRT). Accepts numpy or torch."""

    def __init__(
        self,
        n_angles: int = 360,
        algorithm: str = "FBP_CUDA",
        iterations: int = 200,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.n_angles   = n_angles
        self.algorithm  = algorithm
        self.iterations = iterations
        self._device    = device

    def execute(self, sinogram) -> np.ndarray:
        """sinogram (n_angles, n_det) -> reconstruction (n_det, n_det) float32."""
        if torch.is_tensor(sinogram):
            sinogram = sinogram.detach().cpu().numpy()
        return astra_back_projection_2d(
            sinogram,
            n_angles=self.n_angles,
            algorithm=self.algorithm,
            iterations=self.iterations,
        )


class CorrectProjection(Block):
    """Detach a tensor to CPU numpy (pass-through for numpy input)."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self._device = device

    def execute(self, projection_data) -> np.ndarray:
        if torch.is_tensor(projection_data):
            return projection_data.detach().cpu().numpy()
        return projection_data


# -- Optimisable block ---------------------------------------------------------

class SpectralProjection2D(Block, ISP2D):
    """
    Differentiable 2-D polychromatic forward model (Block + ISP2D).
    Learnable params: I (fluence), mu (attenuation), t (Otsu thresholds, added on first
    forward); gamma frozen. freeze_spectral=True registers I/mu non-trainable -> learn
    only t.
    """

    def __init__(
        self,
        n_angles: int = 360,
        number_of_materials: int = 2,
        gamma: float = 1.0,
        energy_bins: int = 3,
        voxel_size: float = 5.0 / 256,
        freeze_spectral: bool = False,
        mu_eff_mode: str = "fluence",
        spectral_perturb: float = 0.0,
        spectral_perturb_seed: int = 0,
        smooth_sigma: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        Block.__init__(self)
        ISP2D.__init__(
            self,
            n_angles=n_angles,
            number_of_materials=number_of_materials,
            gamma=gamma,
            energy_bins=energy_bins,
            voxel_size=voxel_size,
            mu_eff_mode=mu_eff_mode,
            spectral_perturb=spectral_perturb,
            spectral_perturb_seed=spectral_perturb_seed,
            smooth_sigma=smooth_sigma,
            device=device,
        )
        # freeze_spectral=True -> hold I and mu fixed at their ground-truth
        # initialisation; Block.parameters() only yields trainable params, so the
        # optimizer will then update only t (added on the first forward pass).
        spectral_trainable = not freeze_spectral
        self.add_param(self._I,     "I",     trainable=spectral_trainable)
        self.add_param(self._mu,    "mu",    trainable=spectral_trainable)
        self.add_param(self._gamma, "gamma", trainable=False)
        self._params["I"].to(device)
        self._params["mu"].to(device)
        self._params["gamma"].to(device)
        self._device = device

    def execute(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        reconstruction : float32 tensor (n_pixels, n_pixels)

        Returns
        -------
        A_sim : float32 tensor (n_angles, n_pixels)
        """
        return self.forward(reconstruction)

    def _s(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly dispatch to ISP2D._s so Block.__getattr__ cannot shadow it
        return ISP2D._s(self, x)
