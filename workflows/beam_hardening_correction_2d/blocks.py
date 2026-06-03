"""
blocks.py  (2-D beam-hardening workflow)
----------------------------------------
Blocks for the 2-D beam-hardening correction pipeline.

Block                  Role
─────────────────────  ───────────────────────────────────────────────────────
ProjectionData2D       Generate 2-D phantom, simulate polychromatic sinogram,
                       save spectrum/mu .npy files for ISP2D initialisation.
Reconstruct2D          ASTRA FBP reconstruction from a 2-D sinogram.
CorrectProjection      Detach / move-to-CPU helper (identical to 3-D version).
SpectralProjection2D   Differentiable polychromatic forward model (ISP2D).
"""

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


# ── Input block ───────────────────────────────────────────────────────────────

class ProjectionData2D(Block):
    """
    Simulate a polychromatic CT acquisition of the 2-D Barba phantom.

    Side-effects (written to the module directory for ISP2D to load):
        energy_bins.npy  — (E,) energy bin centres [keV]
        fluence.npy      — (E,) spectral photon fluence per bin
        mu_values.npy    — (2, E) linear attenuation [cm⁻¹] for PMMA and Al
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

    def execute(self) -> np.ndarray:
        """
        Returns
        -------
        sinogram : float32 ndarray (n_angles, size)
            Polychromatic attenuation sinogram −log(I / I₀).
        """
        r = sp.Spek(kvp=self.kvp, th=self.th, dk=self.dk, physics=self.physics)

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


# ── Non-optimisable blocks ────────────────────────────────────────────────────

class Reconstruct2D(Block):
    """
    Reconstruct a 2-D slice from a sinogram using ASTRA FBP (or SIRT).

    Accepts both numpy arrays and torch tensors as input.
    """

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
        """
        Parameters
        ----------
        sinogram : ndarray or tensor (n_angles, n_detectors)

        Returns
        -------
        reconstruction : float32 ndarray (n_detectors, n_detectors)
        """
        if torch.is_tensor(sinogram):
            sinogram = sinogram.detach().cpu().numpy()
        return astra_back_projection_2d(
            sinogram,
            n_angles=self.n_angles,
            algorithm=self.algorithm,
            iterations=self.iterations,
        )


class CorrectProjection(Block):
    """
    Detach a tensor from the autograd graph and move it to CPU as a numpy array.
    Accepts tensors and plain numpy arrays (pass-through for the latter).
    """

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


# ── Optimisable block ─────────────────────────────────────────────────────────

class SpectralProjection2D(Block, ISP2D):
    """
    Differentiable 2-D polychromatic forward model.

    Inherits from both Block (engine integration) and ISP2D (model logic).
    Learnable parameters exposed to the workflow optimizer:
        I     — spectral fluence  (energy_bins,)
        mu    — linear attenuation  (number_of_materials, energy_bins)
        gamma — steepness (frozen)
        t     — Otsu-initialised thresholds (added on first forward pass)
    """

    def __init__(
        self,
        n_angles: int = 360,
        number_of_materials: int = 2,
        gamma: float = 1.0,
        energy_bins: int = 3,
        voxel_size: float = 5.0 / 256,
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
            device=device,
        )
        self.add_param(self._I,     "I",     trainable=True)
        self.add_param(self._mu,    "mu",    trainable=True)
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
