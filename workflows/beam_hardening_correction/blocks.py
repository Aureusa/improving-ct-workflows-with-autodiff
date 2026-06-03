import torch

from ct_autodiff.engine.block import Block
# imports for xray analysis
import xraylib
import spekpy as sp
# plotting and numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

#astra
import astra

from .barba_3D_phantom_1 import render_phantom, generate_linear_attenuation_params, calculate_I, de_bone, astra_back_projection, astra_forward_project
from .isp import ISP

###############################################
#               Input blocks                  #
##############################################
class ProjectionData(Block):
    def __init__(self, kvp=180, th=12, dk=50, physics="spekcalc"):
        super().__init__()
        self.kvp = kvp
        self.th = th
        self.dk = dk
        self.physics = physics

    def execute(self):
        # Work around SpekPy v2 + NumPy 2.x incompatibility in default physics path.
        r = sp.Spek(kvp=self.kvp, th=self.th, dk=self.dk, physics=self.physics)  # Generate a spectrum (180 kV, 12 degree tube angle)
        
        pmma_mu = generate_linear_attenuation_params(r, ("C5H8O2"))  # Get attenuation coefficients for PMMA
        al_mu = generate_linear_attenuation_params(r, "Al")  # Get attenuation coefficients for Aluminum
        phantom = render_phantom(show_projection=False)
        
        sinogram = calculate_I(r, pmma_mu, al_mu, phantom, scale=0.5 / 128, add_gaussian_noise=0.00)

        # Saving
        e_bins = r.get_k()
        np.save("/home/s4861264/CIT_project/workflows/beam_hardening_correction/energy_bins.npy", e_bins)

        # Save spectral fluence (photons per bin) — used to initialise ISP._I
        fluence = r.get_spk()
        np.save("/home/s4861264/CIT_project/workflows/beam_hardening_correction/fluence.npy", fluence)

        # stack into 2d array for easier handling
        mu_values = np.stack([pmma_mu, al_mu], axis=0) # shape (2, energy_bins)
        np.save("/home/s4861264/CIT_project/workflows/beam_hardening_correction/mu_values.npy", mu_values)
        return sinogram

###############################################
#          Non optimizable blocks             #
###############################################

class Reconstruct(Block):
    def __init__(self, n_angles=360, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.n_angles = n_angles
        self._device = device

    def execute(self, sinogram):
        # Placeholder for the actual reconstruction logic
        # This should implement the beam hardening correction algorithm
        reconstruction = astra_back_projection(sinogram=sinogram, n_angles=self.n_angles)
        return reconstruction

class CorrectProjection(Block):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self._device = device
    
    def execute(self, projection_data):
        # Detach, move to cpu and convert to numpy
        projection_data_np = projection_data.detach().cpu().numpy()
        return projection_data_np


###############################################
#              Optimizable blocks             #
###############################################

class SpectralProjection(Block, ISP):
    def __init__(self, n_angles=360, number_of_materials=2, gamma=1.0, energy_bins=3, voxel_size=0.5/128, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        Block.__init__(self)
        ISP.__init__(
            self,
            n_angles=n_angles,
            number_of_materials=number_of_materials,
            gamma=gamma,
            energy_bins=energy_bins,
            voxel_size=voxel_size,
            device=device
        )
        self.add_param(self._I, "I", trainable=True)
        self.add_param(self._mu, "mu", trainable=True)
        # self.add_param(self.t, "t", trainable=True)
        self.add_param(self._gamma, "gamma", trainable=False)
        self._params["I"].to(device)
        self._params["mu"].to(device)
        self._params["gamma"].to(device)
        self._device = device

    def execute(self, reconstruction):
        return self.forward(reconstruction)
    
    def _s(self, x):
        return ISP._s(self, x)
