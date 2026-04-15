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
    def __init__(self):
        super().__init__()

    def execute(self):
        # Work around SpekPy v2 + NumPy 2.x incompatibility in default physics path.
        r = sp.Spek(kvp=120, th=12, physics="spekcalc")  # Generate a spectrum (120 kV, 12 degree tube angle)
        
        pmma_mu = generate_linear_attenuation_params(r, ("C5H8O2"))  # Get attenuation coefficients for PMMA
        al_mu = generate_linear_attenuation_params(r, "Al")  # Get attenuation coefficients for Aluminum
        phantom = render_phantom(show_projection=False)
        
        sinogram = calculate_I(r, pmma_mu, al_mu, phantom, scale=0.5 / 128, add_gaussian_noise=0.02)
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
        
        # Move to torch tensor for further processing in the workflow
        return torch.from_numpy(reconstruction).float().to(self._device)

class CorrectProjection(Block):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self._device = device
    
    def execute(self, projection_data):
        # The projection_data is expected to be (n_angles, n_pixels, n_pixels) tensor
        # We permute so that we get (n_pixels, n_angles, n_pixels) for astra
        projection_data = projection_data.permute(1, 0, 2).cpu().numpy() # (n_pixels, n_angles, n_pixels)
        return projection_data


###############################################
#              Optimizable blocks             #
###############################################

class SpectralProjection(Block, ISP):
    def __init__(self, n_angles=360, number_of_materials=2, gamma=1.0, energy_bins=358, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        Block.__init__(self)
        ISP.__init__(
            self,
            n_angles=n_angles,
            number_of_materials=number_of_materials,
            gamma=gamma,
            energy_bins=energy_bins,
            device=device
        )
        self.add_param(self.I, "I", trainable=True)
        self.add_param(self.mu, "mu", trainable=True)
        self.add_param(self.t, "t", trainable=True)
        self.add_param(self.gamma, "gamma", trainable=False)
        self._device = device

    def execute(self, reconstruction):
        return self.forward(reconstruction)
