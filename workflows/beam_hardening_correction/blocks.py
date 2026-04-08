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

###############################################
#               Input blocks                  #
##############################################
class ProjectionData(Block):
    def __init__(self):
        super().__init__()

    def forward(self):
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
    def __init__(self):
        super().__init__()

    def forward(self, sinogram):
        # Placeholder for the actual reconstruction logic
        # This should implement the beam hardening correction algorithm
        reconstruction = astra_back_projection(sinogram=sinogram, n_angles=360) # TODO: Fix n_angles!!!
        return reconstruction

class CorrectProjection(Block):
    def __init__(self):
        super().__init__()

    def forward(self):
        # Placeholder for the actual projection correction logic
        # This should implement the correction of the projection data
        pass


###############################################
#              Optimizable blocks             #
###############################################

class SpectralProjection(Block):
    def __init__(self):
        super().__init__()

    def forward(self):
        # Placeholder for the actual spectral projection logic
        # This should implement the spectral projection step of the correction
        pass
