"""
isp_2d.py
---------
2-D Iterative Spectral Projection model.

Identical in structure to the 3-D ISP but adapted for 2-D:
  • Reconstruction volume shape : (n_pixels, n_pixels)        ← was (n_pixels, n_pixels, n_pixels)
  • Sinogram shape               : (n_angles, n_pixels)        ← was (n_pixels, n_angles, n_pixels)
  • Material-fraction field s    : (M, n_pixels, n_pixels)     ← was (M, n_pixels, n_pixels, n_pixels)
  • Beer–Lambert einsum          : 'em,map->eap'               ← was 'em,mrap->erap'
  • Mono-sinogram einsum         : 'm,map->ap'                 ← was 'm,mrap->rap'
"""

import os

import numpy as np
import torch
from skimage.filters import threshold_multiotsu

from .utils import tanh_thresholding
from .barba_2d_phantom import astra_forward_project_2d_differentiable

# npy files written by ProjectionData2D live in the same directory as this module
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class ISP2D(torch.nn.Module):
    """
    2-D Iterative Spectral Projection model.

    Learnable parameters
    --------------------
    I   : (energy_bins,)               — spectral photon fluence per bin
    mu  : (number_of_materials, energy_bins) — linear attenuation [cm⁻¹]
    t   : (number_of_materials,)       — Otsu-initialised soft-threshold values
    gamma : scalar                     — steepness of the tanh thresholds (fixed)
    """

    def __init__(
        self,
        n_angles: int = 360,
        number_of_materials: int = 2,
        gamma: float = 1.0,
        energy_bins: int = 3,
        energy_chunk_size: int = 16,
        voxel_size: float = 5.0 / 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ISP2D, self).__init__()

        self.current_iter      = 0
        self.n_angles          = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins       = energy_bins
        self.energy_chunk_size = energy_chunk_size
        self.voxel_size        = voxel_size
        self._device           = device
        self._t_initialized    = False

        # Load initial guesses saved by ProjectionData2D.execute()
        fluence_vals = np.load(os.path.join(_DATA_DIR, "fluence.npy"))
        mu_vals      = np.load(os.path.join(_DATA_DIR, "mu_values.npy"))

        if mu_vals.shape != (number_of_materials, energy_bins):
            raise ValueError(
                f"mu_values.npy has shape {mu_vals.shape}, "
                f"expected ({number_of_materials}, {energy_bins})"
            )
        if fluence_vals.shape != (energy_bins,):
            raise ValueError(
                f"fluence.npy has shape {fluence_vals.shape}, "
                f"expected ({energy_bins},)"
            )

        # Learnable parameters (placeholders — actual values loaded above)
        self._I     = torch.nn.Parameter(
            torch.from_numpy(fluence_vals).float(), requires_grad=True
        )   # (energy_bins,)
        self._mu    = torch.nn.Parameter(
            torch.from_numpy(mu_vals).float(), requires_grad=True
        )   # (number_of_materials, energy_bins)
        self._t     = torch.nn.Parameter(
            torch.rand(self.number_of_materials), requires_grad=True
        )   # (number_of_materials,)  — overwritten by Otsu on first forward pass
        self._gamma = torch.nn.Parameter(
            torch.tensor(gamma), requires_grad=False
        )   # scalar

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        reconstruction : float32 tensor (n_pixels, n_pixels)

        Returns
        -------
        A_sim : float32 tensor (n_angles, n_pixels)
            Simulated polychromatic attenuation sinogram −log(I / I₀).
        """
        reconstruction = reconstruction.to(self._device)
        I_sim = self._compute_I_sim(reconstruction)   # (n_angles, n_pixels)
        A_sim = self._compute_A_sim(I_sim)            # (n_angles, n_pixels)
        return A_sim

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_A_sim(self, I_sim: torch.Tensor) -> torch.Tensor:
        """−log(I_sim / I₀)  — same formula as 3-D."""
        eps  = 1e-8
        I_0  = torch.sum(self.I).clamp_min(eps)          # scalar
        return -torch.log((I_sim / I_0).clamp_min(eps))  # (n_angles, n_pixels)

    def _compute_I_sim(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Polychromatic intensity sum over all energy bins.

        reconstruction : (n_pixels, n_pixels)
        returns        : (n_angles, n_pixels)
        """
        # Material decomposition: soft masks for each material
        s = self._s(reconstruction)  # (M, n_pixels, n_pixels)

        # Forward-project each material's soft mask → path-length sinogram
        l_list = []
        for n in range(self.number_of_materials):
            l_n = astra_forward_project_2d_differentiable(s[n], self.n_angles)
            # l_n : (n_angles, n_pixels)
            l_list.append(l_n)

        l    = torch.stack(l_list, dim=0)  # (M, n_angles, n_pixels)
        As_n = l * self.voxel_size         # (M, n_angles, n_pixels)  [cm]

        # Beer–Lambert sum in energy chunks to avoid OOM
        mu   = self.mu.permute(1, 0).to(self._device)  # (E, M)
        I    = self.I.to(self._device)                  # (E,)
        I_sim = torch.zeros_like(As_n[0])               # (n_angles, n_pixels)

        for start in range(0, self.energy_bins, self.energy_chunk_size):
            end        = min(start + self.energy_chunk_size, self.energy_bins)
            mu_chunk   = mu[start:end]                          # (chunk, M)
            # exponent[e, a, p] = sum_m( mu[e,m] * As_n[m,a,p] )
            exp_chunk  = torch.einsum("em,map->eap", mu_chunk, As_n)
            int_chunk  = I[start:end].view(-1, 1, 1) * torch.exp(-exp_chunk)
            I_sim      = I_sim + int_chunk.sum(dim=0)

        return I_sim  # (n_angles, n_pixels)

    def compute_monochromatic_sinogram(
        self, reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """
        After optimisation, synthesise a beam-hardening-free sinogram:

            mu_eff[m] = Σ_e( I_e · μ[m,e] ) / Σ_e(I_e)
            A_mono    = Σ_m( mu_eff[m] · As_n[m] )

        Parameters
        ----------
        reconstruction : float32 tensor (n_pixels, n_pixels)

        Returns
        -------
        A_mono : float32 tensor (n_angles, n_pixels)
        """
        with torch.no_grad():
            reconstruction = reconstruction.to(self._device)
            s = self._s(reconstruction)  # (M, n_pixels, n_pixels)

            l_list = []
            for n in range(self.number_of_materials):
                l_n = astra_forward_project_2d_differentiable(s[n], self.n_angles)
                l_list.append(l_n)

            l    = torch.stack(l_list, dim=0)   # (M, n_angles, n_pixels)
            As_n = l * self.voxel_size           # (M, n_angles, n_pixels)

            I     = self.I.to(self._device)      # (E,)
            I_sum = I.sum().clamp_min(1e-8)
            mu_eff = (
                self.mu.to(self._device) * I.unsqueeze(0)
            ).sum(dim=1) / I_sum                 # (M,)

            # A_mono[a, p] = Σ_m( mu_eff[m] · As_n[m, a, p] )
            A_mono = torch.einsum("m,map->ap", mu_eff, As_n)  # (n_angles, n_pixels)

        return A_mono

    def _s(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft material-fraction field via cumulative tanh thresholding.

        Parameters
        ----------
        x : float32 tensor (n_pixels, n_pixels)  — reconstruction slice

        Returns
        -------
        s : float32 tensor (M, n_pixels, n_pixels)
            s[m] ≈ 1 where the voxel belongs to material m, with smooth transitions.
        """
        if not self._t_initialized:
            # Initialise thresholds from multi-Otsu segmentation of the reconstruction
            thresholds = threshold_multiotsu(
                x.cpu().detach().numpy(),
                classes=self.number_of_materials + 1,
                nbins=128,
            )
            self._t = torch.nn.Parameter(
                torch.tensor(thresholds, device=self._device, dtype=x.dtype),
                requires_grad=True,
            )
            # Register with Block so the optimizer can update it
            self.add_param(self._t, "t", trainable=True)
            self._t_initialized = True
            t = self.t   # fetch via _params["t"].tensor
        else:
            t = self.t

        # Broadcast t : (M,) → (M, n_pixels, n_pixels)
        t = t.reshape(self.number_of_materials, 1, 1)
        t = t.expand(self.number_of_materials, x.shape[0], x.shape[1])

        # s_cum[m] = σ( γ · (x − t[m]) )  — (M, n_pixels, n_pixels)
        s_cum = tanh_thresholding(x, t, self.gamma)

        # Convert cumulative to exclusive: s[m] = s_cum[m] − s_cum[m+1]
        s = torch.cat([s_cum[:-1] - s_cum[1:], s_cum[-1:]], dim=0)
        return s
