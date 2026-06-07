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
        mu_eff_mode: str = "fluence",
        spectral_perturb: float = 0.0,
        spectral_perturb_seed: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ISP2D, self).__init__()

        self.current_iter      = 0
        self.n_angles          = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins       = energy_bins
        self.energy_chunk_size = energy_chunk_size
        self.voxel_size        = voxel_size
        self.mu_eff_mode       = mu_eff_mode   # 'fluence' (original) | 'transmission'
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

        # Optionally perturb the spectral init AWAY from ground truth (per-bin
        # multiplicative ±spectral_perturb). ISP2D otherwise loads I/mu from the
        # exact spectrum that generated the data, so with freeze_spectral=False the
        # optimisation would still *start* at truth — "loss goes down" then proves
        # nothing (README §6). A nonzero perturb makes recovery an honest test.
        if spectral_perturb > 0:
            g = torch.Generator().manual_seed(spectral_perturb_seed)
            I_fac  = 1.0 + spectral_perturb * (2.0 * torch.rand(self._I.shape,  generator=g) - 1.0)
            mu_fac = 1.0 + spectral_perturb * (2.0 * torch.rand(self._mu.shape, generator=g) - 1.0)
            self._I  = torch.nn.Parameter((self._I.detach()  * I_fac ).clamp_min(0.0), requires_grad=True)
            self._mu = torch.nn.Parameter((self._mu.detach() * mu_fac).clamp_min(0.0), requires_grad=True)

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

            A_mono = Σ_m( mu_eff[m] · As_n[m] )

        mu_eff is the monochromatic-equivalent attenuation per material; see
        _effective_mu for the 'fluence' (original) vs 'transmission' weighting.
        A_mono is linear in path length, so reconstructing it has no cupping.

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
            As_n = l * self.voxel_size           # (M, n_angles, n_pixels)  [cm]

            mu     = self.mu.to(self._device)    # (M, E)
            I      = self.I.to(self._device)     # (E,)
            mu_eff = self._effective_mu(mu, I, As_n)   # (M,)

            # A_mono[a, p] = Σ_m( mu_eff[m] · As_n[m, a, p] )
            A_mono = torch.einsum("m,map->ap", mu_eff, As_n)  # (n_angles, n_pixels)

        return A_mono

    def _effective_mu(
        self, mu: torch.Tensor, I: torch.Tensor, As_n: torch.Tensor
    ) -> torch.Tensor:
        """
        Monochromatic-equivalent linear attenuation per material, mu_eff (M,).

        mode='fluence' (original): mu_eff[m] = Σ_e I_e·μ(e,m) / Σ_e I_e
            — the thin-object initial slope. Dominated by the soft spectral tail
              (μ huge, but those photons are fully absorbed in the object), so it
              can be wildly inflated (Al → 53 cm⁻¹ at dk=2 unfiltered) and the
              corrected reconstruction explodes.

        mode='transmission': weight each energy bin by the photons that actually
            survive a representative object path, T_e = exp(−Σ_m μ(e,m)·L_rep[m]),
            so absorbed soft photons get ~zero weight:
                w_e = I_e · T_e ;   mu_eff[m] = Σ_e w_e·μ(e,m) / Σ_e w_e
            L_rep[m] = mean path through material m over the rays that intersect
            the object (data-driven — no arbitrary reference energy/thickness).
            Restores physical mu_eff (Al ≈ 1.5 cm⁻¹) without needing filtration.

        Dimension-agnostic in the spatial axes: As_n is (M, *rays).
        """
        if getattr(self, "mu_eff_mode", "fluence") == "transmission":
            total_path = As_n.sum(dim=0)                          # (*rays)
            object_rays = total_path > 1e-6                        # rays through the object
            if bool(object_rays.any()):
                L_rep = As_n[:, object_rays].mean(dim=1)          # (M,)
            else:
                L_rep = As_n.reshape(As_n.shape[0], -1).mean(dim=1)
            attn = torch.einsum("me,m->e", mu, L_rep)             # (E,) Σ_m μ(e,m)·L_rep[m]
            w = I * torch.exp(-attn)                               # (E,) detected-photon weight
            return (mu * w.unsqueeze(0)).sum(dim=1) / w.sum().clamp_min(1e-8)

        # default: fluence-weighted (original)
        return (mu * I.unsqueeze(0)).sum(dim=1) / I.sum().clamp_min(1e-8)

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
        # Normalise the reconstruction to [0,1] before thresholding so that gamma
        # is decoupled from the physical recon scale (~0.005-0.02). On raw values
        # gamma*(x-t) stays small and tanh never saturates → mushy masks. Working
        # in [0,1] lets a fixed gamma produce crisp masks. The recon is a constant
        # input (no grad through x), so its min/max are safe to use.
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min).clamp_min(1e-8)

        if not self._t_initialized:
            # Initialise thresholds from multi-Otsu of the *normalised* recon, so
            # the learnable thresholds live in the same [0,1] space as x_norm.
            thresholds = threshold_multiotsu(
                x_norm.cpu().detach().numpy(),
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

        # Thresholds must stay ascending or the exclusive-mask subtraction
        # (s_cum[:-1] - s_cum[1:]) below can go negative. Adam updates each t[m]
        # independently and may reorder them, so sort every forward pass.
        # torch.sort is differentiable (it just permutes the gradient).
        t, _ = torch.sort(t)

        # Broadcast t : (M,) → (M, n_pixels, n_pixels)
        t = t.reshape(self.number_of_materials, 1, 1)
        t = t.expand(self.number_of_materials, x.shape[0], x.shape[1])

        # s_cum[m] = σ( γ · (x_norm − t[m]) )  — (M, n_pixels, n_pixels)
        s_cum = tanh_thresholding(x_norm, t, self.gamma)

        # Convert cumulative to exclusive: s[m] = s_cum[m] − s_cum[m+1]
        s = torch.cat([s_cum[:-1] - s_cum[1:], s_cum[-1:]], dim=0)
        return s
