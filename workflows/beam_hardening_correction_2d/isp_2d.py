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
    I   : (energy_bins,)               -- spectral photon fluence per bin
    mu  : (number_of_materials, energy_bins) -- linear attenuation [cm^-1]
    t   : (number_of_materials,)       -- Otsu-initialised soft-threshold values
    gamma : scalar                     -- steepness of the tanh thresholds (fixed)
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
        smooth_sigma: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ISP2D, self).__init__()

        self.current_iter      = 0
        self.n_angles          = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins       = energy_bins
        self.energy_chunk_size = energy_chunk_size
        self.voxel_size        = voxel_size
        self.mu_eff_mode       = mu_eff_mode   # 'fluence' | 'lstsq'
        self.smooth_sigma      = smooth_sigma  # Gaussian sigma [px] to denoise recon before segmentation (0 = off)
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

        # Learnable parameters (placeholders -- actual values loaded above)
        self._I     = torch.nn.Parameter(
            torch.from_numpy(fluence_vals).float(), requires_grad=True
        )   # (energy_bins,)
        self._mu    = torch.nn.Parameter(
            torch.from_numpy(mu_vals).float(), requires_grad=True
        )   # (number_of_materials, energy_bins)

        # Optionally perturb the spectral init AWAY from ground truth (per-bin
        # multiplicative +/-spectral_perturb). ISP2D otherwise loads I/mu from the
        # exact spectrum that generated the data, so with freeze_spectral=False the
        # optimisation would still *start* at truth.
        if spectral_perturb > 0:
            g = torch.Generator().manual_seed(spectral_perturb_seed)
            I_fac  = 1.0 + spectral_perturb * (2.0 * torch.rand(self._I.shape,  generator=g) - 1.0)
            mu_fac = 1.0 + spectral_perturb * (2.0 * torch.rand(self._mu.shape, generator=g) - 1.0)
            self._I  = torch.nn.Parameter((self._I.detach()  * I_fac ).clamp_min(0.0), requires_grad=True)
            self._mu = torch.nn.Parameter((self._mu.detach() * mu_fac).clamp_min(0.0), requires_grad=True)

        self._t     = torch.nn.Parameter(
            torch.rand(self.number_of_materials), requires_grad=True
        )   # (number_of_materials,)  -- overwritten by Otsu on first forward pass
        self._gamma = torch.nn.Parameter(
            torch.tensor(gamma), requires_grad=False
        )   # scalar

    # -- Forward --------------------------------------------------------------

    def forward(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        reconstruction : float32 tensor (n_pixels, n_pixels)

        Returns
        -------
        A_sim : float32 tensor (n_angles, n_pixels)
            Simulated polychromatic attenuation sinogram -log(I / I0).
        """
        reconstruction = reconstruction.to(self._device)
        I_sim = self._compute_I_sim(reconstruction)   # (n_angles, n_pixels)
        A_sim = self._compute_A_sim(I_sim)            # (n_angles, n_pixels)
        return A_sim

    # -- Internal helpers ------------------------------------------------------

    def _compute_A_sim(self, I_sim: torch.Tensor) -> torch.Tensor:
        """-log(I_sim / I0)  -- same formula as 3-D."""
        eps  = 1e-8
        I_0  = torch.sum(self.I).clamp_min(eps)          # scalar
        return -torch.log((I_sim / I_0).clamp_min(eps))  # (n_angles, n_pixels)

    def _material_path_sinograms(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """
        Soft-segment the recon and forward-project each material mask to a
        path-length sinogram.

        returns : As_n (M, n_angles, n_pixels)  [cm]
        """
        s = self._s(reconstruction)  # (M, n_pixels, n_pixels)
        l_list = []
        for n in range(self.number_of_materials):
            l_n = astra_forward_project_2d_differentiable(s[n], self.n_angles)
            l_list.append(l_n)                       # (n_angles, n_pixels)
        l = torch.stack(l_list, dim=0)               # (M, n_angles, n_pixels)
        return l * self.voxel_size                    # (M, n_angles, n_pixels)  [cm]

    def _I_sim_from_As(self, As_n: torch.Tensor) -> torch.Tensor:
        """
        Polychromatic intensity sum_e I_e*exp(-sum_m mu(e,m)*As_n[m]) given precomputed
        material path sinograms, summed in energy chunks to avoid OOM.

        As_n : (M, n_angles, n_pixels)  ->  (n_angles, n_pixels)
        """
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

    def _compute_I_sim(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Polychromatic intensity from a recon: segment -> project -> spectrum sum."""
        As_n = self._material_path_sinograms(reconstruction)  # (M, n_angles, n_pixels)
        return self._I_sim_from_As(As_n)                      # (n_angles, n_pixels)

    def compute_corrected_sinogram(
        self,
        reconstruction: torch.Tensor,
        y_meas: torch.Tensor = None,
        correction_mode: str = "replace",
    ) -> torch.Tensor:
        """
        Build the sinogram to reconstruct for a beam-hardening-free image, from the
        segmented recon + learned params (As_n = per-material path sinograms, y_poly =
        polychromatic sim, mu_eff = effective attenuation, y_mono = sum_m mu_eff*As_n,
        linear -> no cupping).

        'replace'  -> y_mono (fully synthetic mono sinogram).
        'residual' -> y_meas + (y_mono - y_poly): correct the measured sinogram by the
                      modelled BH difference, preserving real detail.
        Returns (n_angles, n_pixels).
        """
        with torch.no_grad():
            reconstruction = reconstruction.to(self._device)
            As_n   = self._material_path_sinograms(reconstruction)   # (M, n_angles, n_pixels)
            y_poly = self._compute_A_sim(self._I_sim_from_As(As_n))  # (n_angles, n_pixels)

            mu     = self.mu.to(self._device)    # (M, E)
            I      = self.I.to(self._device)     # (E,)
            mu_eff = self._effective_mu(mu, I, As_n, y_poly=y_poly)  # (M,)
            y_mono = torch.einsum("m,map->ap", mu_eff, As_n)         # (n_angles, n_pixels)

            if correction_mode == "residual":
                if y_meas is None:
                    raise ValueError("correction_mode='residual' requires y_meas")
                return y_meas.to(self._device) + (y_mono - y_poly)
            return y_mono

    def compute_monochromatic_sinogram(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias: the 'replace' correction (returns y_mono)."""
        return self.compute_corrected_sinogram(reconstruction, correction_mode="replace")

    def _effective_mu(
        self, mu: torch.Tensor, I: torch.Tensor, As_n: torch.Tensor,
        y_poly: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Monochromatic-equivalent attenuation per material, mu_eff (M,). As_n is (M, *rays).

        'fluence': sum_e I_e*mu / sum_e I_e -- the thin-object slope. Dominated
            by the absorbed soft tail (huge mu), so it inflates (Al -> 53 cm^-1) and the
            correction explodes.
        'lstsq': least-squares regress y_poly onto As_n -- no spectrum
            average, so immune to the soft-tail inflation.
        """
        mode = getattr(self, "mu_eff_mode", "fluence")

        if mode == "lstsq":
            # mu_eff = argmin_a || sum_m a_m*As_n[m] - y_poly ||^2  =  pinv(B)*V,
            #   B[i,j] = <As_i, As_j>,  V[i] = <As_i, y_poly>.
            # Immune to soft-tail inflation: it never weights by the spectrum.
            if y_poly is None:
                raise ValueError("mu_eff_mode='lstsq' requires y_poly")
            A_flat = As_n.reshape(As_n.shape[0], -1)        # (M, K)
            y_flat = y_poly.reshape(-1)                      # (K,)
            B = A_flat @ A_flat.t()                          # (M, M)
            V = A_flat @ y_flat                              # (M,)
            return torch.linalg.pinv(B) @ V                  # (M,)

        # default: fluence-weighted
        return (mu * I.unsqueeze(0)).sum(dim=1) / I.sum().clamp_min(1e-8)

    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Separable 2-D Gaussian blur of a single (H, W) image, edge-replicated so
        the object boundary isn't darkened. Used to denoise the recon before
        segmentation (see _s). Cheap and differentiable (not that grad is needed).
        """
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        k = k / k.sum()
        xb = x[None, None]                                                   # (1,1,H,W)
        xb = torch.nn.functional.pad(xb, (radius, radius, radius, radius), mode="replicate")
        xb = torch.nn.functional.conv2d(xb, k.view(1, 1, 1, -1))             # horizontal pass
        xb = torch.nn.functional.conv2d(xb, k.view(1, 1, -1, 1))             # vertical pass
        return xb[0, 0]

    def _s(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft material-fraction field via cumulative tanh thresholding.

        Parameters
        ----------
        x : float32 tensor (n_pixels, n_pixels)  -- reconstruction slice

        Returns
        -------
        s : float32 tensor (M, n_pixels, n_pixels)
            s[m] ~= 1 where the voxel belongs to material m, with smooth transitions.
        """
        # Optional denoising BEFORE segmentation. A noisy recon makes the tanh
        # thresholds flip labels on per-voxel noise (a voxel near a threshold
        # randomly crosses it) -> speckled masks -> wrong path lengths -> the fit and
        # the mu_eff/correction are both corrupted. Gaussian-smoothing the recon
        # first stabilises the masks. The recon is a constant input (no grad through
        # x), so this is pure preprocessing. Off by default (smooth_sigma=0).
        if getattr(self, "smooth_sigma", 0.0) and self.smooth_sigma > 0:
            x = self._gaussian_blur(x, self.smooth_sigma)

        
        # Normalise the recon to [0,1] before thresholding so that gamma is decoupled
        # from the physical recon scale. On raw values gamma*(x-t) stays
        # tiny and tanh never saturates -> mushy masks -> the forward model cannot match
        # the data. 
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min).clamp_min(1e-8)

        if not self._t_initialized:
            # Initialise thresholds from multi-Otsu of the normalised recon, so
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

        # Broadcast t : (M,) -> (M, n_pixels, n_pixels)
        t = t.reshape(self.number_of_materials, 1, 1)
        t = t.expand(self.number_of_materials, x.shape[0], x.shape[1])

        # s_cum[m] = sigma( gamma * (x_norm - t[m]) )  -- (M, n_pixels, n_pixels)
        s_cum = tanh_thresholding(x_norm, t, self.gamma)

        # Convert cumulative to exclusive: s[m] = s_cum[m] - s_cum[m+1]
        s = torch.cat([s_cum[:-1] - s_cum[1:], s_cum[-1:]], dim=0)
        return s
