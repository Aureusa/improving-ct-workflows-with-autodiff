import os

import torch
import numpy as np
from skimage.filters import threshold_multiotsu

from .utils import tanh_thresholding
from .barba_3D_phantom_1 import astra_forward_project_differentiable

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class ISP(torch.nn.Module):
    def __init__(self, n_angles=360, number_of_materials=2, gamma=1.0, energy_bins=358, energy_chunk_size=16, voxel_size=0.5/128, mu_eff_mode="fluence", spectral_perturb=0.0, spectral_perturb_seed=0, smooth_sigma=0.0, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(ISP, self).__init__()
        self.current_iter = 0
        self.n_angles = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins = energy_bins
        self.energy_chunk_size = energy_chunk_size
        self.voxel_size = voxel_size
        self.mu_eff_mode = mu_eff_mode  # 'fluence' (original) | 'transmission'
        self.smooth_sigma = smooth_sigma  # Gaussian sigma [vox] to denoise recon before segmentation (0 = off)
        self._device = device

        self._t_initialized = False

        fluence_vals = np.load(os.path.join(_DATA_DIR, "fluence.npy"))
        mu_vals = np.load(os.path.join(_DATA_DIR, "mu_values.npy"))

        if mu_vals.shape != (number_of_materials, energy_bins):
            raise ValueError(f"mu_values.npy has shape {mu_vals.shape}, but expected ({number_of_materials}, {energy_bins})")
        if fluence_vals.shape != (energy_bins,):
            raise ValueError(f"fluence.npy has shape {fluence_vals.shape}, but expected ({energy_bins},)")

        self._I = torch.nn.Parameter(torch.from_numpy(np.array(fluence_vals)).float(), requires_grad=True) # (energy_bins,) -- spectral photon fluence
        self._mu = torch.nn.Parameter(torch.from_numpy(mu_vals).float(), requires_grad=True) # (number_of_materials, energy_bins)

        # Optionally perturb the spectral init AWAY from ground truth (per-bin
        # multiplicative 1 +/- spectral_perturb). ISP loads I/mu from the exact spectrum
        # that generated the data (inverse crime), so the optimisation otherwise STARTS
        # at truth and "loss goes down" proves nothing. A nonzero perturb makes spectrum
        # recovery an honest test. Mirrors 2-D ISP2D (see isp_2d.py). Off by default.
        if spectral_perturb > 0:
            g = torch.Generator().manual_seed(spectral_perturb_seed)
            I_fac  = 1.0 + spectral_perturb * (2.0 * torch.rand(self._I.shape,  generator=g) - 1.0)
            mu_fac = 1.0 + spectral_perturb * (2.0 * torch.rand(self._mu.shape, generator=g) - 1.0)
            self._I  = torch.nn.Parameter((self._I.detach()  * I_fac ).clamp_min(0.0), requires_grad=True)
            self._mu = torch.nn.Parameter((self._mu.detach() * mu_fac).clamp_min(0.0), requires_grad=True)

        self._t = torch.nn.Parameter(torch.rand(self.number_of_materials), requires_grad=True) # (number_of_materials,)
        self._gamma = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False) # (1,)

    def forward(self, reconstruction):
        # Ensure reconstruction is on the same device as the parameters
        reconstruction = reconstruction.to(self._device)
        I_sim = self._compute_I_sim(reconstruction) # (n_pixels, n_angles, n_pixels)
        A_sim = self._compute_A_sim(I_sim) # (n_pixels, n_angles, n_pixels)
        return A_sim

    def _compute_A_sim(self, I_sim):
        """
        A = -log(I_sim / I_0)
        I_0 is the intensity of the X-ray beam without any attenuation, which can be approximated by the sum of the intensities across all energy bins.
        """
        eps = 1e-8
        I_0 = torch.sum(self.I).clamp_min(eps) # (1,)
        ratio = (I_sim / I_0).clamp_min(eps)
        A_sim = -torch.log(ratio) # (n_pixels, n_angles, n_pixels)
        return A_sim

    def _material_path_sinograms(self, reconstruction):
        """
        Soft-segment the volume and forward-project each material mask to a
        path-length sinogram.

        reconstruction : (n_pixels, n_pixels, n_pixels)
        returns        : As_n (number_of_materials, n_pixels, n_angles, n_pixels) [cm]
        """
        s = self._s(reconstruction)  # (M, n_pixels, n_pixels, n_pixels)
        l_list = []
        for n in range(self.number_of_materials):
            l_n = astra_forward_project_differentiable(s[n], self.n_angles)  # (n_pixels, n_angles, n_pixels)
            l_list.append(l_n)
        l = torch.stack(l_list, dim=0)  # (M, n_pixels, n_angles, n_pixels)
        # Scale from voxel counts to physical path length (matching calculate_I's scale)
        return l * self.voxel_size       # (M, n_pixels, n_angles, n_pixels) [cm]

    def _I_sim_from_As(self, As_n):
        """
        Polychromatic intensity sum_e I_e*exp(-sum_m mu(e,m)*As_n[m]) from precomputed
        material path sinograms, summed in energy chunks to avoid materialising the
        full (E, M, n_pixels, n_angles, n_pixels) tensor.

        As_n : (M, n_pixels, n_angles, n_pixels)  ->  (n_pixels, n_angles, n_pixels)
        """
        mu = self.mu.permute(1, 0).to(self._device)  # (energy_bins, number_of_materials)
        I = self.I.to(self._device)                  # (energy_bins,)
        I_sim = torch.zeros_like(As_n[0])            # (n_pixels, n_angles, n_pixels)

        for start in range(0, self.energy_bins, self.energy_chunk_size):
            end = min(start + self.energy_chunk_size, self.energy_bins)
            mu_chunk = mu[start:end]  # (chunk, number_of_materials)
            exponent_chunk = torch.einsum('em,mrap->erap', mu_chunk, As_n)
            intensity_chunk = I[start:end].view(-1, 1, 1, 1) * torch.exp(-exponent_chunk)
            I_sim = I_sim + intensity_chunk.sum(dim=0)
        return I_sim

    def _compute_I_sim(self, reconstruction):
        """Polychromatic intensity from a 3-D volume: segment -> project -> spectrum sum.

        reconstruction : (n_pixels, n_pixels, n_pixels) -> (n_pixels, n_angles, n_pixels)
        """
        As_n = self._material_path_sinograms(reconstruction)
        return self._I_sim_from_As(As_n)

    def compute_corrected_sinogram(self, reconstruction, y_meas=None, correction_mode="replace"):
        """
        Build the sinogram to reconstruct for a beam-hardening-free volume.

        From the current (segmented) volume and learned params it computes:
            As_n   -- per-material path sinograms (M, n_pixels, n_angles, n_pixels)
            y_poly -- polychromatic simulation -log(sum_e I_e e^{-sum mu*As} / sumI)
            mu_eff -- monochromatic-equivalent attenuation per material (_effective_mu)
            y_mono = sum_m mu_eff[m]*As_n[m]   (linear in path length -> no cupping)

        correction_mode='replace'  -> return y_mono (synthetic mono sinogram, original).
        correction_mode='residual' -> return y_meas + (y_mono - y_poly): correct the
            measured sinogram by the modelled BH difference (original autodiffCT approach).
        """
        with torch.no_grad():
            reconstruction = reconstruction.to(self._device)
            As_n   = self._material_path_sinograms(reconstruction)   # (M, n_pixels, n_angles, n_pixels)
            y_poly = self._compute_A_sim(self._I_sim_from_As(As_n))  # (n_pixels, n_angles, n_pixels)

            mu     = self.mu.to(self._device)   # (M, E)
            I      = self.I.to(self._device)    # (E,)
            mu_eff = self._effective_mu(mu, I, As_n, y_poly=y_poly)  # (M,)
            y_mono = torch.einsum('m,mrap->rap', mu_eff, As_n)       # (n_pixels, n_angles, n_pixels)

            if correction_mode == "residual":
                if y_meas is None:
                    raise ValueError("correction_mode='residual' requires y_meas")
                return y_meas.to(self._device) + (y_mono - y_poly)
            return y_mono

    def compute_monochromatic_sinogram(self, reconstruction):
        """Backward-compatible alias: the 'replace' correction (returns y_mono)."""
        return self.compute_corrected_sinogram(reconstruction, correction_mode="replace")

    def _effective_mu(self, mu, I, As_n, y_poly=None):
        """
        Monochromatic-equivalent linear attenuation per material, mu_eff (M,).
        See the 2-D ISP2D._effective_mu for the full rationale.

        'fluence' (original): mu_eff[m] = sum_e I_e*mu(e,m) / sum_e I_e -- dominated by
            the absorbed soft spectral tail, so it can be wildly inflated.
        'transmission': weight by detected photons through a representative object
            path, w_e = I_e*exp(-sum_m mu(e,m)*L_rep[m]) -> physical mu_eff w/o filtration.
        'lstsq' (original autodiffCT): least-squares regression of y_poly onto the
            material path sinograms (mu_eff = pinv(B)*V). Measurement-weighted, uses
            no spectrum average -> immune to the soft-tail inflation.
        Dimension-agnostic: As_n is (M, *rays).
        """
        mode = getattr(self, "mu_eff_mode", "fluence")

        if mode == "lstsq":
            # mu_eff = argmin_a ||sum_m a_m*As_n[m] - y_poly||^2 = pinv(B)*V,
            #   B[i,j] = <As_i, As_j>,  V[i] = <As_i, y_poly>.
            if y_poly is None:
                raise ValueError("mu_eff_mode='lstsq' requires y_poly")
            A_flat = As_n.reshape(As_n.shape[0], -1)   # (M, K)
            y_flat = y_poly.reshape(-1)                 # (K,)
            B = A_flat @ A_flat.t()                     # (M, M)
            V = A_flat @ y_flat                         # (M,)
            return torch.linalg.pinv(B) @ V             # (M,)

        if mode == "transmission":
            total_path = As_n.sum(dim=0)
            object_rays = total_path > 1e-6
            if bool(object_rays.any()):
                L_rep = As_n[:, object_rays].mean(dim=1)
            else:
                L_rep = As_n.reshape(As_n.shape[0], -1).mean(dim=1)
            attn = torch.einsum("me,m->e", mu, L_rep)
            w = I * torch.exp(-attn)
            return (mu * w.unsqueeze(0)).sum(dim=1) / w.sum().clamp_min(1e-8)
        return (mu * I.unsqueeze(0)).sum(dim=1) / I.sum().clamp_min(1e-8)

    def _gaussian_blur3d(self, x, sigma):
        """
        Separable 3-D Gaussian blur of a single (D,H,W) volume, edge-replicated so the
        object boundary isn't darkened. Denoises the recon before segmentation (see _s).
        Mirrors 2-D ISP2D._gaussian_blur with an extra spatial axis.
        """
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        k = k / k.sum()
        xb = x[None, None]                                              # (1,1,D,H,W)
        xb = torch.nn.functional.pad(xb, (radius,) * 6, mode="replicate")
        xb = torch.nn.functional.conv3d(xb, k.view(1, 1, -1, 1, 1))     # along depth
        xb = torch.nn.functional.conv3d(xb, k.view(1, 1, 1, -1, 1))     # along height
        xb = torch.nn.functional.conv3d(xb, k.view(1, 1, 1, 1, -1))     # along width
        return xb[0, 0]

    def _s(self, x):
        """
        Soft material-fraction field via cumulative tanh thresholding (3-D).

        x : (n_pixels, n_pixels, n_pixels) reconstruction volume
        returns : (number_of_materials, n_pixels, n_pixels, n_pixels)
        """
        # Optional denoising BEFORE segmentation (mirrors 2-D ISP2D._s). A noisy recon
        # makes the tanh thresholds flip labels on per-voxel noise -> speckled masks ->
        # wrong path lengths -> corrupted fit + correction. Gaussian-smoothing the recon
        # first stabilises the masks. Constant input (no grad through x). Off by default.
        if getattr(self, "smooth_sigma", 0.0) and self.smooth_sigma > 0:
            x = self._gaussian_blur3d(x, self.smooth_sigma)

        # Normalise the recon to [0,1] before thresholding so that gamma is decoupled
        # from the physical recon scale (~0.002-0.008). On raw values gamma*(x-t) stays
        # tiny and tanh never saturates -> mushy masks -> the forward model cannot match
        # the data. Working in [0,1] lets a fixed gamma produce crisp masks. The recon
        # is a constant input (no grad through x), so its min/max are safe.
        # (2-D Sec 8.3 fix, ported to 3-D.)
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min).clamp_min(1e-8)

        if not self._t_initialized:
            # Otsu on the *normalised* recon, so learnable thresholds live in [0,1] too.
            thresholds = threshold_multiotsu(x_norm.cpu().detach().numpy(),
                                             classes=self.number_of_materials + 1,
                                             nbins=128)
            self._t = torch.nn.Parameter(
                torch.tensor(thresholds, device=self._device, dtype=x.dtype),
                requires_grad=True,
            )
            self.add_param(self._t, "t", trainable=True)
            self._t_initialized = True
            t = self.t
        else:
            t = self.t

        # Keep thresholds ascending so the exclusive-mask subtraction can't go negative
        # if Adam reorders them (torch.sort is differentiable).
        t, _ = torch.sort(t)

        t = t.reshape(self.number_of_materials, 1, 1, 1)
        t = t.expand(self.number_of_materials, x.shape[0], x.shape[1], x.shape[2])

        # s_cum[n] = 0.5(1 + tanh(gamma*(x_norm - t[n])))  -- crisp now that x_norm in [0,1]
        s_cum = tanh_thresholding(x_norm, t, self.gamma)
        # exclusive indicators: s[n] = s_cum[n] - s_cum[n+1]  (last stays)
        s = torch.cat([s_cum[:-1] - s_cum[1:], s_cum[-1:]], dim=0)
        return s
        