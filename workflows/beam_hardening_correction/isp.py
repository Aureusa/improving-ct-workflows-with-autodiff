import torch
import numpy as np
from skimage.filters import threshold_multiotsu

from .utils import tanh_thresholding
from .barba_3D_phantom_1 import astra_forward_project_differentiable


class ISP(torch.nn.Module):
    def __init__(self, n_angles=360, number_of_materials=2, gamma=1.0, energy_bins=358, energy_chunk_size=16, voxel_size=0.5/128, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(ISP, self).__init__()
        self.current_iter = 0
        self.n_angles = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins = energy_bins
        self.energy_chunk_size = energy_chunk_size
        self.voxel_size = voxel_size
        self._device = device

        self._t_initialized = False

        energy_bins_vals = np.load("/home/s4861264/CIT_project/workflows/beam_hardening_correction/energy_bins.npy")
        mu_vals = np.load("/home/s4861264/CIT_project/workflows/beam_hardening_correction/mu_values.npy")

        if mu_vals.shape != (number_of_materials, energy_bins):
            raise ValueError(f"mu_values.npy has shape {mu_vals.shape}, but expected ({number_of_materials}, {energy_bins})")
        if energy_bins_vals.shape != (energy_bins,):
            raise ValueError(f"energy_bins.npy has shape {energy_bins_vals.shape}, but expected ({energy_bins},)")
        
        self.I = torch.nn.Parameter(torch.from_numpy(np.array(energy_bins_vals)).float(), requires_grad=True) # (energy_bins,)
        self.mu = torch.nn.Parameter(torch.from_numpy(mu_vals).float(), requires_grad=True) # (number_of_materials, energy_bins)
        self.t = torch.nn.Parameter(torch.rand(self.number_of_materials), requires_grad=True) # (number_of_materials,)
        self.gamma = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False) # (1,)

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

    def _compute_I_sim(self, reconstruction):
        """
        I = sum_over_E_bins(
          (I_e * exp(-sum_over_materials(mu_n * sum_over_pixels(s_n * reconstruction_pixel_value))))
        )

        reconstruction: (n_pixels, n_pixels, n_pixels) 3D volume
        """
        # Get s_n for each material n
        s = self._s(reconstruction) # (number_of_materials, n_pixels, n_pixels, n_pixels)

        # Forward project s_n to get the line integrals for each material n
        # This will give us the line integrals for each material n along each ray in the sinogram
        l_list = []
        for n in range(self.number_of_materials):
            l_n = astra_forward_project_differentiable(s[n], self.n_angles) # (n_pixels, n_angles, n_pixels)

            # Reshape l_n to (n_angles, n_pixels, n_pixels)
            l_n = l_n.reshape(
                self.n_angles,
                reconstruction.shape[1],
                reconstruction.shape[2]
            ) # (n_angles, n_pixels, n_pixels)
            l_list.append(l_n)

        l = torch.stack(l_list, dim=0) # (number_of_materials, n_angles, n_pixels, n_pixels)

        # Scale As_n from voxel counts to physical path length (matching calculate_I's scale)
        As_n = l * self.voxel_size # (number_of_materials, n_angles, n_pixels, n_pixels)

        # Compute the energy contribution in chunks so we do not materialize the
        # full (energy_bins, number_of_materials, n_angles, n_pixels, n_pixels)
        # broadcasted tensor on the GPU.
        mu = self.mu.permute(1, 0).to(self._device) # (energy_bins, number_of_materials)
        I = self.I.to(self._device) # (energy_bins,)
        I_sim = torch.zeros_like(As_n[0]) # (n_angles, n_pixels, n_pixels)

        for start in range(0, self.energy_bins, self.energy_chunk_size):
            end = min(start + self.energy_chunk_size, self.energy_bins)
            mu_chunk = mu[start:end] # (chunk, number_of_materials)
            exponent_chunk = torch.einsum('em,mabp->eabp', mu_chunk, As_n)
            intensity_chunk = I[start:end].view(-1, 1, 1, 1) * torch.exp(-exponent_chunk)
            I_sim = I_sim + intensity_chunk.sum(dim=0)

        # Reshape I_sim back to (n_pixels, n_angles, n_pixels)
        I_sim = I_sim.permute(1, 0, 2) # (n_pixels, n_angles, n_pixels)
        return I_sim

    def _s(self, x):
        """
        Since x is 3D volume (n_pixels, n_pixels, n_pixels)
        t is (number_of_materials,)
        gamma is scalar
        The output should be (number_of_materials, n_pixels, n_pixels, n_pixels)
        """
        if not self._t_initialized:
            thresholds = threshold_multiotsu(x.cpu().detach().numpy(),
                                         classes=self.number_of_materials+1,
                                         nbins=128)
            self.t = torch.nn.Parameter(torch.tensor(thresholds, device=self._device, dtype=x.dtype), requires_grad=True)
        # We need broadcasting to apply the tanh_thresholding function to each material separately
        t = self.t.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, number_of_materials)
        t = t.reshape(self.number_of_materials, 1, 1, 1) # (number_of_materials, 1, 1, 1)
        t = t.expand(t.shape[0], x.shape[0], x.shape[1], x.shape[2]) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        return tanh_thresholding(x, t, self.gamma) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        