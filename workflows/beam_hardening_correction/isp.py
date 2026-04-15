import torch

from .utils import tanh_thresholding
from .barba_3D_phantom_1 import astra_forward_project


class ISP(torch.nn.Module):
    def __init__(self, n_angles=360, number_of_materials=2, gamma=1.0, energy_bins=358, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(ISP, self).__init__()
        self.current_iter = 0
        self.n_angles = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins = energy_bins
        self._device = device

        self.I = torch.nn.Parameter(torch.rand(self.energy_bins), requires_grad=True) # (energy_bins,)
        self.mu = torch.nn.Parameter(torch.rand(self.number_of_materials, self.energy_bins), requires_grad=True) # (number_of_materials, energy_bins)
        self.t = torch.nn.Parameter(torch.rand(self.number_of_materials), requires_grad=True) # (number_of_materials,)
        self.gamma = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False) # (1,)

    def forward(self, reconstruction):
        # Ensure reconstruction is on the same device as the parameters
        reconstruction = reconstruction.to(self._device)
        I_sim = self._compute_I_sim(reconstruction) # (n_angles, n_pixels, n_pixels)
        A_sim = self._compute_A_sim(I_sim) # (n_angles, n_pixels, n_pixels)
        return A_sim

    def _compute_A_sim(self, I_sim):
        """
        A = -log(I_sim / I_0)
        I_0 is the intensity of the X-ray beam without any attenuation, which can be approximated by the sum of the intensities across all energy bins.
        """
        I_0 = torch.sum(self.I) # (1,)
        A_sim = -torch.log(I_sim / I_0) # (n_angles, n_pixels, n_pixels)
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

        # detach s and turn to numpy for astra
        s_detached = s.detach().cpu().numpy() # (number_of_materials, n_pixels, n_pixels, n_pixels)

        # Forward project s_n to get the line integrals for each material n
        # This will give us the line integrals for each material n along each ray in the sinogram
        l = torch.zeros( # As_n
            (self.number_of_materials,
             self.n_angles,
             reconstruction.shape[1],
             reconstruction.shape[2])
        ) # (number_of_materials, n_angles, n_pixels, n_pixels)
        for n in range(self.number_of_materials):
            l_n = astra_forward_project(s_detached[n], self.n_angles) # (n_pixels, n_angles, n_pixels)

            # Convert to tensor
            l_n = torch.from_numpy(l_n).to(self._device)

            # Reshape l_n to (n_angles, n_pixels, n_pixels) and store in l
            l[n] = l_n.reshape(
                self.n_angles,
                reconstruction.shape[1],
                reconstruction.shape[2]
            ) # (n_angles, n_pixels, n_pixels)

        As_n = l.to(self._device) # (number_of_materials, n_angles, n_pixels, n_pixels)

        # Switch mu to (energy_bins, number_of_materials) for easier broadcasting
        mu = self.mu.permute(1, 0).to(self._device) # (energy_bins, number_of_materials)
        
        # unsqueeze mu to (energy_bins, number_of_materials, 1, 1, 1) for broadcasting
        mu = mu.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (energy_bins, number_of_materials, 1, 1, 1)

        exponent = torch.sum(mu * As_n, axis=1) # (energy_bins, n_angles, n_pixels, n_pixels)
        exponential = torch.exp(-exponent)

        I = self.I.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (energy_bins, 1, 1, 1)

        I_sim = torch.sum(I*exponential, axis=0) # (n_angles, n_pixels, n_pixels)
        return I_sim

        # # Denote by J the number of pixels on the reconstruction grid.
        # # We assume that each pixel contains exactly one material defined by a vector of
        # # binary variables s = {sn, j}, with sn,j = 1 if voxel j contains material n, and sn,j = 0 otherwise.
        # # Consider projection lines, denoted by an index i = 1,…, D, for which an intensity
        # # measure is obtained at the detector, and denote by li,j the intersection length of ray i with pixel j.
        # # We have sum over J (sn,j * li,j)
        # sum_over_j = torch.zeros(
        #     (self.number_of_materials, self.n_angles, reconstruction.shape[1], reconstruction.shape[2])
        # ) # (number_of_materials, n_angles, n_pixels, n_pixels) 
        # for n in range(self.number_of_materials):
        #     for j in range(reconstruction.shape[0]):
        #         sum_over_j[n] += s[n, j] * l[n, :, j, :] # (n_angles, n_pixels, n_pixels)

        # # Now the exponent is sum_over_materials(mu_n * sum_over_pixels(s_n * reconstruction_pixel_value))
        # # This is just sum_over_materials(mu_n * sum_over_j)
        # exponent = torch.zeros(
        #     (self.n_angles, reconstruction.shape[1], reconstruction.shape[2])
        # ) # (n_angles, n_pixels, n_pixels)
        # for n in range(self.number_of_materials):
        #     exponent += self.mu[n] * sum_over_j[n] # (n_angles, n_pixels, n_pixels)

        # # Finally, we can compute I_sim
        # I_sim = torch.zeros(
        #     (self.n_angles, reconstruction.shape[1], reconstruction.shape[2])
        # ) # (n_angles, n_pixels, n_pixels)
        # for e in range(self.energy_bins):
        #     I_sim += self.I[e] * torch.exp(-exponent) # (n_angles, n_pixels, n_pixels)
        # return I_sim

    def _s(self, x):
        """
        Since x is 3D volume (n_pixels, n_pixels, n_pixels)
        t is (number_of_materials,)
        gamma is scalar
        The output should be (number_of_materials, n_pixels, n_pixels, n_pixels)
        """
        # We need broadcasting to apply the tanh_thresholding function to each material separately
        t = self.t.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, number_of_materials)
        t = t.reshape(self.number_of_materials, 1, 1, 1) # (number_of_materials, 1, 1, 1)
        t = t.expand(t.shape[0], x.shape[0], x.shape[1], x.shape[2]) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        return tanh_thresholding(x, t, self.gamma) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        