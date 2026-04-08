import torch

from .utils import tanh_thresholding


class ISP(torch.nn.Module):
    def __init__(self, n_angles=360, number_of_materials=2, gamma=1.0, energy_bins=358):
        super(ISP, self).__init__()
        self.current_iter = 0
        self.n_angles = n_angles
        self.number_of_materials = number_of_materials
        self.energy_bins = energy_bins

        self.I = torch.nn.Parameter(torch.rand(self.energy_bins), requires_grad=True) # (energy_bins,)
        self.mu = torch.nn.Parameter(torch.rand(self.number_of_materials, self.energy_bins), requires_grad=True) # (number_of_materials, energy_bins)
        self.t = torch.nn.Parameter(torch.rand(self.number_of_materials), requires_grad=True) # (number_of_materials,)
        self.gamma = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False) # (1,)

    def forward(self, reconstruction):
        # TODO: Implement 
        pass

    def _compute_I_sim(self, reconstruction):
        # reconstruction: (n_pixels, n_pixels, n_pixels) 3D volume
        s = self._s(reconstruction) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        
        # I = sum_over_E_bins(
        #   (I_e * exp(-sum_over_materials(mu_n * sum_over_pixels(s_n * reconstruction_pixel_value))))
        # )

        # TODO: Implement the rest
        pass

    def _s(self, x):
        # Since x is 3D volume (n_pixels, n_pixels, n_pixels)
        # t is (number_of_materials,)
        # gamma is scalar
        # The output should be (n_pixels, n_pixels, n_pixels, number_of_materials)

        # We need broadcasting to apply the tanh_thresholding function to each material separately
        t = self.t.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, number_of_materials)
        t = t.reshape(self.number_of_materials, 1, 1, 1) # (number_of_materials, 1, 1, 1)
        t = t.expand(t.shape[0], x.shape[0], x.shape[1], x.shape[2]) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        return tanh_thresholding(x, t, self.gamma) # (number_of_materials, n_pixels, n_pixels, n_pixels)
        
