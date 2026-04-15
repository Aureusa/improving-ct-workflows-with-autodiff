import torch


class PhiLoss(torch.nn.Module):
    def __init__(self):
        super(PhiLoss, self).__init__()

    def forward(self, A_sim, A_meas):
        sqrt_sum = (A_meas - A_sim)**2
        return sqrt_sum.sum() / len(sqrt_sum)
