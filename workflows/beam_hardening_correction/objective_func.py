import torch


class PhiLoss(torch.nn.Module):
    def __init__(self):
        super(PhiLoss, self).__init__()

    def forward(self, I_meas, I_sim, I_0):
        first_log_term = torch.log(I_meas/I_0) # Maybe log is unnecessary
        second_log_term = torch.log(I_sim) # Maybe log is unnecessary

        sqrt_sum = (first_log_term - second_log_term)**2

        return sqrt_sum.sum() / len(sqrt_sum)
