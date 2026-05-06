import torch


class PhiLoss(torch.nn.Module):
    def __init__(self):
        super(PhiLoss, self).__init__()

    def forward(self, A_sim, A_meas):
        if not torch.is_tensor(A_sim):
            if torch.is_tensor(A_meas):
                A_sim = torch.as_tensor(A_sim, dtype=A_meas.dtype, device=A_meas.device)
            else:
                A_sim = torch.as_tensor(A_sim, dtype=torch.float32)

        if not torch.is_tensor(A_meas):
            A_meas = torch.as_tensor(A_meas, dtype=A_sim.dtype, device=A_sim.device)

        sq_err = (A_meas - A_sim) ** 2
        return sq_err.mean()
