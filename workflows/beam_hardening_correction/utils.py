import torch


def tanh_thresholding(x, t, gamma):
    """
    Applies a hyperbolic tangent thresholding function to the input tensor.

    Args:
        x (torch.Tensor): Input tensor to be thresholded.
        t (float): Threshold value that controls the steepness of the transition.

    Returns:
        torch.Tensor: Thresholded tensor with values between 0 and 1.
    """
    angle = gamma * ((x - t) / t)
    return 0.5 * (1 + torch.tanh(angle))
