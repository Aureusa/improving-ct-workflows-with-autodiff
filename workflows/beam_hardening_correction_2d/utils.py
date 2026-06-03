import torch


def tanh_thresholding(x, t, gamma):
    """
    Smooth differentiable threshold indicator via tanh.

    Args:
        x     : input tensor  (any shape)
        t     : threshold tensor, broadcast-compatible with x
        gamma : steepness scalar

    Returns:
        Tensor with values in (0, 1).
    """
    angle = gamma * (x - t)
    return 0.5 * (1 + torch.tanh(angle))
