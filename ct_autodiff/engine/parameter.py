from typing import Union
import torch


class Parameter:
    def __init__(self, data: Union[float, list, torch.Tensor], symbol: str, trainable=True):
        """
        A wrapper around torch.Tensor that includes metadata about the parameter,
        such as its name and whether it is trainable.

        :param data: The initial value of the parameter, can be a scalar, list, or torch.Tensor.
        :type data: Union[float, list, torch.Tensor]
        :param symbol: A unique identifier for the parameter, used for referencing in blocks.
        :type symbol: str
        :param trainable: Whether the parameter should be updated during training. Defaults to True.
        :type trainable: bool, optional
        """
        if isinstance(data, torch.Tensor):
            tensor = data.clone().detach()
            tensor.requires_grad_(trainable)
        else:
            tensor = torch.tensor(
                data,
                dtype=torch.float32,
                requires_grad=trainable
            )
        self.tensor = tensor
        self.trainable = trainable
        self.symbol = symbol

    @property
    def grad(self):
        """Returns the gradient of the parameter if it is trainable, otherwise None."""
        return self.tensor.grad

    def zero_grad(self):
        """Sets the gradient of the parameter to None."""
        self.tensor.grad = None

    def item(self):
        """Returns the value of the parameter as a Python scalar."""
        return self.tensor.item()
    
    def to(self, device):
        """Moves the parameter to the specified device."""
        self.tensor = self.tensor.to(device).detach().requires_grad_(self.trainable)
        return self

    def __repr__(self):
        return f"Parameter(symbol={self.symbol}, trainable={self.trainable}, data={self.tensor.detach()})"

    def __str__(self):
        return f"{self.symbol}: {self.tensor.detach()} (trainable={self.trainable})"
