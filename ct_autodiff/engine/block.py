from abc import ABC, abstractmethod

from .parameter import Parameter
from ct_autodiff.utils.formating import box_text

class Block(ABC):
    @abstractmethod
    def __init__(self):
        """
        Initializes a block with a name and an empty parameter dictionary.
        
        :param name: A unique identifier for the block, used for referencing in workflows.
        :type name: str
        """
        self.name = self.__class__.__name__
        self._params: dict[str, Parameter] = {}

    @abstractmethod
    def execute(self):
        """Builds computation using param.tensor"""
        pass

    def add_param(self, data, symbol, trainable=True):
        """
        Adds a parameter to the block.

        :param data: The initial data for the parameter.
        :type data: Any
        :param symbol: The name of the parameter.
        :type symbol: str
        :param trainable: Whether the parameter is trainable. Defaults to True.
        :type trainable: bool, optional
        """
        self._params[symbol] = Parameter(data, symbol, trainable)

    def __getattr__(self, name):
        """
        Allows access to parameters as attributes of the block.
        
        :param name: The name of the parameter to access.
        :type name: str
        :return: The tensor of the parameter if it exists.
        :rtype: torch.Tensor
        :raises AttributeError: If the parameter does not exist in the block.
        """
        params = self.__dict__.get("_params", {})
        if name in params:
            return params[name].tensor

        # Delegate to the next class in MRO when mixing Block with classes
        # that implement their own attribute lookup (e.g., torch.nn.Module).
        parent_getattr = getattr(super(), "__getattr__", None)
        if parent_getattr is not None:
            return parent_getattr(name)

        raise AttributeError(f"{name} not found in parameters")

    def parameters(self):
        """
        Yields the tensors of all trainable parameters in the block.
        This is to be compatible with optimizers that expect an iterable of tensors.
        """
        for p_name, p in self._params.items():
            if p.trainable:
                yield (p_name, p.tensor)

    def to(self, device):
        """Moves all parameters in the block to the specified device."""
        for p in self._params.values():
            p.to(device)
        return self
    
    def __repr__(self):
        return f"Block(name={self.name}, params={list(self._params.keys())})"
    
    def __str__(self):
        param_str = "\n".join(f"    {v}" for _, v in self._params.items())
        string = f"{self.name}:\n{param_str}"
        return box_text(string)
