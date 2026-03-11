from ct_autodiff.engine.block import Block


###############################################
#          Non optimizable blocks             #
###############################################

class Reconstruct(Block):
    def __init__(self):
        super().__init__()

    def forward(self):
        # Placeholder for the actual reconstruction logic
        # This should implement the beam hardening correction algorithm
        pass

class CorrectProjection(Block):
    def __init__(self):
        super().__init__()

    def forward(self):
        # Placeholder for the actual projection correction logic
        # This should implement the correction of the projection data
        pass


###############################################
#              Optimizable blocks             #
###############################################

class SpectralProjection(Block):
    def __init__(self):
        super().__init__()

    def forward(self):
        # Placeholder for the actual spectral projection logic
        # This should implement the spectral projection step of the correction
        pass
