import torch

from ct_autodiff.engine.workflow import Workflow

from .blocks import Reconstruct, CorrectProjection, SpectralProjection
from .objective_func import PhiLoss


class BeamHardeningCorrection(Workflow):
    def __init__(self, optim_steps: int, I_0, lr: float = 0.01):
        super().__init__()
        self.add_block(Reconstruct)
        self.add_block(CorrectProjection)
        self.add_block(SpectralProjection)

        self._optim_steps = optim_steps
        self._loss_fn = PhiLoss()
        self._I_0 = I_0

        self._optim = torch.optim.Adam(self.parameters(), lr=lr)

    def run(self, input_data):
        """
        Executes the workflow by sequentially running each block on the output of the previous block.
        
        :param input_data: The initial input data for the workflow, which will be passed to the first block.
        :type input_data: Any
        :return: The output of the final block in the workflow after processing the input data through
                 all blocks.
        :rtype: Any
        """
        current_correct_projection = input_data
        for step in range(self._optim_steps):
            reconstruct_output = self.Reconstruct.execute(current_correct_projection)
            
            # Optimization step
            sim_data = self._optim_step(input_data, reconstruct_output)

            current_correct_projection = self.CorrectProjection.execute(sim_data)
            

    def _optim_step(self, I_meas, reconstruct_output):
        for step in range(self._optim_steps):
            I_sim = self.SpectralProjection.execute(reconstruct_output)

            loss = self._loss_fn(I_meas, I_sim, self._I_0)

            # Step
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        return I_sim
