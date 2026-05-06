import torch
from tqdm import tqdm

from ct_autodiff.engine.workflow import Workflow

from .blocks import Reconstruct, CorrectProjection, SpectralProjection, ProjectionData
from .objective_func import PhiLoss


class BeamHardeningCorrectionWorkflow(Workflow):
    def __init__(
            self,
            optim_steps: int = 500,
            lr: float = 0.01,
            n_angles=360,
            number_of_materials=2,
            energy_bins=358,
            gamma=1.0,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()
        self.add_block(ProjectionData())
        self.add_block(Reconstruct(n_angles=n_angles, device=device))
        self.add_block(CorrectProjection(device=device))
        self.add_block(SpectralProjection(
            n_angles=n_angles,
            number_of_materials=number_of_materials,
            energy_bins=energy_bins,
            gamma=gamma,
            device=device)
        )

        self._optim_steps = optim_steps
        self._loss_fn = PhiLoss()

        self._lr = lr
        self._optim = torch.optim.Adam([p for _, p in self.parameters()], lr=lr)

        self._device = device
        self.to(self._device)

    def run(self):
        """
        Executes the workflow by sequentially running each block on the output of the previous block.
        
        :param input_data: The initial input data for the workflow, which will be passed to the first block.
        :type input_data: Any
        :return: The output of the final block in the workflow after processing the input data through
                 all blocks.
        :rtype: Any
        """
        input_data = self.ProjectionData.execute()
        measured_projection = torch.from_numpy(input_data).float().to(self._device)
        original_reconstruction = self.Reconstruct.execute(input_data)

        # Convert original_reconstruction to torch tensor for optimization
        original_reconstruction_tensor = torch.from_numpy(original_reconstruction).float().to(self._device)
        
        # Optimization loop
        sim_data, history = self._optim_loop(measured_projection, original_reconstruction_tensor)

        # Pass the simulated data through the CorrectProjection block to get it ready for reconstruction
        correct_projection = self.CorrectProjection.execute(sim_data)

        final_reconstruction = self.Reconstruct.execute(correct_projection)
        return original_reconstruction, final_reconstruction, history
            
    def _optim_loop(self, input_data, reconstructed_data):
        history = []
        A_meas = input_data # (n_angles, n_pixels, n_pixels)

        # Warmup forward pass: triggers Otsu initialization of t so it is added
        # to _params before the optimizer is (re)built.
        with torch.no_grad():
            self.SpectralProjection.execute(reconstructed_data)
        self._optim = torch.optim.Adam([p for _, p in self.parameters()], lr=self._lr)

        for _ in tqdm(range(self._optim_steps), desc="Optimizing Spectral Projection"):
            A_sim = self.SpectralProjection.execute(reconstructed_data) # (n_pixels, n_angles, n_pixels)
            loss = self._loss_fn(A_sim, A_meas)

            # Step
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()            

            history.append(loss.item())
        return A_sim, history
