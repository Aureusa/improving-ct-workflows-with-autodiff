import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
from workflows.beam_hardening_correction.workflow import BeamHardeningCorrectionWorkflow
from workflows.beam_hardening_correction.plotting import plot_reconstruction

if __name__ == "__main__":
    workflow = BeamHardeningCorrectionWorkflow(
        optim_steps=1000,
        energy_bins=3
    )

    original_reconstruction, final_reconstruction, history = workflow.run()

    import matplotlib.pyplot as plt

    plt.plot(history)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Optimization Loss History")
    plt.yscale("log")
    plt.savefig(os.path.join(_HERE, "optimization_history.png"))

    plot_reconstruction(
        original_reconstruction,
        title="Original Reconstruction",
        save_path=os.path.join(_HERE, "original_reconstruction.png")
    )

    plot_reconstruction(
        final_reconstruction,
        title="Final Reconstruction",
        save_path=os.path.join(_HERE, "final_reconstruction.png")
    )
