import sys

sys.path.append("/home/s4861264/CIT_project/")
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
    plt.savefig("/home/s4861264/CIT_project/optimization_history.png")

    plot_reconstruction(
        original_reconstruction,
        title="Original Reconstruction",
        save_path="/home/s4861264/CIT_project/original_reconstruction.png"
    )

    plot_reconstruction(
        final_reconstruction,
        title="Final Reconstruction",
        save_path="/home/s4861264/CIT_project/final_reconstruction.png"
    )
