import sys

sys.path.append("/home/s4861264/CIT_project/")
from workflows.beam_hardening_correction.workflow import BeamHardeningCorrectionWorkflow

if __name__ == "__main__":
    workflow = BeamHardeningCorrectionWorkflow(
        optim_steps=100
    )

    final_reconstruction, history = workflow.run()

    import matplotlib.pyplot as plt

    plt.plot(history)
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Optimization Loss History")
    plt.savefig("/home/s4861264/CIT_project/optimization_history.png")
