import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)

import matplotlib.pyplot as plt

from workflows.beam_hardening_correction_2d.workflow import BeamHardeningCorrectionWorkflow2D
from workflows.beam_hardening_correction_2d.plotting import (
    plot_reconstruction_2d,
    plot_comparison_2d,
    plot_segmentation_comparison_2d,
)
from workflows.beam_hardening_correction_2d.barba_2d_phantom import render_phantom_2d

if __name__ == "__main__":
    workflow = BeamHardeningCorrectionWorkflow2D(
        optim_steps=50000,
        lr=0.001,
        outer_iters=3,   # iterative correction; total steps = optim_steps * outer_iters
    )

    original_reconstruction, final_reconstruction, history = workflow.run()

    # Ground-truth phantom (same default seed/params as ProjectionData2D)
    phantom = render_phantom_2d(show=False)

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("Optimisation step")
    plt.ylabel("Loss")
    plt.title("Optimisation Loss History (2-D)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(_HERE, "optimization_history_2d.png"))
    plt.close()

    # Individual reconstruction plots
    plot_reconstruction_2d(
        original_reconstruction,
        title="Original Reconstruction (beam-hardened)",
        save_path=os.path.join(_HERE, "original_reconstruction_2d.png"),
    )

    plot_reconstruction_2d(
        final_reconstruction,
        title="Corrected Reconstruction",
        save_path=os.path.join(_HERE, "final_reconstruction_2d.png"),
    )

    # Side-by-side comparison with ground-truth phantom
    plot_comparison_2d(
        original_reconstruction,
        final_reconstruction,
        phantom=phantom,
        title="Beam-Hardening Correction — 2-D Barba Phantom",
        save_path=os.path.join(_HERE, "comparison_2d.png"),
    )

    # Segmentation comparison using learned ISP thresholds
    learned_thresholds = workflow.SpectralProjection2D.t.detach().cpu().numpy()
    plot_segmentation_comparison_2d(
        original_reconstruction,
        final_reconstruction,
        thresholds=learned_thresholds,
        phantom=phantom,
        title="Segmentation Comparison — Learned ISP Thresholds",
        save_path=os.path.join(_HERE, "comparison_threshold_2d.png"),
    )

    print("\nSaved:")
    print("  optimization_history_2d.png")
    print("  original_reconstruction_2d.png")
    print("  final_reconstruction_2d.png")
    print("  comparison_2d.png")
    print("  comparison_threshold_2d.png")
