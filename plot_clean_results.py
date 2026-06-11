import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# plotting.py imports only matplotlib + numpy, so this does NOT pull in torch/astra
from workflows.beam_hardening_correction_2d.plotting import (
    plot_comparison_2d,
    plot_cupping_validation_2d,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays", default=os.path.join(_HERE, "_clean_arrays"),
                    help="directory holding original/corrected/phantom/history .npy")
    ap.add_argument("--out", default=_HERE, help="directory to write the PNGs into")
    ap.add_argument("--suffix", default="", help="suffix appended to output PNG names")
    ap.add_argument("--comparison", default=None, help="explicit output path for the comparison figure")
    ap.add_argument("--cupping", default=None, help="explicit output path for the cupping figure")
    ap.add_argument("--history", default=None, help="explicit output path for the loss-history figure")
    args = ap.parse_args()
    sfx = args.suffix
    comp_path = args.comparison or os.path.join(args.out, f"comparison_2d_clean{sfx}.png")
    cupp_path = args.cupping or os.path.join(args.out, f"cupping_validation_2d_clean{sfx}.png")
    hist_path = args.history or os.path.join(args.out, f"optimization_history_2d_clean{sfx}.png")

    original = np.load(os.path.join(args.arrays, "original.npy"))
    corrected = np.load(os.path.join(args.arrays, "corrected.npy"))
    phantom = np.load(os.path.join(args.arrays, "phantom.npy"))
    history = np.load(os.path.join(args.arrays, "history.npy"))

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("Optimisation step")
    plt.ylabel("Loss")
    plt.title("Optimisation Loss History (2-D clean validation)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # Side-by-side comparison
    plot_comparison_2d(
        original, corrected, phantom=phantom,
        title=f"Beam-Hardening Correction -- 2-D Barba Phantom{(' ' + sfx) if sfx else ''}",
        save_path=comp_path,
    )

    # Cupping diagnostics (PMMA radial profile + per-material uniformity)
    plot_cupping_validation_2d(
        original, corrected, phantom=phantom,
        title="Cupping Validation -- PMMA radial profile + material uniformity",
        save_path=cupp_path,
    )

    print("PLOT_OK")


if __name__ == "__main__":
    main()
