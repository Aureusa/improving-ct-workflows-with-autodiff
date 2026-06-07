"""
plot_clean_results.py
---------------------
Render the clean-validation figures from saved arrays, in a process that imports
ONLY numpy + matplotlib (NO torch / astra).

main_2d_clean.py runs this as a subprocess because matplotlib's native rendering
hard-crashes (silent, no Python traceback) when the torch + astra + MKL OpenMP
runtimes are loaded in the same process on this conda/Windows box —
KMP_DUPLICATE_LIB_OK does not prevent it. Plotting in a fresh, torch/astra-free
process side-steps the conflict. It also works standalone for re-plotting:

    python plot_clean_results.py --arrays _clean_arrays --out .

Reads : <arrays>/{original,corrected,phantom,history}.npy
Writes: <out>/{optimization_history_2d_clean,comparison_2d_clean,cupping_validation_2d_clean}.png
"""

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
    args = ap.parse_args()
    sfx = args.suffix

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
    plt.savefig(os.path.join(args.out, f"optimization_history_2d_clean{sfx}.png"))
    plt.close()

    # Side-by-side comparison
    plot_comparison_2d(
        original, corrected, phantom=phantom,
        title=f"Clean Validation — 2-D Barba Phantom{(' ' + sfx) if sfx else ''}",
        save_path=os.path.join(args.out, f"comparison_2d_clean{sfx}.png"),
    )

    # Cupping diagnostics (PMMA radial profile + per-material uniformity)
    plot_cupping_validation_2d(
        original, corrected, phantom=phantom,
        title="Cupping Validation — PMMA radial profile + material uniformity",
        save_path=os.path.join(args.out, f"cupping_validation_2d_clean{sfx}.png"),
    )

    print("PLOT_OK")


if __name__ == "__main__":
    main()
