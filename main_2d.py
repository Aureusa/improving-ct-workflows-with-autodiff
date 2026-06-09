"""
main_2d.py — primary 2-D results runner
=======================================
Runs the beam-hardening correction with the most faithful + robust configuration
we have (closely matching the original authors' autodiffCT method, see README §8.8):

  mu_eff_mode     = "lstsq"     least-squares effective attenuation — regress the
                                polychromatic simulation onto the material path
                                sinograms. No spectrum average, so it is immune to
                                the soft-tail mu_eff inflation (README §8.5).
  correction_mode = "residual" y_corrected = y_meas + (y_mono - y_poly): correct the
                                *measured* sinogram by the modelled BH difference,
                                preserving real measurement detail.
  freeze_spectral = False       learn spectrum + attenuations + thresholds (honest —
                                no ground-truth spectrum is assumed; lstsq does not
                                rely on the learned spectrum being exact).
  dk = 2, add_gaussian_noise = 0.

Local result (this config): PMMA cupping 25.3% -> ~3%, PMMA CoV 0.16 -> 0.07.

Produces (the canonical result figures, overwritten on each run):
  comparison_2d.png            phantom | original | corrected | centre-line profiles
  cupping_2d.png               PMMA radial profile + per-material mean/std
  optimization_history_2d.png  loss curve
and prints a quantitative report.

Plotting runs in a SUBPROCESS (plot_clean_results.py) because matplotlib's native
rendering crashes alongside torch + astra + MKL in this conda env (README §8.5). If
that subprocess fails, re-plot the saved arrays with another interpreter.

Run
---
    python main_2d.py            # requires ASTRA + CUDA (FP/FBP)
"""

import os
import shutil
import subprocess
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)

import numpy as np

from workflows.beam_hardening_correction_2d.workflow import BeamHardeningCorrectionWorkflow2D
from workflows.beam_hardening_correction_2d.plotting import compute_validation_metrics
from workflows.beam_hardening_correction_2d.barba_2d_phantom import render_phantom_2d


def _print_report(metrics, header):
    print("\n" + "=" * 70)
    print(" 2-D BEAM-HARDENING CORRECTION — RESULTS  |  " + header)
    print("=" * 70)
    print(f"{'Material':<8} {'Recon':<10} {'mean':>13} {'std':>13} {'CoV':>9}")
    print("-" * 70)
    for name, entry in metrics["materials"].items():
        for key in ("original", "corrected"):
            s = entry[key]
            print(f"{name:<8} {key:<10} {s['mean']:>13.5f} {s['std']:>13.5f} {s['cov']:>9.3f}")
    print("-" * 70)
    cup = metrics["pmma_cupping_pct"]
    print(f"PMMA cupping %   original : {cup['original']:>7.2f}%")
    print(f"PMMA cupping %   corrected: {cup['corrected']:>7.2f}%")
    print("  (closer to 0% = flatter PMMA; lower CoV = more uniform material)")
    print("=" * 70)


if __name__ == "__main__":
    workflow = BeamHardeningCorrectionWorkflow2D(
        optim_steps=500,
        lr=0.001,
        dk=2.0,                      # strong, unambiguous beam hardening
        add_gaussian_noise=0.0,      # clean demonstration (set >0 for noisy data)
        mu_eff_mode="lstsq",         # least-squares effective attenuation (autodiffCT)
        correction_mode="residual",  # correct the measured sinogram (autodiffCT)
        freeze_spectral=False,       # learn the spectrum — honest, not ground-truth-pinned
    )

    original, final, history = workflow.run()
    phantom = render_phantom_2d(show=False)

    metrics = compute_validation_metrics(original, final, phantom)
    _print_report(metrics, "mu_eff=lstsq correction=residual freeze=False dk=2 noise=0 outer=1")

    # ── Render figures in a separate (torch/astra-free) process ───────────────
    arrays_dir = os.path.join(_HERE, "_arrays_main")
    os.makedirs(arrays_dir, exist_ok=True)
    np.save(os.path.join(arrays_dir, "original.npy"), original)
    np.save(os.path.join(arrays_dir, "corrected.npy"), final)
    np.save(os.path.join(arrays_dir, "phantom.npy"), phantom)
    np.save(os.path.join(arrays_dir, "history.npy"), np.asarray(history, dtype=np.float32))

    rc = subprocess.run(
        [sys.executable, os.path.join(_HERE, "plot_clean_results.py"),
         "--arrays", arrays_dir, "--out", _HERE,
         "--comparison", os.path.join(_HERE, "comparison_2d.png"),
         "--cupping", os.path.join(_HERE, "cupping_2d.png"),
         "--history", os.path.join(_HERE, "optimization_history_2d.png")],
    ).returncode

    if rc == 0:
        shutil.rmtree(arrays_dir, ignore_errors=True)
        print("\nSaved:")
        print("  comparison_2d.png")
        print("  cupping_2d.png")
        print("  optimization_history_2d.png")
    else:
        print(f"\n[warn] plotting subprocess exited {rc}; arrays kept for manual plotting:")
        print(f"       python plot_clean_results.py --arrays {arrays_dir} "
              f"--comparison comparison_2d.png --cupping cupping_2d.png "
              f"--history optimization_history_2d.png")
