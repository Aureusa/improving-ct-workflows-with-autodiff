"""
main_2d_clean.py
----------------
Clean-validation / honest-test harness for the 2-D beam-hardening correction.

Defaults reproduce the validated "fix" config (README §8.6): strong hardening
(dk=2, no filtration), noise off, I/mu frozen at ground truth, transmission-weighted
mu_eff. CLI flags expose the other regimes — in particular the recovery test
(--no-freeze --perturb 0.3), which lets I/mu be *learned* from a *perturbed*
(non-truth) start, so "loss goes down" actually tests whether an unknown spectrum
can be recovered (README §6 caveat).

Plotting runs in a subprocess (plot_clean_results.py) because matplotlib's native
rendering crashes alongside torch+astra+MKL on this conda/Windows box; if that env's
matplotlib is broken too, re-plot with another interpreter:
    python plot_clean_results.py --arrays _clean_arrays<suffix> --suffix <suffix>

Examples
--------
    python main_2d_clean.py                                # the fix (frozen, transmission)
    python main_2d_clean.py --mu-eff-mode fluence          # the broken baseline
    python main_2d_clean.py --no-freeze --perturb 0.3 --suffix _recovery   # honest recovery test
"""

import argparse
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

_PKG = os.path.join(_HERE, "workflows", "beam_hardening_correction_2d")


def _trans_mu_eff(I, mu, L_rep=(2.5, 0.1)):
    """Transmission-weighted mu_eff (M,) for a fixed representative path [cm] (numpy)."""
    L = np.asarray(L_rep, dtype=np.float64)[:, None]      # (M,1)
    w = I * np.exp(-(mu * L).sum(axis=0))                 # (E,)
    return (mu * w).sum(axis=1) / w.sum()                 # (M,)


def _print_report(metrics, header, mu_eff_gt=None, mu_eff_learned=None):
    print("\n" + "=" * 70)
    print(" CLEAN-VALIDATION REPORT  |  " + header)
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
    if mu_eff_gt is not None and mu_eff_learned is not None:
        rel = 100.0 * np.abs(mu_eff_learned - mu_eff_gt) / np.abs(mu_eff_gt)
        print("-" * 70)
        print("transmission mu_eff [cm^-1]  (representative path PMMA=2.5, Al=0.1 cm):")
        print(f"   ground truth : PMMA {mu_eff_gt[0]:8.3f}   Al {mu_eff_gt[1]:8.3f}")
        print(f"   learned      : PMMA {mu_eff_learned[0]:8.3f}   Al {mu_eff_learned[1]:8.3f}")
        print(f"   |rel error|  : PMMA {rel[0]:7.1f}%   Al {rel[1]:7.1f}%")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Clean-validation / honest-test harness (2-D BHC).")
    ap.add_argument("--mu-eff-mode", choices=["fluence", "transmission", "lstsq"],
                    default="transmission",
                    help="effective-attenuation weighting. 'lstsq' = original autodiffCT "
                         "least-squares regression (most robust).")
    ap.add_argument("--correction-mode", choices=["replace", "residual"], default="replace",
                    help="'replace' = reconstruct synthetic mono sinogram; "
                         "'residual' = y_meas + (y_mono - y_poly), preserves real detail (autodiffCT).")
    ap.add_argument("--freeze", action=argparse.BooleanOptionalAction, default=True,
                    help="freeze I/mu at ground truth (learn only t). --no-freeze learns I/mu too.")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="perturb spectral init away from truth by this fraction (per-bin ±).")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--al-filter", type=float, default=0.0, help="added Al filtration [mm]")
    ap.add_argument("--dk", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--outer", type=int, default=1,
                    help="correction passes. Default 1: the outer loop is non-monotonic "
                         "with a learned spectrum (sweep in README §8.7) — 1 pass already "
                         "recovers mu_eff and most cupping; extra passes can destabilise it.")
    ap.add_argument("--suffix", type=str, default="", help="suffix for output PNG names")
    args = ap.parse_args()

    workflow = BeamHardeningCorrectionWorkflow2D(
        optim_steps=args.steps, lr=0.001, outer_iters=args.outer,
        dk=args.dk, add_gaussian_noise=0.0,
        freeze_spectral=args.freeze, mu_eff_mode=args.mu_eff_mode,
        correction_mode=args.correction_mode,
        al_filter_mm=args.al_filter,
        spectral_perturb=args.perturb, spectral_perturb_seed=args.perturb_seed,
    )

    original, final, history = workflow.run()
    phantom = render_phantom_2d(show=False)
    metrics = compute_validation_metrics(original, final, phantom)

    # mu_eff recovery: ground-truth (data-generating) spectrum vs the learned one
    gt_I  = np.load(os.path.join(_PKG, "fluence.npy")).astype(np.float64)
    gt_mu = np.load(os.path.join(_PKG, "mu_values.npy")).astype(np.float64)
    sp = workflow.SpectralProjection2D
    learned_I  = sp.I.detach().cpu().numpy().astype(np.float64)
    learned_mu = sp.mu.detach().cpu().numpy().astype(np.float64)
    mu_eff_gt      = _trans_mu_eff(gt_I, gt_mu)
    mu_eff_learned = _trans_mu_eff(learned_I, learned_mu)

    header = (f"mu_eff={args.mu_eff_mode} correction={args.correction_mode} "
              f"freeze={args.freeze} perturb={args.perturb} "
              f"dk={args.dk} al_filter={args.al_filter}mm steps={args.steps}x{args.outer}")
    _print_report(metrics, header, mu_eff_gt, mu_eff_learned)

    # Render in a separate (torch/astra-free) process
    arrays_dir = os.path.join(_HERE, f"_clean_arrays{args.suffix}")
    os.makedirs(arrays_dir, exist_ok=True)
    np.save(os.path.join(arrays_dir, "original.npy"), original)
    np.save(os.path.join(arrays_dir, "corrected.npy"), final)
    np.save(os.path.join(arrays_dir, "phantom.npy"), phantom)
    np.save(os.path.join(arrays_dir, "history.npy"), np.asarray(history, dtype=np.float32))

    rc = subprocess.run(
        [sys.executable, os.path.join(_HERE, "plot_clean_results.py"),
         "--arrays", arrays_dir, "--out", _HERE, "--suffix", args.suffix],
    ).returncode

    if rc == 0:
        shutil.rmtree(arrays_dir, ignore_errors=True)
        print(f"\nSaved: comparison_2d_clean{args.suffix}.png, "
              f"cupping_validation_2d_clean{args.suffix}.png, "
              f"optimization_history_2d_clean{args.suffix}.png")
    else:
        print(f"\n[warn] plotting subprocess exited {rc}; arrays kept for manual plotting:")
        print(f"       python plot_clean_results.py --arrays {arrays_dir} --suffix {args.suffix}")


if __name__ == "__main__":
    main()
