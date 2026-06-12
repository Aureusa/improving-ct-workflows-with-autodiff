"""
main.py -- 3-D entry point
=========================
Runs the 3-D beam-hardening correction (least-squares mu_eff + residual correction,
normalized soft-segmentation + per-parameter LR) and builds an
INTERACTIVE plotly viewer you open in a browser:

    reconstruction_3d.html

The viewer lets you rotate/zoom the original and corrected volumes, scrub axial slices,
inspect the central orthogonal cuts, and read profiles / histograms / metrics. Plotly is
pure-Python (no native rendering), so this runs end-to-end without the matplotlib+torch+
astra crash. The reconstruction arrays are also saved to `_arrays_3d/` so the viewer can
be rebuilt without re-running the (heavy) pipeline:

    python plot_3d_interactive.py --arrays _arrays_3d --out reconstruction_3d.html

Run
---
    python main.py            # requires ASTRA + CUDA (FP3D / SIRT3D)
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)

import numpy as np

from workflows.beam_hardening_correction.workflow import BeamHardeningCorrectionWorkflow
from workflows.beam_hardening_correction.barba_3D_phantom_1 import render_phantom
from plot_3d_interactive import build_html

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="3-D beam-hardening correction + interactive viewer.")
    ap.add_argument("--correction-mode", choices=["replace", "residual"], default="residual",
                    help="'residual' (default) = y_meas + (y_mono - y_poly); "
                         "'replace' = synthetic mono sinogram.")
    ap.add_argument("--mu-eff-mode", choices=["fluence", "lstsq"], default="lstsq")
    ap.add_argument("--steps", type=int, default=300, help="3-D is heavy (128^3 x ~35 bins)")
    ap.add_argument("--dk", type=float, default=5.0)
    ap.add_argument("--perturb", type=float, default=0.2,
                    help="perturb the spectral init (I and mu) away from ground truth by this "
                         "fraction (per-bin +/-). The 3-D spectrum is always learned, so a nonzero "
                         "value makes recovery an honest test instead of starting at the answer "
                         "(inverse crime). The lstsq+residual correction is unaffected (rides on "
                         "y_poly~=y_meas, not on the exact spectrum).")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.1,
                    help="fractional Gaussian noise  on the simulated sinogram (0 = clean). "
                         "Constant intensity noise -> hits thick rays hardest after -log.")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian [vox] to denoise the recon BEFORE segmentation "
                         "(0 = off; ~1-1.5 stabilises the masks under --noise).")
    args = ap.parse_args()
    print(f"config: correction={args.correction_mode}  "
          f"mu_eff={args.mu_eff_mode}  dk={args.dk}  steps={args.steps}  "
          f"perturb={args.perturb} (seed {args.perturb_seed})  "
          f"noise={args.noise} (seed {args.noise_seed})  smooth={args.smooth}")

    workflow = BeamHardeningCorrectionWorkflow(
        optim_steps=args.steps,
        dk=args.dk,
        mu_eff_mode=args.mu_eff_mode,
        correction_mode=args.correction_mode,
        spectral_perturb=args.perturb,
        spectral_perturb_seed=args.perturb_seed,
        add_gaussian_noise=args.noise,
        noise_seed=args.noise_seed,
        smooth_sigma=args.smooth,
    )

    original, final, history = workflow.run()
    phantom = render_phantom(show_3d=False, show_projection=False)  # ground-truth labels for metrics

    # Save arrays so the interactive viewer can be rebuilt without re-running the pipeline
    arrays_dir = os.path.join(_HERE, "_arrays_3d")
    os.makedirs(arrays_dir, exist_ok=True)
    np.save(os.path.join(arrays_dir, "original.npy"), original)
    np.save(os.path.join(arrays_dir, "corrected.npy"), final)
    np.save(os.path.join(arrays_dir, "phantom.npy"), phantom)
    np.save(os.path.join(arrays_dir, "history.npy"), np.asarray(history, dtype=np.float32))

    # Build the interactive HTML (plotly is pure-Python -> safe in this process)
    out_html = os.path.join(_HERE, "reconstruction_3d.html")
    build_html(original, final, phantom, np.asarray(history), out_html)

    print("\nDone. Open in a browser:")
    print(f"  {out_html}")
    print(f"(arrays kept in {arrays_dir} -- rebuild with: python plot_3d_interactive.py --arrays _arrays_3d)")
