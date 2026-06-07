"""
main.py — 3-D entry point
=========================
Runs the 3-D beam-hardening correction (least-squares mu_eff + residual correction,
normalized soft-segmentation + per-parameter LR — see README §8.8/§8.9) and builds an
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
    workflow = BeamHardeningCorrectionWorkflow(
        optim_steps=300,             # 3-D is heavy (128^3 × ~35 energy bins)
        dk=5,                        # ~35 energy bins → real beam hardening (dk=50 gave ~0%) — README §8.9
        mu_eff_mode="lstsq",         # least-squares effective attenuation (autodiffCT) — §8.8
        correction_mode="residual",  # correct the measured sinogram: y_meas + (y_mono - y_poly) — §8.8
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

    # Build the interactive HTML (plotly is pure-Python → safe in this process)
    out_html = os.path.join(_HERE, "reconstruction_3d.html")
    build_html(original, final, phantom, np.asarray(history), out_html)

    print("\nDone. Open in a browser:")
    print(f"  {out_html}")
    print(f"(arrays kept in {arrays_dir} — rebuild with: python plot_3d_interactive.py --arrays _arrays_3d)")
