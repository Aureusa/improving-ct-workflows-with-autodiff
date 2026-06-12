import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
import numpy as np
import spekpy as sp


_RESULTS = os.path.join(_HERE, "results")
os.makedirs(os.path.join(_RESULTS, "2d"), exist_ok=True)
os.makedirs(os.path.join(_RESULTS, "3d"), exist_ok=True)

# matplotlib hard-crashes (exit 127 / 0xC06D7FFF) in the Tomography conda env even in a
# fresh process, so render the 2-D PNGs with the base anaconda python (working matplotlib).
_PLOT_PY = r"C:/Users/ivanb/anaconda3/python.exe"
if not os.path.exists(_PLOT_PY):
    _PLOT_PY = sys.executable


def _save(mode, data):
    with open(os.path.join(_RESULTS, f"_exp_{mode}.json"), "w") as f:
        json.dump(data, f, indent=2)


def _render2d(orig, final, phantom, hist, png_path, title=""):
    """Render comparison + cupping PNGs in a torch/astra-free subprocess (matplotlib
    crashes in-process alongside torch+astra). png_path = the comparison figure path."""
    import tempfile, subprocess, shutil
    d = tempfile.mkdtemp()
    np.save(os.path.join(d, "original.npy"), orig)
    np.save(os.path.join(d, "corrected.npy"), final)
    np.save(os.path.join(d, "phantom.npy"), phantom)
    np.save(os.path.join(d, "history.npy"), np.asarray(hist, dtype=np.float32))
    base = png_path[:-4] if png_path.endswith(".png") else png_path
    cmd = [_PLOT_PY, os.path.join(_HERE, "plot_clean_results.py"),
           "--arrays", d, "--out", os.path.dirname(png_path) or ".",
           "--comparison", png_path,
           "--cupping", base + "_cupping.png",
           "--history", base + "_loss.png"]
    if title:
        cmd += ["--title", title]
    rc = subprocess.run(cmd).returncode
    shutil.rmtree(d, ignore_errors=True)
    if rc != 0:
        print(f"  [warn] plot subprocess exited {rc} for {png_path}")


def _droop(kvp, dk, mat="C5H8O2", L_cm=5.0):
    """A(L) droop % through L_cm of mat for a kvp/dk spectrum (input hardening severity)."""
    from workflows.beam_hardening_correction_2d.barba_2d_phantom import generate_linear_attenuation_params
    r = sp.Spek(kvp=kvp, th=12, dk=dk, physics="spekcalc")
    f = np.array(r.get_spk()); mu = np.array(generate_linear_attenuation_params(r, mat))
    L = np.linspace(0, L_cm, 60)
    A = -np.log((f[:, None] * np.exp(-mu[:, None] * L[None, :])).sum(0) / f.sum())
    slope0 = A[1] / L[1]
    return int(len(f)), float((slope0 * L[-1] - A[-1]) / (slope0 * L[-1]) * 100)


# ----------------------------------- 2-D --------------------------------------
def _metrics2d(orig, final, phantom):
    from workflows.beam_hardening_correction_2d.plotting import compute_validation_metrics
    m = compute_validation_metrics(orig, final, phantom)
    M, c = m["materials"], m["pmma_cupping_pct"]
    return dict(cup_orig=c["original"], cup_corr=c["corrected"],
                pmma_cov_orig=M["PMMA"]["original"]["cov"], pmma_cov_corr=M["PMMA"]["corrected"]["cov"],
                al_cov_corr=M["Al"]["corrected"]["cov"])


def _run2d(dk, noise=0.0, seed=0, smooth=0.0, fig_path=None):
    from workflows.beam_hardening_correction_2d.workflow import BeamHardeningCorrectionWorkflow2D
    from workflows.beam_hardening_correction_2d.barba_2d_phantom import render_phantom_2d
    wf = BeamHardeningCorrectionWorkflow2D(
        optim_steps=300, lr=0.001, dk=dk, add_gaussian_noise=noise, noise_seed=seed,
        freeze_spectral=False, mu_eff_mode="lstsq", correction_mode="residual",
        spectral_perturb=0.0, smooth_sigma=smooth)
    orig, final, hist = wf.run()
    phantom = render_phantom_2d(show=False)
    if fig_path:
        title = f"dk={dk}  noise={noise:.0%}  smooth={smooth}"
        _render2d(orig, final, phantom, hist, fig_path, title=title)
    return _metrics2d(orig, final, phantom)


def sweep_2d():
    out = {"dk": [], "noise": []}
    # Exp A: hardening vs dk (no noise) -- 120 kVp
    for dk in (2, 5, 10, 20):
        r = dict(dk=dk, droop_pct=_droop(120, dk)[1], **_run2d(dk))
        out["dk"].append(r); _save("2d", out)
        print(f"DONE 2d-dk | dk={dk:<3} droop {r['droop_pct']:5.1f}% | cup {r['cup_orig']:6.2f} -> {r['cup_corr']:7.2f}")
    # Exp B: noise effect at dk=2 (seed-averaged); smooth=0, plus smooth=1.0 at the top level
    for noise, smooth in [(0.0, 0.0), (0.02, 0.0), (0.02, 1.0), (0.05, 0.0), (0.05, 1.0)]:
        rows = []
        for seed in (0, 1, 2):
            try:
                rows.append(_run2d(2, noise=noise, seed=seed, smooth=smooth))
            except Exception as e:
                rows.append(dict(error=repr(e)))
        ok = [r for r in rows if "error" not in r]
        agg = dict(noise=noise, smooth=smooth,
                   cup_corr_mean=float(np.mean([abs(r["cup_corr"]) for r in ok])),
                   cup_corr_std=float(np.std([abs(r["cup_corr"]) for r in ok])),
                   pmma_cov_corr_mean=float(np.mean([r["pmma_cov_corr"] for r in ok])),
                   cup_orig_mean=float(np.mean([r["cup_orig"] for r in ok])))
        out["noise"].append(agg); _save("2d", out)
        print(f"DONE 2d-noise | noise={noise} smooth={smooth} | |cup_corr| {agg['cup_corr_mean']:.2f}+/-{agg['cup_corr_std']:.2f}  pmmaCoV {agg['pmma_cov_corr_mean']:.3f}")
    return out


# ----------------------------------- 3-D --------------------------------------
def _run3d(dk, noise=0.0, seed=0, smooth=0.0, steps=200, html_path=None):
    from workflows.beam_hardening_correction.workflow import BeamHardeningCorrectionWorkflow
    from workflows.beam_hardening_correction.barba_3D_phantom_1 import render_phantom
    from plot_3d_interactive import _metrics, build_html
    wf = BeamHardeningCorrectionWorkflow(
        optim_steps=steps, dk=dk, mu_eff_mode="lstsq", correction_mode="residual",
        spectral_perturb=0.0, add_gaussian_noise=noise, noise_seed=seed, smooth_sigma=smooth)
    orig, final, hist = wf.run()
    phantom = render_phantom(show_3d=False, show_projection=False)
    if html_path:
        title = f"dk={dk}  noise={noise:.0%}  smooth={smooth}"
        build_html(orig, final, phantom, np.asarray(hist), html_path, title=title)
    rows, cup = _metrics(orig, final, phantom)
    return dict(cup_orig=cup["original"], cup_corr=cup["corrected"],
                pmma_cov_orig=rows["PMMA"]["original"]["cov"], pmma_cov_corr=rows["PMMA"]["corrected"]["cov"],
                al_cov_corr=rows["Al"]["corrected"]["cov"])


def sweep_3d():
    out = {"dk": [], "noise": []}
    for dk in (5, 20):                       # high hardening vs low hardening (no noise)
        r = dict(dk=dk, droop_pct=_droop(120, dk)[1], **_run3d(dk))
        out["dk"].append(r); _save("3d", out)
        print(f"DONE 3d-dk | dk={dk:<3} droop {r['droop_pct']:5.1f}% | cup {r['cup_orig']:6.2f} -> {r['cup_corr']:7.2f}")
    for noise, smooth in [(0.02, 0.0), (0.05, 0.0), (0.05, 1.0)]:   # noise effect at dk=5
        r = dict(noise=noise, smooth=smooth, **_run3d(5, noise=noise, smooth=smooth))
        out["noise"].append(r); _save("3d", out)
        print(f"DONE 3d-noise | noise={noise} smooth={smooth} | cup {r['cup_orig']:6.2f} -> {r['cup_corr']:7.2f}  pmmaCoV {r['pmma_cov_corr']:.3f}")
    return out


# ----------------------------- figure generation ------------------------------
# One representative figure per config (seed 0). 2-D -> PNG, 3-D -> interactive HTML.
_FIG_2D = [("dk2", dict(dk=2)), ("dk5", dict(dk=5)), ("dk10", dict(dk=10)), ("dk20", dict(dk=20)),
           ("dk2_noise0.02", dict(dk=2, noise=0.02)),
           ("dk2_noise0.02_smooth1.0", dict(dk=2, noise=0.02, smooth=1.0)),
           ("dk2_noise0.05", dict(dk=2, noise=0.05)),
           ("dk2_noise0.05_smooth1.0", dict(dk=2, noise=0.05, smooth=1.0))]
_FIG_3D = [("dk5", dict(dk=5)), ("dk20", dict(dk=20)),
           ("dk5_noise0.02", dict(dk=5, noise=0.02)), ("dk5_noise0.05", dict(dk=5, noise=0.05)),
           ("dk5_noise0.05_smooth1.0", dict(dk=5, noise=0.05, smooth=1.0))]


def figures_2d():
    for name, c in _FIG_2D:
        m = _run2d(fig_path=os.path.join(_RESULTS, "2d", name + ".png"), **c)
        print(f"saved results/2d/{name}.png   | cup {m['cup_orig']:6.1f} -> {m['cup_corr']:6.1f}")


def figures_3d():
    for name, c in _FIG_3D:
        m = _run3d(html_path=os.path.join(_RESULTS, "3d", name + ".html"), **c)
        print(f"saved results/3d/{name}.html  | cup {m['cup_orig']:6.1f} -> {m['cup_corr']:6.1f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "2d"
    {"2d": sweep_2d, "3d": sweep_3d,
     "fig2d": figures_2d, "fig3d": figures_3d}[mode]()
    print(f"\nDone: {mode} -> results/")
