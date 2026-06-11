import os, sys, subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
os.makedirs(_RESULTS, exist_ok=True)
_NPZ = os.path.join(_RESULTS, "_AL_data.npz")
_BASE_PY = r"C:/Users/ivanb/anaconda3/python.exe"

# (material, xraylib compound, L_max [cm], kVp, dk, Al-filtration [mm], label)
# al_mm = 0 -> the bare (soft-tail-heavy) experiment spectrum; >0 -> a realistic filtered beam.
_AL_FILTER_MM = float(os.environ.get("AL_FILTER_MM", "2.5"))
_CASES = [("PMMA", "C5H8O2", 5.0, 120, 2, _AL_FILTER_MM, "PMMA  (120 kVp)"),
          ("Al",   "Al",     1.0, 120, 2, _AL_FILTER_MM, "Al  (120 kVp)")]


def compute():
    import spekpy as sp
    from workflows.beam_hardening_correction_2d.barba_2d_phantom import generate_linear_attenuation_params
    data = {}
    for tag, compound, Lmax, kvp, dk, al_mm, label in _CASES:
        r = sp.Spek(kvp=kvp, th=12, dk=dk, physics="spekcalc")
        if al_mm > 0:
            r.filter("Al", al_mm)
        label = label + (f", {al_mm:g} mm Al" if al_mm > 0 else ", unfiltered")
        I = np.array(r.get_spk(), dtype=np.float64)
        mu = np.array(generate_linear_attenuation_params(r, compound), dtype=np.float64)
        L = np.linspace(0, Lmax, 300)
        A_poly = -np.log((I[:, None] * np.exp(-mu[:, None] * L[None, :])).sum(0) / I.sum())
        mu_bar = (I * mu).sum() / I.sum()          # initial slope (eq. 4)
        data[f"{tag}_L"] = L
        data[f"{tag}_poly"] = A_poly
        data[f"{tag}_mono"] = mu_bar * L
        data[f"{tag}_label"] = label
        data[f"{tag}_droop"] = 100.0 * (mu_bar * L[-1] - A_poly[-1]) / (mu_bar * L[-1])
    np.savez(_NPZ, **data)
    print("saved", _NPZ)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(_NPZ, allow_pickle=True)
    tags = [c[0] for c in _CASES]
    fig, axes = plt.subplots(1, len(tags), figsize=(5.4 * len(tags), 4.4))
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        L = d[f"{tag}_L"]; poly = d[f"{tag}_poly"]; mono = d[f"{tag}_mono"]
        label = str(d[f"{tag}_label"])
        ax.plot(L, mono, "k--", lw=2.0, label=r"monochromatic")
        ax.plot(L, poly, color="C0", lw=2.4,
                label=r"polychromatic")
        ax.set_title(label)
        ax.set_xlabel("path length  L  [cm]")
        ax.set_ylabel(r"attenuation  $A=-\ln(I/I_0)$")
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, L[-1]); ax.set_ylim(bottom=0)
    fig.suptitle("Beam hardening: monochromatic (linear) vs polychromatic (concave) attenuation",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    sfx = f"_{_AL_FILTER_MM:g}mmAl" if _AL_FILTER_MM > 0 else "_unfiltered"
    out = os.path.join(_RESULTS, f"AL_mono_vs_poly{sfx}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("compute", "all"):
        compute()
    if mode == "render":
        render()
    if mode == "all":
        # auto-render with the base python (working matplotlib)
        py = _BASE_PY if os.path.exists(_BASE_PY) else sys.executable
        subprocess.run([py, os.path.join(_HERE, "plot_AL.py"), "render"])
