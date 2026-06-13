"""
plot_3d_interactive.py -- interactive plotly viewer for the 3-D results. Produces one
self-contained HTML: 3-D original/corrected volumes, axial-slice slider, central
orthogonal cuts, profiles (centre line + PMMA radial cupping), attenuation histograms,
loss curve, and a metrics table. Standalone (numpy + plotly only):

    python plot_3d_interactive.py --arrays _arrays_3d --out reconstruction_3d.html
"""

import argparse
import os

import numpy as np


# -- metrics helpers -----------------------------------------------------------

def _radial_profile(arr, mask, n_bins=30):
    """Mean of `arr` over `mask` voxels, binned by 3-D radius from the volume centre."""
    ii, jj, kk = np.indices(arr.shape)
    c = (np.array(arr.shape) - 1) / 2.0
    r = np.sqrt((ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2)
    rr, vv = r[mask], arr[mask]
    if rr.size == 0:
        return np.array([]), np.array([])
    bins = np.linspace(0, float(rr.max()), n_bins + 1)
    idx = np.clip(np.digitize(rr, bins) - 1, 0, n_bins - 1)
    prof = np.array([vv[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])
    return 0.5 * (bins[:-1] + bins[1:]), prof


def _cupping(arr, mask, inner=0.3, outer=0.3):
    _, prof = _radial_profile(arr, mask)
    prof = prof[~np.isnan(prof)]
    if prof.size == 0:
        return float("nan")
    ni = max(1, int(inner * prof.size)); no = max(1, int(outer * prof.size))
    rim = prof[-no:].mean()
    return float(100.0 * (rim - prof[:ni].mean()) / rim) if rim else float("nan")


def _metrics(orig, final, phantom):
    rows = {}
    for lab, name in [(0.0, "Air"), (1.0, "PMMA"), (2.0, "Al")]:
        m = phantom == lab
        if not m.any():
            continue
        entry = {}
        for arr, key in [(orig, "original"), (final, "corrected")]:
            v = arr[m]; mean = float(v.mean()); std = float(v.std())
            entry[key] = {"mean": mean, "std": std, "cov": (std / abs(mean) if mean else float("nan"))}
        rows[name] = entry
    pmma = phantom == 1.0
    cup = {"original": _cupping(orig, pmma), "corrected": _cupping(final, pmma)}
    return rows, cup


def _metrics_table_html(rows, cup):
    h = ["<h2>Metrics (using the ground-truth phantom labels)</h2>",
         "<table border='1' cellpadding='6' style='border-collapse:collapse;font-family:monospace'>",
         "<tr><th>material</th><th>recon</th><th>mean</th><th>std</th><th>CoV (std/mean)</th></tr>"]
    for name, d in rows.items():
        for key in ("original", "corrected"):
            s = d[key]
            hl = " style='background:#eef'" if key == "corrected" else ""
            h.append(f"<tr{hl}><td>{name}</td><td>{key}</td><td>{s['mean']:.5f}</td>"
                     f"<td>{s['std']:.5f}</td><td>{s['cov']:.3f}</td></tr>")
    h.append("</table>")
    h.append(f"<p><b>PMMA cupping %:</b> original {cup['original']:.1f}% &rarr; "
             f"corrected <b>{cup['corrected']:.1f}%</b> &nbsp;(0% = flat = no residual beam hardening; "
             "lower CoV = more uniform material)</p>")
    return "".join(h)


# -- figure builder ------------------------------------------------------------

def _square_axes(fig, n_panels):
    """
    Make heatmap panels render square. A blanket update_yaxes(scaleanchor='x') wrongly
    anchors every y-axis to panel 1's x-axis -> panels 2..N stretch; instead anchor each
    y-axis to its own x-axis and constrain to the cell domain (letterbox, not distort).
    """
    for i in range(1, n_panels + 1):
        sfx = "" if i == 1 else str(i)
        fig.update_layout(**{f"yaxis{sfx}": dict(scaleanchor=f"x{sfx}", scaleratio=1,
                                                 constrain="domain")})
        fig.update_layout(**{f"xaxis{sfx}": dict(constrain="domain")})


def build_html(original, corrected, phantom, history, out_path, vol_stride=2, n_slices=20, title=""):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    original = np.asarray(original, dtype=np.float32)
    corrected = np.asarray(corrected, dtype=np.float32)
    vmin = float(min(original.min(), corrected.min()))
    vmax = float(max(original.max(), corrected.max()))
    nx, ny, nz = original.shape

    # 1) 3-D volumes (downsampled for browser performance)
    def vol(v):
        ds = v[::vol_stride, ::vol_stride, ::vol_stride]
        X, Y, Z = np.mgrid[0:ds.shape[0], 0:ds.shape[1], 0:ds.shape[2]]
        return go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=ds.flatten(),
            isomin=vmin + 0.12 * (vmax - vmin), isomax=vmax,
            opacity=0.12, surface_count=14, colorscale="Gray",
            cmin=vmin, cmax=vmax, showscale=False,
        )
    fig3d = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                          subplot_titles=("Original (beam-hardened)", "Corrected"))
    fig3d.add_trace(vol(original), 1, 1)
    fig3d.add_trace(vol(corrected), 1, 2)
    fig3d.update_layout(height=620, margin=dict(l=0, r=0, t=40, b=0),
                        title="3-D reconstructions - drag to rotate, scroll to zoom",
                        scene=dict(aspectmode="data"), scene2=dict(aspectmode="data"))

    if phantom is not None and (phantom == 1.0).any():
        pm = original[phantom == 1.0]
        wlo, whi = float(pm.mean() - 3 * pm.std()), float(pm.mean() + 3 * pm.std())
    else:
        wlo, whi = float(np.percentile(original, 40)), float(np.percentile(original, 99))

    # 2) axial slice slider (original | corrected)
    idxs = np.unique(np.linspace(0, nz - 1, n_slices).astype(int))
    mid = len(idxs) // 2
    figS = make_subplots(rows=1, cols=2, subplot_titles=("Original", "Corrected"))
    for i, z in enumerate(idxs):
        figS.add_trace(go.Heatmap(z=original[:, :, z].T, zmin=wlo, zmax=whi, colorscale="Gray",
                                  visible=(i == mid), showscale=False), 1, 1)
        figS.add_trace(go.Heatmap(z=corrected[:, :, z].T, zmin=wlo, zmax=whi, colorscale="Gray",
                                  visible=(i == mid), showscale=(i == mid)), 1, 2)
    steps = []
    for i, z in enumerate(idxs):
        vis = [False] * (2 * len(idxs)); vis[2 * i] = True; vis[2 * i + 1] = True
        steps.append(dict(method="update", args=[{"visible": vis}], label=str(int(z))))
    figS.update_layout(height=470,
                       title="Axial slice explorer - drag the slider",
                       sliders=[dict(active=mid, steps=steps, currentvalue={"prefix": "axial z = "})])
    _square_axes(figS, 2)

    # 3) central orthogonal cuts (orig top / corrected bottom)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    figC = make_subplots(rows=2, cols=3, subplot_titles=(
        "axial (z)", "coronal (y)", "sagittal (x)", "axial (z)", "coronal (y)", "sagittal (x)"))
    for r_i, src in enumerate((original, corrected)):
        for c_i, sl in enumerate((src[:, :, cz].T, src[:, cy, :].T, src[cx, :, :].T)):
            figC.add_trace(go.Heatmap(z=sl, zmin=vmin, zmax=vmax, colorscale="Gray", showscale=False),
                           r_i + 1, c_i + 1)
    figC.update_layout(height=620, title="Central cuts - original (top) vs corrected (bottom)")
    _square_axes(figC, 6)

    # 4) profiles: central line + PMMA radial (cupping)
    figP = make_subplots(rows=1, cols=2, subplot_titles=(
        "Central-line profile", "PMMA radial profile (cupping)"))
    figP.add_trace(go.Scatter(y=original[cx, cy, :], name="original", line=dict(color="royalblue")), 1, 1)
    figP.add_trace(go.Scatter(y=corrected[cx, cy, :], name="corrected",
                              line=dict(color="orange", dash="dash")), 1, 1)
    if phantom is not None and (phantom == 1.0).any():
        rO, pO = _radial_profile(original, phantom == 1.0)
        rC, pC = _radial_profile(corrected, phantom == 1.0)
        figP.add_trace(go.Scatter(x=rO, y=pO, name="orig PMMA", line=dict(color="royalblue")), 1, 2)
        figP.add_trace(go.Scatter(x=rC, y=pC, name="corr PMMA",
                                  line=dict(color="orange", dash="dash")), 1, 2)
    figP.update_layout(height=400, title="Profiles (flat corrected PMMA radial = cupping removed)")

    # 5) attenuation histogram of OBJECT voxels (pre-binned -> small HTML)
    obj = (phantom > 0) if phantom is not None else np.ones_like(original, dtype=bool)
    lo, hi = vmin, vmax
    oh, edges = np.histogram(original[obj], bins=120, range=(lo, hi))
    ch, _ = np.histogram(corrected[obj], bins=120, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    figH = go.Figure()
    figH.add_trace(go.Bar(x=centers, y=oh, name="original", opacity=0.6, marker_color="royalblue"))
    figH.add_trace(go.Bar(x=centers, y=ch, name="corrected", opacity=0.6, marker_color="orange"))
    figH.update_layout(height=350, barmode="overlay", title="Attenuation histogram of object voxels "
                       "(sharper, more separated peaks = better)", yaxis_type="log",
                       xaxis_title="reconstructed attenuation", yaxis_title="voxel count")

    # 6) loss curve
    figL = go.Figure()
    if len(history):
        figL.add_trace(go.Scatter(y=list(history), mode="lines", name="loss"))
    figL.update_layout(height=320, title="Optimisation loss",
                       xaxis_title="step", yaxis_title="MSE loss",
                       yaxis=dict(type="log", exponentformat="power", dtick=1))

    table = _metrics_table_html(*_metrics(original, corrected, phantom)) if phantom is not None else ""

    parts = [
        fig3d.to_html(full_html=False, include_plotlyjs=True),
        figS.to_html(full_html=False, include_plotlyjs=False),
        figC.to_html(full_html=False, include_plotlyjs=False),
        figP.to_html(full_html=False, include_plotlyjs=False),
        figH.to_html(full_html=False, include_plotlyjs=False),
        figL.to_html(full_html=False, include_plotlyjs=False),
    ]
    html = (
        "<html><head><meta charset='utf-8'>"
        "<title>3-D Beam-Hardening Correction - interactive</title></head>"
        "<body style='font-family:sans-serif;max-width:1300px;margin:24px auto;padding:0 12px'>"
        "<h1>3-D Beam-Hardening Correction - interactive results</h1>"
        + (f"<h2 style='color:#555;margin-top:0'>{title}</h2>" if title else "")
        + "Residual correction. Drag the 3-D views to rotate; "
        "use the slider to scrub slices.</p>"
        + table + "".join(parts) + "</body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", out_path, f"({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arrays", default="_arrays_3d", help="dir with original/corrected/phantom/history .npy")
    ap.add_argument("--out", default="reconstruction_3d.html")
    a = ap.parse_args()
    original = np.load(os.path.join(a.arrays, "original.npy"))
    corrected = np.load(os.path.join(a.arrays, "corrected.npy"))
    pj = os.path.join(a.arrays, "phantom.npy")
    hj = os.path.join(a.arrays, "history.npy")
    phantom = np.load(pj) if os.path.exists(pj) else None
    history = np.load(hj) if os.path.exists(hj) else []
    build_html(original, corrected, phantom, history, a.out)
    print("HTML_OK")


if __name__ == "__main__":
    main()
