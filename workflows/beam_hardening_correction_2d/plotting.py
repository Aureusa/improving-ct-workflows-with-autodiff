import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction_2d(
    reconstruction: np.ndarray,
    scatter_stride: int = 1,       # kept for API compatibility with 3-D version
    marker_size: float = 1.5,
    marker_opacity: float = 0.3,
    title: str = "Reconstruction",
    save_path: str = None,
) -> None:
    """Show a 2-D reconstruction: imshow + horizontal centre-line profile (reveals cupping)."""
    mid = reconstruction.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    axes[0].imshow(reconstruction.T, origin="lower", cmap="gray")
    axes[0].set_title("Reconstruction")
    axes[0].axis("off")

    axes[1].plot(reconstruction[mid], lw=1.2)
    axes[1].set_title("Centre-line profile  (cupping check)")
    axes[1].set_xlabel("Pixel")
    axes[1].set_ylabel("Reconstructed attenuation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    plt.close()


def plot_comparison_2d(
    original: np.ndarray,
    corrected: np.ndarray,
    phantom: np.ndarray = None,
    title: str = "Beam-Hardening Correction",
    save_path: str = None,
) -> None:
    """
    Side-by-side original (beam-hardened) vs corrected reconstructions, optionally with
    the ground-truth phantom (adds a 4th panel + overlays the phantom centre-line).
    """
    from matplotlib.colors import ListedColormap

    mid  = original.shape[0] // 2
    vmin = min(original.min(), corrected.min())
    vmax = max(original.max(), corrected.max())

    ncols = 4 if phantom is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(title)

    col = 0  # running column index

    # -- Phantom image (optional) ----------------------------------------------
    if phantom is not None:
        cmap_gt = ListedColormap(["#1a1a2e", "#4e9af1", "#f4a261"])
        axes[col].imshow(
            phantom.T, origin="lower", cmap=cmap_gt,
            vmin=-0.5, vmax=2.5, interpolation="nearest",
        )
        axes[col].set_title("Ground-truth phantom")
        axes[col].axis("off")
        col += 1

    # -- Reconstruction images -------------------------------------------------
    axes[col].imshow(original.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[col].set_title("Original (beam-hardened)")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(corrected.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[col].set_title("Corrected")
    axes[col].axis("off")
    col += 1

    # -- Centre-line profiles --------------------------------------------------
    ax_att = axes[col]
    ax_att.plot(original[mid],  label="Original",  lw=1.2, color="tab:blue")
    ax_att.plot(corrected[mid], label="Corrected", lw=1.2, color="tab:orange", linestyle="--")
    ax_att.set_title("Centre-line profiles")
    ax_att.set_xlabel("Pixel")
    ax_att.set_ylabel("Attenuation")
    ax_att.grid(True, alpha=0.3)

    if phantom is not None:
        ax_mat = ax_att.twinx()
        ax_mat.step(
            np.arange(phantom.shape[1]), phantom[mid],
            where="mid", lw=1.0, color="tab:green",
            linestyle=":", label="Phantom (material)",
        )
        ax_mat.set_ylabel("Material label", color="tab:green")
        ax_mat.set_yticks([0, 1, 2])
        ax_mat.set_yticklabels(["Air", "PMMA", "Al"], color="tab:green")
        ax_mat.set_ylim(-0.3, 3.5)
        # Merge legends from both axes
        lines_att, labels_att = ax_att.get_legend_handles_labels()
        lines_mat, labels_mat = ax_mat.get_legend_handles_labels()
        ax_att.legend(lines_att + lines_mat, labels_att + labels_mat, fontsize=8)
    else:
        ax_att.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    plt.close()


def plot_segmentation_comparison_2d(
    original: np.ndarray,
    corrected: np.ndarray,
    thresholds: np.ndarray,
    phantom: np.ndarray = None,
    title: str = "Segmentation Comparison",
    save_path: str = None,
) -> None:
    """
    Hard-segment the original (beam-hardened) and corrected reconstructions with the
    learned ISP thresholds and plot them side-by-side so the BH effect is visible.
    thresholds (M,) are sorted before use; phantom (optional) is the ground-truth labels.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    thresholds = np.sort(thresholds)          # ensure ascending order
    n_labels   = len(thresholds) + 1          # e.g. 3 for 2 thresholds
    vmin, vmax = -0.5, n_labels - 0.5        # noqa: F841

    # Hard segmentation via digitize: result is 0 ... n_labels-1
    seg_orig = np.digitize(original,  thresholds).astype(int)
    seg_corr = np.digitize(corrected, thresholds).astype(int)

    # Shared colormap + legend
    colors         = ["#1a1a2e", "#4e9af1", "#f4a261", "#6fcf97"][:n_labels]
    mat_names      = ["Air", "PMMA", "Al", "Mat-3"][:n_labels]
    cmap_seg       = ListedColormap(colors)
    norm_seg       = BoundaryNorm(np.arange(-0.5, n_labels), ncolors=n_labels)
    legend_patches = [Patch(facecolor=c, label=l) for c, l in zip(colors, mat_names)]

    mid   = original.shape[0] // 2
    ncols = 4 if phantom is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(title)

    col = 0

    # -- Ground-truth phantom (optional) --------------------------------------
    if phantom is not None:
        axes[col].imshow(
            phantom.T, origin="lower", cmap=cmap_seg, norm=norm_seg,
            interpolation="nearest",
        )
        axes[col].set_title("Ground truth")
        axes[col].legend(handles=legend_patches, loc="lower right", fontsize=7)
        axes[col].axis("off")
        col += 1

    # -- Segmentation: original (beam-hardened) --------------------------------
    axes[col].imshow(
        seg_orig.T, origin="lower", cmap=cmap_seg, norm=norm_seg,
        interpolation="nearest",
    )
    axes[col].set_title("Original segmentation\n(beam-hardened FBP)")
    axes[col].legend(handles=legend_patches, loc="lower right", fontsize=7)
    axes[col].axis("off")
    col += 1

    # -- Segmentation: corrected -----------------------------------------------
    axes[col].imshow(
        seg_corr.T, origin="lower", cmap=cmap_seg, norm=norm_seg,
        interpolation="nearest",
    )
    axes[col].set_title("Corrected segmentation")
    axes[col].legend(handles=legend_patches, loc="lower right", fontsize=7)
    axes[col].axis("off")
    col += 1

    # -- Centre-line material-label profiles -----------------------------------
    ax = axes[col]
    if phantom is not None:
        ax.step(
            np.arange(phantom.shape[1]), phantom[mid],
            where="mid", lw=1.2, color="tab:green",
            linestyle=":", label="Ground truth",
        )
    ax.step(
        np.arange(seg_orig.shape[1]), seg_orig[mid],
        where="mid", lw=1.6, color="tab:blue", label="Original (BH)",
    )
    ax.step(
        np.arange(seg_corr.shape[1]), seg_corr[mid],
        where="mid", lw=1.6, color="tab:orange",
        linestyle="--", label="Corrected",
    )
    # Horizontal lines marking the threshold boundaries
    for i in range(n_labels - 1):
        ax.axhline(i + 0.5, color="gray", lw=0.6, linestyle=":")
    ax.set_yticks(list(range(n_labels)))
    ax.set_yticklabels(mat_names[:n_labels])
    ax.set_ylim(-0.5, n_labels - 0.3)
    ax.set_title("Centre-line material labels")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Material")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    plt.close()


# -- Clean-validation diagnostics (used by plot_clean_results.py) --------------

def _radial_profile_masked(image: np.ndarray, mask: np.ndarray, n_bins: int = 40):
    """
    Azimuthally-averaged radial profile of `image` over `mask` pixels; (radii, profile),
    empty bins NaN. Restricting to one material (e.g. PMMA) isolates cupping (profile rises
    toward the rim) far more cleanly than a through-the-bubbles centre line.
    """
    ii, jj = np.indices(image.shape)
    c0 = (image.shape[0] - 1) / 2.0
    c1 = (image.shape[1] - 1) / 2.0
    r = np.sqrt((ii - c0) ** 2 + (jj - c1) ** 2)

    if not mask.any():
        return np.array([]), np.array([])

    r_in = r[mask]
    vals = image[mask]
    bins = np.linspace(0.0, float(r_in.max()), n_bins + 1)
    idx = np.clip(np.digitize(r_in, bins) - 1, 0, n_bins - 1)

    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            prof[b] = vals[sel].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof


def compute_validation_metrics(
    original: np.ndarray,
    corrected: np.ndarray,
    phantom: np.ndarray,
    inner_frac: float = 0.3,
    outer_frac: float = 0.3,
) -> dict:
    """
    Quantify BH severity/correction vs phantom labels (0=air, 1=PMMA, 2=Al). Returns
    {"materials"[name][recon]={mean,std,cov}, "pmma_cupping_pct"[recon]=100*(rim-centre)/rim}.
    A good correction drives cupping toward 0 and lowers the PMMA CoV.
    """
    labels = [(0.0, "Air"), (1.0, "PMMA"), (2.0, "Al")]
    metrics = {"materials": {}}

    for label, name in labels:
        m = phantom == label
        if not m.any():
            continue
        entry = {}
        for arr, key in [(original, "original"), (corrected, "corrected")]:
            v = arr[m]
            mean = float(v.mean())
            std = float(v.std())
            entry[key] = {
                "mean": mean,
                "std": std,
                "cov": float(std / abs(mean)) if mean != 0 else float("nan"),
            }
        metrics["materials"][name] = entry

    pmma = phantom == 1.0
    cupping = {}
    for arr, key in [(original, "original"), (corrected, "corrected")]:
        _, prof = _radial_profile_masked(arr, pmma)
        prof = prof[~np.isnan(prof)]
        if prof.size == 0:
            cupping[key] = float("nan")
            continue
        n_in = max(1, int(round(inner_frac * prof.size)))
        n_out = max(1, int(round(outer_frac * prof.size)))
        centre_mean = prof[:n_in].mean()
        rim_mean = prof[-n_out:].mean()
        cupping[key] = (
            float(100.0 * (rim_mean - centre_mean) / rim_mean)
            if rim_mean != 0 else float("nan")
        )
    metrics["pmma_cupping_pct"] = cupping
    return metrics


def plot_cupping_validation_2d(
    original: np.ndarray,
    corrected: np.ndarray,
    phantom: np.ndarray,
    title: str = "Clean Validation",
    save_path: str = None,
) -> None:
    """
    Two-panel cupping diagnostic:
      1. PMMA-only radial attenuation profile (original vs corrected). Flat =
         no cupping; a rise toward the rim = residual beam hardening.
      2. Per-material mean +/- std bar chart (lower std = more uniform material).
    """
    pmma = phantom == 1.0
    r_o, p_o = _radial_profile_masked(original, pmma)
    r_c, p_c = _radial_profile_masked(corrected, pmma)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title)

    ax = axes[0]
    ax.plot(r_o, p_o, color="tab:blue", lw=1.6, label="Original (beam-hardened)")
    ax.plot(r_c, p_c, color="tab:orange", lw=1.6, linestyle="--", label="Corrected")
    ax.set_title("PMMA radial profile (cupping check)")
    ax.set_xlabel("Radius from centre [pixels]")
    ax.set_ylabel("Mean reconstructed attenuation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    names = ["Air", "PMMA", "Al"]
    labels = [0.0, 1.0, 2.0]
    o_means, o_stds, c_means, c_stds = [], [], [], []
    for label in labels:
        m = phantom == label
        o_means.append(float(original[m].mean()) if m.any() else np.nan)
        o_stds.append(float(original[m].std()) if m.any() else np.nan)
        c_means.append(float(corrected[m].mean()) if m.any() else np.nan)
        c_stds.append(float(corrected[m].std()) if m.any() else np.nan)
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, o_means, w, yerr=o_stds, capsize=4,
           color="tab:blue", alpha=0.8, label="Original")
    ax.bar(x + w / 2, c_means, w, yerr=c_stds, capsize=4,
           color="tab:orange", alpha=0.8, label="Corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Per-material mean +/- std")
    ax.set_ylabel("Reconstructed attenuation")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    plt.close()
