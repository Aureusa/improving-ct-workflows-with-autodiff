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
    """
    Visualise a 2-D reconstruction with three panels:
      1. imshow of the slice
      2. Horizontal centre-line profile (reveals the beam-hardening cupping)

    Parameters
    ----------
    reconstruction : float32 ndarray (n_pixels, n_pixels)
    title          : figure title
    save_path      : if given, save the figure to this path
    """
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
    Side-by-side comparison of original (beam-hardened) and corrected reconstructions,
    optionally alongside the ground-truth phantom.

    Parameters
    ----------
    original  : float32 ndarray (n_pixels, n_pixels)
    corrected : float32 ndarray (n_pixels, n_pixels)
    phantom   : float32 ndarray (n_pixels, n_pixels), optional
        Material-label array (0 = air, 1 = PMMA, 2 = Al).
        When supplied a fourth image panel is added and the phantom centre-line
        is overlaid on the profiles plot using a right-hand y-axis.
    """
    from matplotlib.colors import ListedColormap

    mid  = original.shape[0] // 2
    vmin = min(original.min(), corrected.min())
    vmax = max(original.max(), corrected.max())

    ncols = 4 if phantom is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(title)

    col = 0  # running column index

    # ── Phantom image (optional) ──────────────────────────────────────────────
    if phantom is not None:
        cmap_gt = ListedColormap(["#1a1a2e", "#4e9af1", "#f4a261"])
        axes[col].imshow(
            phantom.T, origin="lower", cmap=cmap_gt,
            vmin=-0.5, vmax=2.5, interpolation="nearest",
        )
        axes[col].set_title("Ground-truth phantom")
        axes[col].axis("off")
        col += 1

    # ── Reconstruction images ─────────────────────────────────────────────────
    axes[col].imshow(original.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[col].set_title("Original (beam-hardened)")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(corrected.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[col].set_title("Corrected")
    axes[col].axis("off")
    col += 1

    # ── Centre-line profiles ──────────────────────────────────────────────────
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
    Apply the learned ISP thresholds as hard cuts to both the beam-hardened
    and corrected reconstructions and plot the resulting segmentations
    side-by-side so the effect of beam hardening is clearly visible.

    Parameters
    ----------
    original   : float32 ndarray (n_pixels, n_pixels)
        FBP reconstruction from the polychromatic (beam-hardened) sinogram.
    corrected  : float32 ndarray (n_pixels, n_pixels)
        FBP reconstruction from the corrected monochromatic sinogram.
    thresholds : ndarray (M,)
        Learned threshold values from ISP2D (e.g. shape (2,) for air/PMMA/Al).
        Will be sorted before use.
    phantom    : float32 ndarray (n_pixels, n_pixels), optional
        Ground-truth material-label array (0=air, 1=PMMA, 2=Al).
    title      : str
    save_path  : str, optional
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    thresholds = np.sort(thresholds)          # ensure ascending order
    n_labels   = len(thresholds) + 1          # e.g. 3 for 2 thresholds
    vmin, vmax = -0.5, n_labels - 0.5        # noqa: F841

    # Hard segmentation via digitize: result is 0 … n_labels-1
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

    # ── Ground-truth phantom (optional) ──────────────────────────────────────
    if phantom is not None:
        axes[col].imshow(
            phantom.T, origin="lower", cmap=cmap_seg, norm=norm_seg,
            interpolation="nearest",
        )
        axes[col].set_title("Ground truth")
        axes[col].legend(handles=legend_patches, loc="lower right", fontsize=7)
        axes[col].axis("off")
        col += 1

    # ── Segmentation: original (beam-hardened) ────────────────────────────────
    axes[col].imshow(
        seg_orig.T, origin="lower", cmap=cmap_seg, norm=norm_seg,
        interpolation="nearest",
    )
    axes[col].set_title("Original segmentation\n(beam-hardened FBP)")
    axes[col].legend(handles=legend_patches, loc="lower right", fontsize=7)
    axes[col].axis("off")
    col += 1

    # ── Segmentation: corrected ───────────────────────────────────────────────
    axes[col].imshow(
        seg_corr.T, origin="lower", cmap=cmap_seg, norm=norm_seg,
        interpolation="nearest",
    )
    axes[col].set_title("Corrected segmentation")
    axes[col].legend(handles=legend_patches, loc="lower right", fontsize=7)
    axes[col].axis("off")
    col += 1

    # ── Centre-line material-label profiles ───────────────────────────────────
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
