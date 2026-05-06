import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction(
        reconstruction,
        scatter_stride=5,
        marker_size=1.5,
        marker_opacity=0.3,
        title="Reconstruction",
        save_path=None
    ):
    threshold = (reconstruction.max() + reconstruction.min()) / 6
    x, y, z = np.where(reconstruction > threshold)
    vals = reconstruction[x, y, z]
    xs, ys, zs, vs = (
        x[::scatter_stride],
        y[::scatter_stride],
        z[::scatter_stride],
        vals[::scatter_stride],
    )
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=vs, s=marker_size, alpha=marker_opacity, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    plt.close()
    