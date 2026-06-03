#!/usr/bin/env python3
"""
barba_2d_phantom.py
-------------------
2-D Barba CT phantom — a 2-D analogue of barba_3D_phantom_1.py.

Phantom material labels
  0  –  Air        (background + bubble voids)
  1  –  PMMA       (plexiglass, C5H8O2, ρ = 1.18 g/cm³)
  2  –  Aluminium  (Al, ρ = 2.70 g/cm³)

Design mirrors the 3-D version:
  • Two overlapping PMMA ellipses  →  body + secondary lobe (analogous to
    the 3-D bottom and top ellipsoids, projected into the XY plane)
  • Al rods arranged on a ring     →  cross-sections of the 3-D cylinders
  • Random air-bubble voids        →  same role as in 3-D

Pipeline (main block):
  render_phantom_2d → calculate_I_2d (polychromatic) → astra_back_projection_2d (FBP)

Run
---
    python barba_2d_phantom.py

Output
------
    phantom_2d.png | sinogram_2d.png | reconstruction_2d.png
"""

import xraylib
import spekpy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import astra
import torch


# ── 1.  Phantom construction ─────────────────────────────────────────────────

def render_phantom_2d(
    size: int = 256,
    # Large outer ellipse — main body (≈ 3-D bottom blob)
    outer_rx: float = 0.40,
    outer_ry: float = 0.40,
    outer_cy: float = -0.05,
    # Small inner ellipse — secondary lobe (≈ 3-D top blob)
    inner_rx: float = 0.18,
    inner_ry: float = 0.20,
    inner_cy: float = 0.18,
    # Al rods arranged on a symmetric ring
    n_rods: int = 6,
    rod_radius: float = 0.035,
    rod_ring_radius: float = 0.27,
    # Air-bubble voids
    n_bubbles: int = 8,
    bubble_radius: float = 0.04,
    bubble_xy_range: tuple = (-0.22, 0.22),
    # Misc
    seed: int = 69,
    show: bool = True,
) -> np.ndarray:
    """
    Construct a 2-D Barba CT phantom.

    Parameters
    ----------
    size              : pixel grid (size × size)
    outer_rx / ry     : equatorial / polar radii of the main ellipse
    outer_cy          : y-centre of the main ellipse
    inner_rx / ry     : radii of the secondary lobe ellipse
    inner_cy          : y-centre of the secondary ellipse
    n_rods            : number of Al rods on the ring
    rod_radius        : radius of each rod
    rod_ring_radius   : distance of rod centres from the phantom centre
    n_bubbles         : number of air-bubble voids
    bubble_radius     : radius of each bubble
    bubble_xy_range   : uniform XY sampling range for bubble centres
    seed              : RNG seed for reproducibility
    show              : if True, saves phantom_2d.png

    Returns
    -------
    phantom : float32 ndarray, shape (size, size)
        0 = air, 1 = PMMA, 2 = Al
    """
    coords = np.linspace(-0.5, 0.5, size)
    # indexing="ij": phantom[x, y]  — consistent with 3-D version
    X, Y = np.meshgrid(coords, coords, indexing="ij")

    # Body = union of two PMMA ellipses
    outer = (X**2 / outer_rx**2 + (Y - outer_cy)**2 / outer_ry**2) < 1.0
    inner = (X**2 / inner_rx**2 + (Y - inner_cy)**2 / inner_ry**2) < 1.0
    phantom = (outer | inner).astype(np.float32)

    rng = np.random.default_rng(seed)

    # Al rods on a ring (random rotational offset for variety)
    angle_offset = rng.uniform(0, 2 * np.pi)
    for i in range(n_rods):
        angle = angle_offset + 2 * np.pi * i / n_rods
        cx = rod_ring_radius * np.cos(angle)
        cy = rod_ring_radius * np.sin(angle)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        phantom[(dist < rod_radius) & (phantom > 0)] = 2.0

    # Air bubbles (only inside PMMA, never inside rods)
    for _ in range(n_bubbles):
        cx = rng.uniform(*bubble_xy_range)
        cy = rng.uniform(*bubble_xy_range)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        phantom[(dist < bubble_radius) & (phantom == 1.0)] = 0.0

    if show:
        _plot_phantom(phantom)

    return phantom


def _plot_phantom(
    phantom: np.ndarray,
    title: str = "2-D Barba Phantom",
    fname: str = "phantom_2d.png",
) -> None:
    """Colour-coded plot of the phantom material map + centre-line profile."""
    cmap = ListedColormap(["#1a1a2e", "#4e9af1", "#f4a261"])   # air / PMMA / Al

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title)

    # Material map — transpose so X is horizontal, Y vertical, origin=lower
    im = axes[0].imshow(
        phantom.T, origin="lower", cmap=cmap,
        vmin=-0.5, vmax=2.5, interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=axes[0], ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Air (0)", "PMMA (1)", "Al (2)"])
    axes[0].set_title("Material map")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")

    # Centre-line profile
    mid = phantom.shape[0] // 2
    axes[1].step(np.arange(phantom.shape[1]), phantom[mid], where="mid", lw=1.2)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Air", "PMMA", "Al"])
    axes[1].set_title("Horizontal centre-line profile")
    axes[1].set_xlabel("Y pixel  (x = centre)")
    axes[1].set_ylabel("Material label")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ── 2.  Material / attenuation helpers ───────────────────────────────────────

def LUT_materials(numpy_id: float):
    """Map phantom label to xraylib compound string."""
    return {0.0: None, 1.0: "C5H8O2", 2.0: "Al"}.get(float(numpy_id))


def generate_mu_values(ray, compound: str) -> list:
    """Mass attenuation coefficients [cm²/g] at each spectral energy bin."""
    return [xraylib.CS_Total_CP(compound, e) for e in ray.get_k()]


def generate_linear_attenuation_params(ray, compound: str) -> np.ndarray:
    """Linear attenuation coefficients [cm⁻¹] at each spectral energy bin."""
    density = {"Al": 2.70, "C5H8O2": 1.18}[compound]
    return np.asarray(generate_mu_values(ray, compound)) * density


def de_bone(phantom: np.ndarray, material: str) -> np.ndarray:
    """Return a float32 binary mask for a single material."""
    m = material.lower()
    if m == "pmma":
        return (phantom == 1.0).astype(np.float32)
    if m in ("aluminum", "aluminium", "al"):
        return (phantom == 2.0).astype(np.float32)
    raise ValueError(f"Unknown material '{material}'. Use 'pmma' or 'aluminum'.")


# ── 3.  ASTRA 2-D forward / back projection ──────────────────────────────────

def astra_forward_project_2d(
    volume: np.ndarray,
    n_angles: int = 360,
) -> np.ndarray:
    """
    Parallel-beam forward projection using ASTRA FP_CUDA (GPU).

    Parameters
    ----------
    volume   : float32 ndarray (size, size)
    n_angles : projection angles uniformly distributed over [0, π)

    Returns
    -------
    sinogram : float32 ndarray (n_angles, size)
    """
    size = volume.shape[0]
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    vol_geom  = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom("parallel", 1.0, size, angles)

    vol_id  = astra.data2d.create("-vol",  vol_geom,  volume.astype(np.float32))
    proj_id = astra.data2d.create("-sino", proj_geom)

    cfg = astra.astra_dict("FP_CUDA")
    cfg["VolumeDataId"]     = vol_id
    cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    sino = astra.data2d.get(proj_id).astype(np.float32)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(vol_id)
    astra.data2d.delete(proj_id)

    return sino


def _astra_back_project_2d(
    grad_sino: np.ndarray,
    size: int,
    n_angles: int,
) -> np.ndarray:
    """
    Un-filtered back-projection (adjoint of forward projection).
    Used in the autograd backward pass of _AstraFP2DFunction.
    """
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    vol_geom  = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom("parallel", 1.0, size, angles)

    proj_id = astra.data2d.create("-sino", proj_geom, grad_sino.astype(np.float32))
    vol_id  = astra.data2d.create("-vol",  vol_geom)

    cfg = astra.astra_dict("BP_CUDA")
    cfg["ProjectionDataId"]     = proj_id
    cfg["ReconstructionDataId"] = vol_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    vol_grad = astra.data2d.get(vol_id).astype(np.float32)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(proj_id)
    astra.data2d.delete(vol_id)

    return vol_grad


class _AstraFP2DFunction(torch.autograd.Function):
    """
    Differentiable 2-D parallel-beam forward projector.
    Forward : FP_CUDA   (volume  → sinogram)
    Backward: BP_CUDA   (sinogram-grad → volume-grad)
    """

    @staticmethod
    def forward(ctx, volume_tensor: torch.Tensor, n_angles: int) -> torch.Tensor:
        vol_np  = volume_tensor.detach().cpu().numpy()
        sino_np = astra_forward_project_2d(vol_np, n_angles)
        ctx.save_for_backward(volume_tensor)
        ctx.n_angles = n_angles
        ctx.size     = vol_np.shape[0]
        return torch.from_numpy(sino_np).to(
            device=volume_tensor.device, dtype=volume_tensor.dtype
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_np     = grad_output.detach().cpu().numpy()
        vol_grad_np = _astra_back_project_2d(grad_np, ctx.size, ctx.n_angles)
        vol_grad = torch.from_numpy(vol_grad_np).to(
            device=grad_output.device, dtype=grad_output.dtype
        )
        return vol_grad, None   # None for n_angles (not a tensor)


def astra_forward_project_2d_differentiable(
    volume_tensor: torch.Tensor,
    n_angles: int = 360,
) -> torch.Tensor:
    """Differentiable 2-D forward projection, usable in torch autograd graphs."""
    return _AstraFP2DFunction.apply(volume_tensor, n_angles)


# ── 4.  Polychromatic intensity simulation ────────────────────────────────────

def calculate_I_2d(
    ray,
    mu_pmma: np.ndarray,
    mu_aluminum: np.ndarray,
    phantom: np.ndarray,
    scale: float = 5.0 / 256,
    add_gaussian_noise: float = 0.02,
    n_angles: int = 360,
) -> np.ndarray:
    """
    Simulate polychromatic CT measurements via Beer–Lambert law (2-D).

    Each spectral bin contributes I₀ · exp(−μ_pmma · L_pmma − μ_al · L_al),
    where L is the path length [cm] through each material.  A log-ratio
    sinogram −log(I / I₀) is returned, mimicking the scanner output.

    Parameters
    ----------
    ray               : SpekPy Spek object (already generated)
    mu_pmma           : linear attenuation [cm⁻¹], length = energy bins
    mu_aluminum       : linear attenuation [cm⁻¹], length = energy bins
    phantom           : float32 (size, size) — material label array
    scale             : cm per voxel  (physical_width_cm / n_pixels)
    add_gaussian_noise: fractional Gaussian noise σ relative to max(I)
    n_angles          : number of projection angles

    Returns
    -------
    sinogram : float32 (n_angles, size)
        Effective polychromatic attenuation  −log(I / I₀)
    """
    pmma_proj = astra_forward_project_2d(de_bone(phantom, "pmma"),     n_angles)
    al_proj   = astra_forward_project_2d(de_bone(phantom, "aluminum"), n_angles)

    fluence = ray.get_spk()
    I_total = np.zeros_like(pmma_proj, dtype=np.float64)

    for n, I0 in enumerate(fluence):
        p = pmma_proj * mu_pmma[n] * scale + al_proj * mu_aluminum[n] * scale
        I_total += I0 * np.exp(-p)

    noise   = np.random.normal(0, add_gaussian_noise * np.max(I_total), I_total.shape)
    I_total = np.clip(I_total + noise, 1e-10, None)

    I0_total = float(np.sum(fluence))
    return (-np.log(I_total / I0_total)).astype(np.float32)


# ── 5.  FBP / SIRT reconstruction ────────────────────────────────────────────

def astra_back_projection_2d(
    sinogram: np.ndarray,
    n_angles: int = 360,
    algorithm: str = "FBP_CUDA",
    iterations: int = 200,
) -> np.ndarray:
    """
    Reconstruct a 2-D slice from a sinogram.

    Parameters
    ----------
    sinogram   : float32 (n_angles, n_detectors)
    n_angles   : must match sinogram.shape[0]
    algorithm  : 'FBP_CUDA' (default, fast) or 'SIRT_CUDA' (iterative)
    iterations : number of iterations — used only for SIRT_CUDA

    Returns
    -------
    reconstruction : float32 (n_detectors, n_detectors)
    """
    n_det  = sinogram.shape[1]
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    vol_geom  = astra.create_vol_geom(n_det, n_det)
    proj_geom = astra.create_proj_geom("parallel", 1.0, n_det, angles)

    sino_id = astra.data2d.create("-sino", proj_geom, sinogram.astype(np.float32))
    vol_id  = astra.data2d.create("-vol",  vol_geom)

    cfg = astra.astra_dict(algorithm)
    cfg["ProjectionDataId"]     = sino_id
    cfg["ReconstructionDataId"] = vol_id
    if algorithm == "FBP_CUDA":
        cfg["FilterType"] = "ram-lak"

    alg_id = astra.algorithm.create(cfg)
    if algorithm == "SIRT_CUDA":
        astra.algorithm.run(alg_id, iterations)
    else:
        astra.algorithm.run(alg_id)

    recon = astra.data2d.get(vol_id).astype(np.float32)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(vol_id)

    return recon


# ── 6.  Plotting helpers ──────────────────────────────────────────────────────

def plot_sinogram_2d(
    sinogram: np.ndarray,
    title: str = "Sinogram (2-D)",
    fname: str = "sinogram_2d.png",
) -> None:
    """Save a sinogram image and a central line profile."""
    n_angles, n_det = sinogram.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].imshow(
        sinogram, aspect="auto", cmap="gray",
        extent=[0, n_det, 180, 0],
    )
    axes[0].set_title("Sinogram")
    axes[0].set_xlabel("Detector position [pixels]")
    axes[0].set_ylabel("Angle [°]")

    mid_angle = n_angles // 2
    axes[1].plot(sinogram[mid_angle], lw=1.2)
    axes[1].set_title(f"Line profile at angle {mid_angle * 180 // n_angles}°")
    axes[1].set_xlabel("Detector position [pixels]")
    axes[1].set_ylabel("Effective attenuation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_reconstruction_2d(
    reconstruction: np.ndarray,
    phantom: np.ndarray = None,
    title: str = "FBP Reconstruction (2-D)",
    fname: str = "reconstruction_2d.png",
) -> None:
    """
    Save:
      • Reconstruction image
      • Central line profile (for spotting the beam-hardening cupping artefact)
      • Ground-truth phantom for visual comparison (optional)
    """
    ncols = 3 if phantom is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(title)

    # Reconstruction — transpose to match phantom orientation
    axes[0].imshow(reconstruction.T, origin="lower", cmap="gray")
    axes[0].set_title("Reconstruction")
    axes[0].axis("off")

    # Centre-line profile
    mid = reconstruction.shape[0] // 2
    axes[1].plot(reconstruction[mid], lw=1.2)
    axes[1].set_title("Centre-line profile  (look for cupping)")
    axes[1].set_xlabel("Pixel")
    axes[1].set_ylabel("Reconstructed attenuation")
    axes[1].grid(True, alpha=0.3)

    # Ground-truth comparison
    if phantom is not None:
        cmap_gt = ListedColormap(["#1a1a2e", "#4e9af1", "#f4a261"])
        axes[2].imshow(
            phantom.T, origin="lower", cmap=cmap_gt,
            vmin=-0.5, vmax=2.5, interpolation="nearest",
        )
        axes[2].set_title("Ground-truth phantom")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ── 7.  Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    SIZE    = 256
    # scale: cm per voxel — controls physical beam-hardening severity.
    # 5.0 cm total object width → SCALE ≈ 0.020 cm/voxel gives clearly
    # visible polychromatic cupping in the reconstruction.
    SCALE   = 5.0 / SIZE

    # ── Spectrum ──────────────────────────────────────────────────────────────
    print("Generating X-ray spectrum (120 kV, 12° anode) …")
    ray = sp.Spek(kvp=120, th=12, physics="spekcalc")

    mu_pmma = generate_linear_attenuation_params(ray, "C5H8O2")
    mu_al   = generate_linear_attenuation_params(ray, "Al")

    # Save for downstream use
    mu_values = np.stack([mu_pmma, mu_al], axis=0)
    np.save("mu_values_2d.npy", mu_values)
    energy_bins = ray.get_k()
    np.save("energy_bins_2d.npy", energy_bins)

    # ── Phantom ───────────────────────────────────────────────────────────────
    print("Rendering 2-D phantom …")
    phantom = render_phantom_2d(size=SIZE, show=True)   # → phantom_2d.png

    # ── Polychromatic sinogram ────────────────────────────────────────────────
    print("Computing polychromatic sinogram (360 angles) …")
    sinogram = calculate_I_2d(
        ray, mu_pmma, mu_al, phantom,
        scale=SCALE, add_gaussian_noise=0.02,
    )
    plot_sinogram_2d(
        sinogram,
        title="Polychromatic Sinogram — 2-D Barba Phantom",
    )

    # ── FBP reconstruction ────────────────────────────────────────────────────
    print("Running FBP reconstruction (Ram-Lak filter) …")
    recon = astra_back_projection_2d(sinogram, algorithm="FBP_CUDA")
    plot_reconstruction_2d(
        recon, phantom=phantom,
        title="FBP Reconstruction — 2-D Barba Phantom",
    )

    print("\nAll done.  Output files:")
    print("  phantom_2d.png")
    print("  sinogram_2d.png")
    print("  reconstruction_2d.png")
