# imports for xray analysis
import xraylib
import spekpy as sp
# plotting and numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

#astra
import astra


def render_phantom(
    size: int = 128,
    # Bottom blob 
    bot_r: float = 0.40,
    bot_z_center: float = -0.15,
    bot_z_radius: float = 0.40,
    # Top blob 
    top_r: float = 0.28,
    top_z_center: float = 0.20,
    top_z_radius: float = 0.28,
    # Rods  
    n_cylinders: int = 6,
    cyl_radius: float = 0.04,
    cyl_z_range: tuple = (-0.4, 0.4),
    # Bubbles
    n_bubbles: int = 8,
    bubble_radius: float = 0.05,
    bubble_xy_range: tuple = (-0.25, 0.25),
    bubble_z_range: tuple = (-0.35, 0.35),
    # Scatter plot
    scatter_stride: int = 5,
    marker_size: float = 1.5,
    marker_opacity: float = 0.3,
    # Output
    seed: int = 69,
    show_3d: bool = True,
    show_projection: bool = True,
) -> np.ndarray:
    """
    Define and visualise a CT phantom.

    Parameters
    ----------
    size            : voxel grid size (size^3)
    bot_r / top_r   : radii of the two ellipsoidal blobs
    bot/top_z_center: z-centre of each blob
    bot/top_z_radius: polar (z-axis) radii of each blob
    n_cylinders     : number of rods punched through the phantom
    cyl_radius      : radius of each rod
    cyl_z_range     : uniform sampling range for rod z-intercept
    n_bubbles       : number of air-bubble voids
    bubble_radius   : radius of each bubble
    bubble_xy_range : XY sampling range for bubble centres
    bubble_z_range  : Z  sampling range for bubble centres
    scatter_stride  : plot every Nth filled voxel (for speed)
    marker_size     : 3-D scatter marker size
    marker_opacity  : 3-D scatter marker opacity
    seed            : RNG seed for reproducibility
    show_3d         : display interactive 3-D scatter
    show_projection : display 2-D side projection

    Returns
    -------
    volume : float32 ndarray (size, size, size)
        0 = air, 1 = phantom body, 2 = rod material
    """
    coords = np.linspace(-0.5, 0.5, size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    bot = (R**2 / bot_r**2 + (Z - bot_z_center)**2 / bot_z_radius**2) < 1.0
    top = (R**2 / top_r**2 + (Z - top_z_center)**2 / top_z_radius**2) < 1.0
    volume = (bot | top).astype(np.float32)

    rng = np.random.default_rng(seed)

    # Rods (perpendicular to Z)
    for _ in range(n_cylinders):
        z_pos = rng.uniform(*cyl_z_range)
        angle = rng.uniform(0, np.pi)
        dx, dy = np.cos(angle), np.sin(angle)
        dist = np.sqrt(
            (Y * 0 - (Z - z_pos) * dy) ** 2
            + ((Z - z_pos) * dx - X * 0) ** 2
            + (X * dy - Y * dx) ** 2
        )
        volume[(dist < cyl_radius) & (volume > 0)] = 2.0

    # Air bubbles
    for _ in range(n_bubbles):
        cx = rng.uniform(*bubble_xy_range)
        cy = rng.uniform(*bubble_xy_range)
        cz = rng.uniform(*bubble_z_range)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        volume[(dist < bubble_radius) & (volume != 2.0)] = 0.0

    # 3-D scatter plot, stride for visibility
    if show_3d:
        x, y, z = np.where(volume > 0)
        vals = volume[x, y, z]
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
        plt.savefig("3d_phantom.png", dpi=100)
        plt.close()

    # Side projection (max along Y, Z is abscissa)
    if show_projection:
        projection = np.max(volume, axis=1)
        plt.figure(figsize=(6, 6))
        plt.imshow(projection, origin="lower", aspect="auto")
        plt.xlabel("Z")
        plt.ylabel("X")
        plt.title("Side projection")
        plt.savefig("side_projection.png")
        plt.close()

    return volume

def LUT_materials(numpy_id):
    if numpy_id == 0.0:
        return None
    if numpy_id == 1.0:
        return "C5H8O2" # plexiglass
    if numpy_id == 2.0:
        return "Al"

# Using spekpy we generate beam spectrum
def generate_mu_vlaues(ray, compound):
    mu_values=[]
    bins = ray.get_k()
    for e in bins:
        mu=xraylib.CS_Total_CP(compound, e)
        mu_values.append(mu)
    return mu_values
    
def generate_linear_attenuation_params(ray, compound):
    if compound == "Al":
        density = 2.70       
    elif compound == "C5H8O2":       
        density = 1.18
    attenuation=np.array(generate_mu_vlaues(ray,compound))
    
    attenuation = attenuation * density
    return attenuation
    
    
#disect materials (only PMMA or ALUMINUM RODS)
def de_bone(phantom, material):
    if material.lower()=="pmma":
        phantom_debonned = phantom == 1.0
    elif material.lower()=="aluminum":
        phantom_debonned = phantom == 2.0
    else:
        raise AttributeError("Material not supported. Use 'PMMA' or 'Aluminum'.")
    
    return phantom_debonned.astype(np.float32)


"""
Do forward projection of the phantom using ASTRA toolbox, with a parallel beam geometry. Sensor is as tall as the volume and long as the volume, with 360 views over 180 degrees.
The function takes the 3D volume of the phantom (of homogenoeus material) and returns the corresponding sinogram 

"""


#volume is phantom mask
def astra_forward_project(volume, n_angles=360):

    #rotate around Z
    #vol = np.transpose(volume, (2, 0, 1))  
    size = volume.shape[0]
    #360 steps over 180 degrees
    angles = np.linspace(0, np.pi, n_angles, endpoint=False) 
    vol_geom = astra.create_vol_geom(size, size, size)
    #parallel beam geometry for 3D projection, sensors as tall as the volume and long as the volume, 360 views
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, size, size, angles) 
    
    # Explicitly create data objects
    #create the volume data object with the phantom
    vol_id = astra.data3d.create('-vol', vol_geom, volume)
    #create the projection data object 
    proj_id = astra.data3d.create('-proj3d', proj_geom)
    
    # Create and run forward projection algorithm
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['VolumeDataId'] = vol_id
    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id)
    
    projection = astra.data3d.get(proj_id)
    
    # Clean up
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)
    
    return projection 


def plot_sinogram(sinogram, title="Sinogram"):
    n_rows, n_angles, n_cols = sinogram.shape
    mid_row = n_rows // 2
    mid_angle = 90

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title)

    # Classic sinogram: one horizontal slice
    axes[0].imshow(sinogram[mid_row], aspect='auto', cmap='gray')
    axes[0].set_title(f'Sinogram (row {mid_row})')
    axes[0].set_xlabel('Detector column')
    axes[0].set_ylabel('Angle index')

    # What the detector sees at one angle
    axes[1].imshow(sinogram[:, mid_angle, :], aspect='auto', cmap='gray')
    axes[1].set_title(f'Projection at angle {mid_angle}')
    axes[1].set_xlabel('Detector column')
    axes[1].set_ylabel('Detector row')

    # Line profile: one row of the sinogram at one angle
    profile = sinogram[mid_row, mid_angle, :]
    axes[2].plot(profile)
    axes[2].set_title('Line profile through center')
    axes[2].set_xlabel('Detector column')
    axes[2].set_ylabel('Attenuation')

    plt.tight_layout()
    plt.savefig("sinogram.png", dpi=100)
    plt.close()

"""
---------------------------------------------------------------------------------------------------
Full I calculation for a polychromatic ray passing through the phantom, using the Beer-Lambert law.
--------------------------------------------------------------------------------------------------
"""
def calculate_I(ray,mu_pmma,mu_aluminum,phantom,scale=0.1,add_gaussian_noise=0.02):
    pmma_part=de_bone(phantom, "pmma")
    al_part=de_bone(phantom, "aluminum")
    pmma_projection=astra_forward_project(pmma_part)
    al_projection=astra_forward_project(al_part)
    fluence = ray.get_spk()
    I_total = 0

    for n,e in enumerate(fluence):
        I_0= e
        p_pmma = pmma_projection * mu_pmma[n] * scale
        p_al = al_projection * mu_aluminum[n] * scale
        I_total += I_0 * np.exp(-(p_pmma + p_al))

    # Add Gaussian noise to simulate measurement imperfections
    noise = np.random.normal(0, add_gaussian_noise * np.max(I_total), size=I_total.shape)
    I_total += noise

    I0_total = np.sum(fluence)
    rec = -np.log(I_total / I0_total)
    return rec


"""
---------------------------------------------------------------------------------------------------
Back-projection using ASTRA toolbox, SIRT3D_CUDA algorithm.
---------------------------------------------------------------------------------------------------
"""

def astra_back_projection(sinogram, n_angles=360):
    N = 128
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, N, N, angles)
    vol_geom = astra.create_vol_geom(N, N, N)
    sino_id = astra.data3d.create('-proj3d', proj_geom, sinogram)

    # Calculate backprojection
    backprojection_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = backprojection_id
    algorithm_id = astra.algorithm.create(cfg)

    astra.algorithm.run(algorithm_id,iterations=1000)

    backprojection = astra.data3d.get(backprojection_id)
    #backprojection   = np.transpose(backprojection, (1, 2, 0))
    # Clean up
    astra.data3d.delete([sino_id, backprojection_id])
    astra.algorithm.delete(algorithm_id)
    
    return backprojection
# def run_projection


# def calculate_attenuation

# #WHERE WAS THE NOISE ADDED?

# def sum_attnuations

# def 



def plot_reconstruction(reconstruction, title="FBP Reconstruction"):
    size = reconstruction.shape[0]
    mid = size // 2

    # 2D slices in all three directions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title)
    axes[0].imshow(reconstruction[mid, :, :], cmap='gray')
    axes[0].set_title(f'Axial (z={mid})')
    axes[1].imshow(reconstruction[:, mid, :], cmap='gray')
    axes[1].set_title(f'Coronal (y={mid})')
    im = axes[2].imshow(reconstruction[:, :, mid], cmap='gray')
    axes[2].set_title(f'Sagittal (x={mid})')
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig("reconstruction_slices.png", dpi=100)
    plt.close()

    # 3D volume rendering — show voxels above a threshold
    threshold = (reconstruction.max() + reconstruction.min()) / 2
    x, y, z = np.where(reconstruction > threshold)
    vals = reconstruction[x, y, z]
    fig3d = plt.figure(figsize=(7, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(x[::5], y[::5], z[::5], c=vals[::5], cmap='gray', s=1.5, alpha=0.3)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_title(f'{title} (3D)')
    plt.tight_layout()
    plt.savefig("3d_reconstruction.png", dpi=100)
    plt.close()

    # Line profile through center — this is where you see cupping
    profile = reconstruction[mid, mid, :]
    plt.figure(figsize=(8, 4))
    plt.plot(profile)
    plt.title('Line profile through center (look for cupping)')
    plt.xlabel('Pixel position')
    plt.ylabel('Reconstructed attenuation')
    plt.tight_layout()
    plt.savefig("line_profile.png", dpi=100)
    plt.close()









# def get_incidence(): 
#     pass

if __name__ == "__main__":
   # render_phantom()
    # Work around SpekPy v2 + NumPy 2.x incompatibility in default physics path.
    r = sp.Spek(kvp=180, th=12, physics="spekcalc")  # Generate a spectrum (120 kV, 12 degree tube angle)
    
    pmma_mu = generate_linear_attenuation_params(r, ("C5H8O2"))  # Get attenuation coefficients for PMMA
    al_mu = generate_linear_attenuation_params(r, "Al")  # Get attenuation coefficients for Aluminum
    phantom=render_phantom(show_projection=False)

    pmma_mask = de_bone(phantom, "pmma")    
    al_mask = de_bone(phantom, "aluminum")  

    pmma_density = pmma_mask
    al_density = al_mask
    
    I=calculate_I(r, pmma_mu, al_mu, phantom, scale=0.5 / 128, add_gaussian_noise=0.02)
    plot_sinogram(I, title="Simulated Sinogram with Polychromatic Beam and Noise")
    
    reconstruction = astra_back_projection(I)
    plot_reconstruction(reconstruction, title="FBP Reconstruction")
    #Simulate polychromatic ray - maybe for the sinogram stage
 
#Simulate ray effect from a 3D - maybe for the sinogram stage