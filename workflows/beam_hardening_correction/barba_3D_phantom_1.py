# imports for xray analysis
import xraylib
import spekpy as sp
# plotting and numpy
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

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
        fig = go.Figure(
            data=go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=marker_size, color=vs, opacity=marker_opacity),
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(showticklabels=False, title=""),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=700,
            height=700,
        )
        fig.show()

    # Side projection (max along Y, Z is abscissa)
    if show_projection:
        projection = np.max(volume, axis=1)
        plt.figure(figsize=(6, 6))
        plt.imshow(projection, origin="lower", aspect="auto")
        plt.xlabel("Z")
        plt.ylabel("X")
        plt.title("Side projection")
        plt.show()

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

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'Sinogram (row {mid_row})',
            f'Projection at angle {mid_angle}',
            'Line profile through center'
        ]
    )

    # Classic sinogram: one horizontal slice
    fig.add_trace(
        go.Heatmap(z=sinogram[mid_row], colorscale='gray', showscale=False),
        row=1, col=1
    )
    fig.update_xaxes(title_text='Detector column', row=1, col=1)
    fig.update_yaxes(title_text='Angle index', row=1, col=1)

    # What the detector sees at one angle
    fig.add_trace(
        go.Heatmap(z=sinogram[:, mid_angle, :], colorscale='gray', showscale=False),
        row=1, col=2
    )
    fig.update_xaxes(title_text='Detector column', row=1, col=2)
    fig.update_yaxes(title_text='Detector row', row=1, col=2)

    # Line profile: one row of the sinogram at one angle
    # This shows the attenuation profile — you should see higher peaks where Al is
    profile = sinogram[mid_row, mid_angle, :]
    fig.add_trace(
        go.Scatter(y=profile, mode='lines', name='Attenuation'),
        row=1, col=3
    )
    fig.update_xaxes(title_text='Detector column', row=1, col=3)
    fig.update_yaxes(title_text='Attenuation', row=1, col=3)

    fig.update_layout(title=title, height=400, width=1400, showlegend=False)
    fig.show()

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
        p_pmma = pmma_projection * mu_pmma[n]*scale
        p_al = al_projection * mu_aluminum[n]*scale
        I_total += I_0 * np.exp(-(p_pmma + p_al))
    #add noise after summing contributions for an energy
        if add_gaussian_noise:
            I_total += np.random.normal(0, add_gaussian_noise)
    return p_al



"""
---------------------------------------------------------------------------------------------------
Back-projection using ASTRA toolbox, SIRT3D_CUDA algorithm.
---------------------------------------------------------------------------------------------------
"""

def astra_back_projection(sinogram, N_ANGLES=360):
    N = 128
    angles = np.linspace(0, np.pi, N_ANGLES, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, N, N, angles)
    vol_geom = astra.create_vol_geom(N, N, N)
    sino_id = astra.data3d.create('-proj3d', proj_geom, sinogram)
    # Calculate backprojection
    backprojection_id = astra.data3d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = backprojection_id
    algorithm_id = astra.algorithm.create(cfg)

    astra.algorithm.run(algorithm_id,iterations=    100)

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
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f'Axial (z={mid})',
            f'Coronal (y={mid})',
            f'Sagittal (x={mid})'
        ]
    )

    fig.add_trace(
        go.Heatmap(z=reconstruction[mid, :, :], colorscale='gray', showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=reconstruction[:, mid, :], colorscale='gray', showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=reconstruction[:, :, mid], colorscale='gray', showscale=True),
        row=1, col=3
    )

    fig.update_layout(title=title, height=400, width=1400)
    fig.show()

    # 3D volume rendering — show voxels above a threshold
    threshold = (reconstruction.max() + reconstruction.min()) / 2
    x, y, z = np.where(reconstruction > threshold)
    vals = reconstruction[x, y, z]

    fig3d = go.Figure(data=go.Scatter3d(
        x=x[::5], y=y[::5], z=z[::5],
        mode='markers',
        marker=dict(
            size=1.5,
            color=vals[::5],
            colorscale='gray',
            opacity=0.3,
            colorbar=dict(title='Attenuation')
        )
    ))
    fig3d.update_layout(
        title=f'{title} (3D)',
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
        ),
        width=700, height=700
    )
    fig3d.show()

    # Line profile through center — this is where you see cupping
    profile = reconstruction[mid, mid, :]
    fig_line = go.Figure(data=go.Scatter(
        y=profile, mode='lines', name='Attenuation'
    ))
    fig_line.update_layout(
        title='Line profile through center (look for cupping)',
        xaxis_title='Pixel position',
        yaxis_title='Reconstructed attenuation',
        height=400, width=800
    )
    fig_line.show()












# def get_incidence(): 
#     pass

if __name__ == "__main__":
   # render_phantom()
    r = sp.Spek(kvp=180, th=12)  # Generate a spectrum (120 kV, 12 degree tube angle)

    
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
