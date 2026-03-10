# imports for xray analysis
import xraylib
import spekpy as sp
# plotting and numpy
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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
    
def get_incidence(): 
    pass

if __name__ == "__main__":
   # render_phantom()
   r = sp.Spek(kvp=80, th=12)  # Generate a spectrum (80 kV, 12 degree tube angle)
   
   for id in [1.0,2.0]:
       print(generate_mu_vlaues(r,LUT_materials(id)))
       print("-"*100)
       print(generate_linear_attenuation_params(r,LUT_materials(id)))

   render_phantom()


#Simulate polychromatic ray - maybe for the sinogram stage
 
#Simulate ray effect from a 3D - maybe for the sinogram stage