import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

class Plate:
    def __init__(self, size, SDF, initial_temp, t_max, alpha, spatial_step, nt):
        """
        Initializes a 2D plate for forward simulation of the heat equation

        Parameters
        ----------
        size : float
            Size of the simulation domain (size x size)
        SDF : function
            Signed distance function defining the geometry of the plate
        initial_temp : function
            Function that takes (x, y) coordinates and returns the initial temperature at that point
        t_max : float
            Total simulation time
        alpha : float
            Thermal diffusivity constant
        spatial_step : float
            Spatial step size for the simulation grid
        nt : int
            Number of time steps for the simulation
        """
        self.initial_temp = initial_temp
        self.size = size
        self.SDF = SDF
        self.t_max = t_max
        self.alpha = alpha

        # Calculate nx from spatial_step and size
        self.nx = int(size / spatial_step) + 1
        self.dx = size / (self.nx - 1)

        # Calculate dt from t_max and nt
        self.nt = nt
        self.dt = t_max / (nt - 1)

        # Check numerical stability: Fo = alpha * dt / dx^2 <= 0.25
        Fo = alpha * self.dt / self.dx**2
        if Fo > 0.25:
            raise ValueError(
                f"Numerical instability detected: Fo={Fo:.4f} > 0.25. Reduce dt or increase dx."
            )

        # Create coordinate grid
        x_coords = np.linspace(-self.size / 2, self.size / 2, self.nx)
        xx, yy = np.meshgrid(x_coords, x_coords)
        self.coords = np.stack([xx, yy], axis=-1)

        # Precompute SDF values for the grid
        SDF_values = self.SDF(self.coords)
        self.shape_mask = SDF_values <= 0.0  # interior + boundary
        self.interior_mask = SDF_values < -self.dx  # interior only
        self.boundary_mask = self.shape_mask & ~self.interior_mask

        # Initialize temperature array
        self.temperature = np.zeros((self.nt, self.nx, self.nx))
        self._initialize_temperature()

    def _initialize_temperature(self):
        """
        Initializes the temperature distribution based on the
        initial temperature function and the geometry defined by the SDF.
        """
        self.temperature[0][self.shape_mask] = self.initial_temp(
            self.coords[self.shape_mask]
        )

    def step(self, t):
        """
        Advances the simulation by one time step using the finite difference method.

        Parameters
        ----------
        t : int
            Current time step index
        """
        T = self.temperature[t - 1]
        T_new = np.copy(T)

        # Computes Laplacian. np.roll is cyclical, so SDF should not touch the boundaries
        d2T_dx2 = (np.roll(T, -1, axis=1) - 2 * T + np.roll(T, 1, axis=1)) / self.dx**2
        d2T_dy2 = (np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0)) / self.dx**2

        # Update interior points
        T_new[self.interior_mask] += (
            self.alpha
            * self.dt
            * (d2T_dx2[self.interior_mask] + d2T_dy2[self.interior_mask])
        )

        # Enforce BCs: T=0 at boundary (and outside)
        T_new[self.boundary_mask] = 0.0
        T_new[~self.shape_mask] = 0.0

        self.temperature[t] = T_new

    def run(self):
        """
        Runs the forward simulation for the specified number of time steps.
        """
        for t in tqdm(range(1, self.nt), desc="Running forward simulation"):
            self.step(t)


    def visualize(self, step=None, figsize=(8, 6), cmap='hot', sensors=None):
        """
        Visualizes the temperature distribution at a given time step.
        If no step is given, shows an animation across all time steps.

        Parameters
        ----------
        step : int or None
            Time step to visualize. If None, animates all steps.
        figsize : tuple
            Figure size in inches.
        cmap : str
            Matplotlib colormap name.
        sensors : list of tuples, optional
            List of (x, y) coordinates for thermocouple/sensor locations.
        """
        # Mask outside cells as NaN so they render transparently
        def masked(t):
            frame = self.temperature[t].copy().astype(float)
            frame[~self.shape_mask] = np.nan
            return frame

        vmin = np.nanmin(self.temperature[0])
        vmax = np.nanmax(self.temperature[0])
        extent = [-self.size/2, self.size/2, -self.size/2, self.size/2]

        # Configure colormap to render NaN values (outside boundary) as black
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='black')

        if step is not None:
            # Single frame
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(masked(step), origin='lower', extent=extent,
                        cmap=cmap_obj, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label='Temperature')
            t_val = step * self.dt
            ax.set_title(f't = {t_val:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            if sensors is not None:
                sx, sy = zip(*sensors)
                ax.scatter(sx, sy, color='cyan', marker='x', s=50, label='Thermocouples')
                ax.legend()
                
            plt.tight_layout()
            plt.show()
        else:
            # Interactive Slider
            fig, ax = plt.subplots(figsize=figsize)
            plt.subplots_adjust(bottom=0.25)
            im = ax.imshow(masked(0), origin='lower', extent=extent,
                        cmap=cmap_obj, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label='Temperature')
            title = ax.set_title('t = 0.0000')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            if sensors is not None:
                sx, sy = zip(*sensors)
                ax.scatter(sx, sy, color='cyan', marker='x', s=50, label='Thermocouples')
                ax.legend(loc='upper right')
            
            # Create slider axis and slider
            ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
            time_slider = Slider(
                ax=ax_slider,
                label='Time Step',
                valmin=0,
                valmax=self.nt - 1,
                valinit=0,
                valstep=1
            )

            # Update function for the slider
            def update(val):
                t = int(time_slider.val)
                im.set_data(masked(t))
                title.set_text(f't = {t * self.dt:.4f}')
                fig.canvas.draw_idle()

            time_slider.on_changed(update)
            plt.show()
            # Return the slider object to prevent it from being garbage collected
            return time_slider


    def export_sparse(self, points, filename, step=1, noise_std=0.0):
        """
        Export the temperature data at specified points for all time steps to a CSV file.
        Parameters
        ----------
        points : list of tuples
            A list of (x, y) coordinates where the temperature data should be extracted.
        filename : str
            The name of the CSV file to which the data will be exported.
        """
        import pandas as pd
        
        data = []
        for i, (x, y) in enumerate(points):
            for t in range(0, self.nt, step):
                time = t * self.dt
                # Convert physical coordinates to grid indices
                ix = int(round((x + self.size / 2) / self.dx))
                iy = int(round((y + self.size / 2) / self.dx))
                data.append({
                    'sensor_id': i,
                    'x': x,
                    'y': y,
                    'time': time,
                    'temperature': self.temperature[t, iy, ix] + np.random.normal(0, noise_std)
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


def gaussian(coords, c, sigma=0.1):
    """
    Parameters
    ----------
    coords : (N, 2) array
    c1, c2 : (2,) array-like — centers of the two gaussians e.g. [0.3, 0.2]
    sigma  : float — width of both gaussians
    """
    c = np.array(c)
    g = np.exp(-np.sum((coords - c) ** 2, axis=-1) / (2 * sigma**2))
    
    return 100.0 * g


def sdf_box(p, center, half_extents):
    """
    SDF for an axis-aligned box.

    Parameters
    ----------
    p : (N, 2) array of points to evaluate
    center : (2,) array-like — center of the box e.g. [0.0, 0.0]
    half_extents : (2,) array-like — half the width and height of the box e.g. [0.5, 0.5]
    """
    q = np.abs(p - center) - half_extents
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)


def sdf_t_block(p, bar_width, bar_height, stem_width, stem_height):
    """
    T-shape as union of two boxes.
    Origin is at the center of the full bounding box.

        ┌─────────────┐  ↑
        │   top bar   │  bar_height
        └────┬───┬────┘  ↓
             │ s │  ↑
             │ t │  stem_height
             │ e │
             │ m │  ↓
             └───┘
    """
    bar_center = np.array([0.0, stem_height / 2])
    stem_center = np.array([0.0, -bar_height / 2])

    overlap = 0.05
    bar = sdf_box(p, bar_center, np.array([bar_width / 2, bar_height / 2]))
    stem = sdf_box(
        p, 
        stem_center + np.array([0.0, overlap / 2]), 
        np.array([stem_width / 2, stem_height / 2 + overlap / 2])
    )

    return np.minimum(bar, stem)


if __name__ == "__main__":
    size = 2.0
    t_max = 5.0
    alpha = 0.01
    spatial_step = 0.01
    nt = 18751
    sigma = size / 4

    initial_temp = lambda coords: gaussian(
        coords, c=[0.5, 0.5], sigma=sigma
    )
    SDF = lambda coords: sdf_t_block(
        coords, bar_width=1.5, bar_height=0.385, stem_width=0.385, stem_height=1.0
    )

    thermocouple_locations = [
        (0, 0),
        (0, 0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0, -0.5),
    ]

    plate = Plate(
        size=size,
        SDF=SDF,
        initial_temp=initial_temp,
        t_max=t_max,
        alpha=alpha,
        spatial_step=spatial_step,
        nt=nt,
    )
    plate.run()
    plate.export_sparse(thermocouple_locations, 'training_data/T_plate.csv', step=6)
    plate.visualize(sensors=thermocouple_locations)
