import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib.colors as colors
import matplotlib

matplotlib.use('TkAgg')  # macOS-friendly backend

class Plate:
    def __init__(self, size, SDF, initial_temp, t_max, alpha, q, spatial_step, nt):
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
        self.q = q

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
        self.boundary_mask = (-2*self.dx < SDF_values) & (SDF_values <= 0.0)
        self.interior_mask = self.shape_mask & ~self.boundary_mask

        # Precompute normalized SDF gradients for BC enforcement
        sdf_grad_x = (self.SDF(self.coords + np.array([self.dx, 0])) - self.SDF(self.coords - np.array([self.dx, 0]))) / (2 * self.dx)
        sdf_grad_y = (self.SDF(self.coords + np.array([0, self.dx])) - self.SDF(self.coords - np.array([0, self.dx]))) / (2 * self.dx)
        grad_norm = np.sqrt(sdf_grad_x**2 + sdf_grad_y**2) + 1e-8  # Avoid division by zero
        self.sdf_grad_x = sdf_grad_x / grad_norm 
        self.sdf_grad_y = sdf_grad_y / grad_norm

        # Precompute flux value and interior neighbor coordinates for Neumann BC
        self.flux = q / alpha
        boundary_indices = np.where(self.boundary_mask)
        # For each boundary point, compute coords one step inward (opposite of normal)
        boundary_x = self.coords[self.boundary_mask, 0]
        boundary_y = self.coords[self.boundary_mask, 1]
        inward_normal_x = -self.sdf_grad_x[self.boundary_mask]
        inward_normal_y = -self.sdf_grad_y[self.boundary_mask]
        self.interior_neighbor_coords = np.stack([
            boundary_x + self.dx * inward_normal_x,
            boundary_y + self.dx * inward_normal_y
        ], axis=1)

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

        # Compute Laplacian with ghost cells where neighbors would go outside
        # X-direction: check if rolled neighbors are outside domain
        T_right = np.roll(T, -1, axis=1)  # values from (i, j+1)
        neighbor_right_mask = np.roll(self.shape_mask, -1, axis=1)  # is (i, j+1) inside?
        # Where neighbor is outside, use ghost cell
        T_right[~neighbor_right_mask] = T[~neighbor_right_mask] - self.flux * self.dx * self.sdf_grad_x[~neighbor_right_mask]
        
        T_left = np.roll(T, 1, axis=1)   # values from (i, j-1)
        neighbor_left_mask = np.roll(self.shape_mask, 1, axis=1)  # is (i, j-1) inside?
        T_left[~neighbor_left_mask] = T[~neighbor_left_mask] + self.flux * self.dx * self.sdf_grad_x[~neighbor_left_mask]
        
        d2T_dx2 = (T_right - 2 * T + T_left) / self.dx**2
        
        # Y-direction: check if rolled neighbors are outside domain
        T_down = np.roll(T, -1, axis=0)  # values from (i+1, j)
        neighbor_down_mask = np.roll(self.shape_mask, -1, axis=0)
        T_down[~neighbor_down_mask] = T[~neighbor_down_mask] - self.flux * self.dx * self.sdf_grad_y[~neighbor_down_mask]
        
        T_up = np.roll(T, 1, axis=0)     # values from (i-1, j)
        neighbor_up_mask = np.roll(self.shape_mask, 1, axis=0)
        T_up[~neighbor_up_mask] = T[~neighbor_up_mask] + self.flux * self.dx * self.sdf_grad_y[~neighbor_up_mask]
        
        d2T_dy2 = (T_down - 2 * T + T_up) / self.dx**2

        # Update all points in shape_mask
        T_new[self.shape_mask] += (
            self.alpha
            * self.dt
            * (d2T_dx2[self.shape_mask] + d2T_dy2[self.shape_mask])
        )

        # T=0 outside boundary
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
        Visualizes the temperature distribution at a given time step using a contour plot.
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
        # Create coordinate grids for contour plot
        x_coords = np.linspace(-self.size/2, self.size/2, self.nx)
        y_coords = np.linspace(-self.size/2, self.size/2, self.nx)
        X, Y = np.meshgrid(x_coords, y_coords)

        vmin = 0.0
        vmax = 100.0
        norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

        if step is not None:
            # Single frame
            fig, ax = plt.subplots(figsize=figsize)
            frame = self.temperature[step].copy().astype(float)
            frame[~self.shape_mask] = np.nan
            
            contour = ax.contourf(X, Y, frame, levels=30, cmap=cmap_obj, norm=norm)
            plt.colorbar(contour, ax=ax, label='Temperature')
            t_val = step * self.dt
            ax.set_title(f't = {t_val:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            
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
            
            frame = self.temperature[0].copy().astype(float)
            frame[~self.shape_mask] = np.nan
            contour = ax.contourf(X, Y, frame, levels=30, cmap=cmap_obj, norm=norm)
            cbar = plt.colorbar(contour, ax=ax, label='Temperature')
            title = ax.set_title('t = 0.0000')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            
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
                frame = self.temperature[t].copy().astype(float)
                frame[~self.shape_mask] = np.nan
                
                ax.clear()
                contour = ax.contourf(X, Y, frame, levels=30, cmap=cmap_obj, norm=norm)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_aspect('equal')
                title.set_text(f't = {t * self.dt:.4f}')
                ax.set_title(title.get_text())
                if sensors is not None:
                    ax.scatter(sx, sy, color='cyan', marker='x', s=50, label='Thermocouples')
                    ax.legend(loc='upper right')
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


def sdf_box(p, top_left, bottom_right):
    """
    SDF for an axis-aligned box.

    Parameters
    ----------
    p : (N, 2) array of points to evaluate
    top_left : (2,) array-like — coordinates of the top-left corner of the box
    bottom_right : (2,) array-like — coordinates of the bottom-right corner of the box
    """
    top_left = np.array(top_left)
    bottom_right = np.array(bottom_right)
    
    center = (top_left + bottom_right) / 2
    half_size = np.abs(bottom_right - top_left) / 2  # Use absolute value

    d = np.abs(p - center) - half_size
    outside_dist = np.linalg.norm(np.maximum(d, 0), axis=-1)
    inside_dist = -np.minimum(np.abs(d[..., 0]), np.abs(d[..., 1]))

    inside = np.all(d <= 0, axis=-1)
    return np.where(inside, inside_dist, outside_dist)


def sdf_circle(p, center, radius):
    """
    SDF for a circle.

    Parameters
    ----------
    p : (N, 2) array of points to evaluate
    center : (2,) array-like — coordinates of the circle center
    radius : float — radius of the circle
    """
    center = np.array(center)
    return np.linalg.norm(p - center, axis=-1) - radius


def sdf_union(p, sdf1, sdf2):
    """
    SDF of the union of two SDFs.
    Parameters
    ----------
    p : (N, 2) array of points to evaluate
    sdf1, sdf2 : functions that take (N, 2) array and return (N,) array of signed distances
    """
    return np.minimum(sdf1(p), sdf2(p))


if __name__ == "__main__":
    size = 2.0
    t_max = 5.0
    alpha = 0.01
    q = 0.5
    spatial_step = 0.01
    nt = 18751
    sigma = size / 6

    initial_temp = lambda coords: gaussian(
        coords, c=[-0.25, 0.5], sigma=sigma
    )

    rect_sdf = lambda p: sdf_box(p, top_left=[-0.75, 0.75], bottom_right=[-0.25, -0.75])
    circ_sdf = lambda p: sdf_circle(p, center=[0.0, 0.375], radius=0.5)
    SDF = lambda p: sdf_union(p, rect_sdf, circ_sdf)

    thermocouple_locations = [
        (0, 0.375),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.0, 0.0),
        (-0.5, 0.0),
    ]

    plate = Plate(
        size=size,
        SDF=SDF,
        initial_temp=initial_temp,
        t_max=t_max,
        alpha=alpha,
        q=q,
        spatial_step=spatial_step,
        nt=nt,
    )
    plate.run()
    plate.export_sparse(thermocouple_locations, 'training_data/SDF_CF.csv', step=6)
    print(f"avg. initial temp: {plate.temperature[0].mean()}")
    print(f"avg. final temp: {plate.temperature[-1].mean()}")
    plate.visualize(sensors=thermocouple_locations)
