import numpy as np
from tqdm import tqdm

class Plate():
    def __init__(self, initial_temperature, size, t_max, alpha, spatial_step, Fo=0.2):
        """
        Initialize a a 2D plate for forward simulation of the heat equation.

        Parameters
        ----------
        initial_temperature : function
            A function of the form initial_temperature(size, nx) that returns a 2D array of shape (nx, nx) representing the initial temperature distribution on the plate.
        size : float
            The physical size of the plate (length of one side).
        t_max : float
            The maximum time for the simulation.
        alpha : float
            The thermal diffusivity of the material.
        spatial_step : float
            The target spatial step size (distance between grid points).
        Fo : float, optional
            The desired Fourier number (default is 0.2).
            <= 0.25 is required for numerical stability
        """
        self.initial_temperature = initial_temperature
        self.size = size
        self.t_max = t_max
        self.alpha = alpha
        
        # Calculate nx from spatial_step
        self.nx = int(size / spatial_step) + 1
        self.dx = size / (self.nx - 1)

        # Calculate dt such that Fo = 0.2
        self.dt = Fo * self.dx**2 / alpha
        
        # Calculate nt from dt
        self.nt = int(t_max / self.dt) + 1

        self.temperature = np.zeros((self.nt, self.nx, self.nx), dtype=float)
        self.temperature[0] = self.initial_temperature(size, self.nx)

    
    def step(self, t):
        """
        Perform a single time step of the forward simulation using the FTCS
        finite difference method. Vectorized for performance.

        Parameters
        ----------
        t : int
            The current time step index (starting from 0).
        """
        T = self.temperature[t]
        T_new = np.copy(T)

        # d^2T / dx^2  (Left and Right neighbors)
        d2T_dx2 = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dx**2)        
        # d^2T / dy^2  (Top and Bottom neighbors)
        d2T_dy2 = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dx**2)
        # Apply the explicit update equation to all interior points at once
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.alpha * self.dt * (d2T_dx2 + d2T_dy2)
        # Enforce zero temperature at the boundaries (Dirichlet boundary conditions)
        T_new[0, :] = 0
        T_new[-1, :] = 0
        T_new[:, 0] = 0
        T_new[:, -1] = 0
        
        self.temperature[t + 1] = T_new


    def run(self):
        """
        Run the forward simulation for the specified number of time steps with a progress bar.
        """
        for t in tqdm(range(self.nt - 1), desc="Simulating heat propagation"):
            self.step(t)


    def export_sparse(self, points, filename, step=1):
        """
        Export the temperature data at specified points for all time steps to a CSV file.
        Parameters
        ----------
        points : list of tuples
            A list of (x, y) coordinates where the temperature data should be extracted.
        filename : str
            The name of the CSV file to which the data will be exported.
        step : int, optional
            Export every nth time step (default is 1, meaning all time steps).
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
                    'time': time,
                    'x': x,
                    'y': y,
                    'temperature': self.temperature[t, iy, ix]
                })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


    def animate(self, points=None):
        """
        Display an interactive heat map with a time slider.

        Parameters
        ----------
        points : list of tuples, optional
            A list of (x, y) coordinates to display as markers on the plate.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Make room for the slider
        
        # Plot initial frame to set up the colorbar and axes
        extent = [-self.size/2, self.size/2, -self.size/2, self.size/2]
        im = ax.imshow(self.temperature[0], cmap='hot', vmin=self.temperature.min(), vmax=self.temperature.max(), extent=extent, origin='lower')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

        # Plot points if provided
        if points is not None:
            px, py = zip(*points)
            ax.scatter(px, py, color='cyan', marker='x', label='Sensors')
        
        # Add a slider for time control
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        max_time = (self.nt - 1) * self.dt
        slider = Slider(ax_slider, '', 0, max_time, valinit=0, valstep=self.dt)
        ax_slider.set_xlabel('Time')
        
        def update_plot(val):
            # The slider is the "source of truth" for the current frame
            frame = int(round(val / self.dt))
            im.set_data(self.temperature[frame])
            ax.set_title(f'Temperature at t={val:.4f}')
            fig.canvas.draw_idle()

        slider.on_changed(update_plot)
        
        # Keep reference to prevent garbage collection
        self._slider = slider
        self._ax_slider = ax_slider
        
        plt.show(block=True)


def gaussian_initial_temperature(size, nxy):
    """
    Example initial temperature distribution: a Gaussian centered in the middle of the plate.

    Parameters
    ----------
    size : float
        The physical size of the plate (length of one side).
    nxy : int
        The number of grid points in each spatial dimension.

    Returns
    -------
    np.ndarray
        A 2D array of shape (nxy, nxy) representing the initial temperature distribution on the plate.
    """
    x = np.linspace(-size/2, size/2, nxy)
    y = np.linspace(-size/2, size/2, nxy)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian parameters
    sigma = size / 5
    center_x = 0
    center_y = 0
    
    return np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))


def dual_gaussian_initial_temperature(size, nxy):
    """
    Initial temperature distribution: sum of two Gaussians with different centers and standard deviations.

    Parameters
    ----------
    size : float
        The physical size of the plate (length of one side).
    nxy : int
        The number of grid points in each spatial dimension.

    Returns
    -------
    np.ndarray
        A 2D array of shape (nxy, nxy) representing the initial temperature distribution on the plate.
    """
    x = np.linspace(-size/2, size/2, nxy)
    y = np.linspace(-size/2, size/2, nxy)
    X, Y = np.meshgrid(x, y)
    
    # First Gaussian
    sigma1 = size / 6
    center_x1 = -size / 6
    center_y1 = -size / 6
    gaussian1 = np.exp(-((X - center_x1)**2 + (Y - center_y1)**2) / (2 * sigma1**2))
    
    # Second Gaussian
    sigma2 = size / 8
    center_x2 = size / 6
    center_y2 = size / 6
    gaussian2 = np.exp(-((X - center_x2)**2 + (Y - center_y2)**2) / (2 * sigma2**2))
    
    return gaussian1 + gaussian2


if __name__ == "__main__":
    size = 2.0
    t_max = 5.0
    alpha = 0.3
    spatial_step = 0.02

    thermocouple_locations = [
        (-0.5, -0.25),
        (0.45, 0.35),
        (0.25, -0.45),
        (0,0),
        (-0.75, 0.75),
        (0.75, -0.75),
        (0.75, 0.75),
        (-0.75, -0.75)
    ]

    plate = Plate(dual_gaussian_initial_temperature, size, t_max, alpha, spatial_step)
    plate.run()
    plate.export_sparse(thermocouple_locations, 'training_data/dual_gaussian.csv', step=6)
    plate.animate(points=thermocouple_locations)