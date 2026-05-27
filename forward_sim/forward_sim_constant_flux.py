import numpy as np
from tqdm import tqdm

class Plate():
    def __init__(self, initial_temperature, size, t_max, alpha, q, spatial_step, nt):
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
        q : float
            The constant heat flux applied to the plate (units of temperature per unit time).
        spatial_step : float
            The target spatial step size (distance between grid points).
        nt : int
            The number of time steps for the simulation.
        """
        self.initial_temperature = initial_temperature
        self.size = size
        self.t_max = t_max
        self.alpha = alpha
        self.q = q
        # Calculate nx from spatial_step
        self.nx = int(size / spatial_step) + 1
        self.dx = size / (self.nx - 1)

        # Calculate dt from nt
        self.dt = t_max / (nt - 1)
        self.nt = nt

        # Check numerical stability: Fo = alpha * dt / dx^2 must be <= 0.25
        Fo = alpha * self.dt / (self.dx**2)
        if Fo > 0.25:
            raise ValueError(
                f"Fourier number {Fo:.6f} exceeds stability limit of 0.25. "
                f"To fix: reduce nt (currently {nt}), increase spatial_step (currently {spatial_step}), "
                f"or increase alpha (currently {alpha})."
            )

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
        k_thermal : float
            Thermal conductivity (used to convert flux to gradient)
        """
        T = self.temperature[t]
        T_new = np.copy(T)
        
        grad = -self.q / self.alpha  # dT/dn = -q/alpha

        # Interior points (unchanged)
        d2T_dx2 = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dx**2)        
        d2T_dy2 = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dx**2)
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.alpha * self.dt * (d2T_dx2 + d2T_dy2)
        
        # Boundaries using ghost cells
        # Left boundary (x = -size/2)
        T_ghost_left = T[1:-1, 0] + grad * self.dx
        d2T_dx2_left = (T[1:-1, 1] - 2 * T[1:-1, 0] + T_ghost_left) / (self.dx**2)
        d2T_dy2_left = (T[2:, 0] - 2 * T[1:-1, 0] + T[:-2, 0]) / (self.dx**2)
        T_new[1:-1, 0] = T[1:-1, 0] + self.alpha * self.dt * (d2T_dx2_left + d2T_dy2_left)
        
        # Right boundary (x = size/2)
        T_ghost_right = T[1:-1, -1] + grad * self.dx
        d2T_dx2_right = (T_ghost_right - 2 * T[1:-1, -1] + T[1:-1, -2]) / (self.dx**2)
        d2T_dy2_right = (T[2:, -1] - 2 * T[1:-1, -1] + T[:-2, -1]) / (self.dx**2)
        T_new[1:-1, -1] = T[1:-1, -1] + self.alpha * self.dt * (d2T_dx2_right + d2T_dy2_right)

        # Top boundary (y = size/2)
        T_ghost_top = T[0, 1:-1] + grad * self.dx
        d2T_dx2_top = (T[0, 2:] - 2 * T[0, 1:-1] + T[0, :-2]) / (self.dx**2)
        d2T_dy2_top = (T[1, 1:-1] - 2 * T[0, 1:-1] + T_ghost_top) / (self.dx**2)
        T_new[0, 1:-1] = T[0, 1:-1] + self.alpha * self.dt * (d2T_dx2_top + d2T_dy2_top)

        # Bottom boundary (y = -size/2)
        T_ghost_bottom = T[-1, 1:-1] + grad * self.dx
        d2T_dx2_bottom = (T[-1, 2:] - 2 * T[-1, 1:-1] + T[-1, :-2]) / (self.dx**2)
        d2T_dy2_bottom = (T_ghost_bottom - 2 * T[-1, 1:-1] + T[-2, 1:-1]) / (self.dx**2)
        T_new[-1, 1:-1] = T[-1, 1:-1] + self.alpha * self.dt * (d2T_dx2_bottom + d2T_dy2_bottom)

        # Corners (average of adjacent boundaries)
        T_new[0, 0] = (T_new[0, 1] + T_new[1, 0]) / 2
        T_new[0, -1] = (T_new[0, -2] + T_new[1, -1]) / 2
        T_new[-1, 0] = (T_new[-2, 0] + T_new[-1, 1]) / 2
        T_new[-1, -1] = (T_new[-2, -1] + T_new[-1, -2]) / 2
        
        self.temperature[t + 1] = T_new


    def run(self):
        """
        Run the forward simulation for the specified number of time steps with a progress bar.
        """
        for t in tqdm(range(self.nt - 1), desc="Simulating heat propagation"):
            self.step(t)


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


    def export_random(self, filename, num_points=1000, noise_std=0.0):
        """
        Export the temperature data at randomly sampled points and times to a CSV file.

        Parameters
        ----------
        filename : str
            The name of the CSV file to which the data will be exported.
        num_points : int, optional
            The number of random points to sample (default is 1000).
        noise_std : float, optional
            The standard deviation of the Gaussian noise to add to the temperature data (default is 0.0).
        """
        import pandas as pd

        data = []
        for _ in range(num_points):
            ix = np.random.randint(0, self.nx)
            iy = np.random.randint(0, self.nx)
            it = np.random.randint(0, self.nt)

            x = -self.size / 2 + ix * self.dx
            y = -self.size / 2 + iy * self.dx
            time = it * self.dt
            temperature = self.temperature[it, iy, ix] + np.random.normal(0, noise_std)

            data.append({
                'sensor_id': 0,
                'x': x,
                'y': y,
                'time': time,
                'temperature': temperature
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


    def export_large(self, filename,xy_step=1, t_step=1):
        """
        Export the full temperature data for all grid points and time steps to a CSV file.
        Parameters
        ----------
        filename : str
            The name of the CSV file to which the data will be exported.
        xy_step : int, optional
            The step size for spatial sampling (default is 1, which means no downsampling).
        t_step : int, optional
            The step size for time sampling (default is 1, which means no downsampling).
        """
        import pandas as pd
        
        data = []
        for t in range(0, self.nt, t_step):
            time = t * self.dt
            for ix in range(0, self.nx, xy_step):
                x = -self.size / 2 + ix * self.dx
                for iy in range(0, self.nx, xy_step):
                    y = -self.size / 2 + iy * self.dx
                    data.append({
                        'sensor_id': 0,
                        'x': x,
                        'y': y,
                        'time': time,
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

        # Create coordinate grids for contour plot
        x = np.linspace(-self.size/2, self.size/2, self.nx)
        y = np.linspace(-self.size/2, self.size/2, self.nx)
        X, Y = np.meshgrid(x, y)

        # Plot initial frame to set up the colorbar and axes
        contour = ax.contourf(X, Y, self.temperature[0], levels=20, cmap='hot', vmin=self.temperature.min(), vmax=self.temperature.max())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        cbar = plt.colorbar(contour, ax=ax)

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
            ax.clear()
            contour = ax.contourf(X, Y, self.temperature[frame], levels=20, cmap='hot', vmin=self.temperature.min(), vmax=self.temperature.max())
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.set_title(f'Temperature at t={val:.4f}')
            # Plot points if provided
            if points is not None:
                px, py = zip(*points)
                ax.scatter(px, py, color='cyan', marker='x', label='Sensors')
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


def square_step_initial_temperature(size, nxy):
    """
    Initial temperature distribution: a square step function in the center of the plate.

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
    temperature = np.zeros((nxy, nxy), dtype=float)
    half_square_size = size / 4
    for i in range(nxy):
        for j in range(nxy):
            x = -size/2 + i * (size / (nxy - 1))
            y = -size/2 + j * (size / (nxy - 1))
            if abs(x) < half_square_size and abs(y) < half_square_size:
                temperature[j, i] = 1.0  # Note: j is the row index, i is the column index
    return temperature


if __name__ == "__main__":
    size = 2.0
    t_max = 5.0
    alpha = 0.01
    q = 0.0
    spatial_step = 0.02 # nx = 101
    nt = 18751  # w/ step=6, gives dt = 0.0016
    noise_std = 0.01

    thermocouple_locations = [
        (0,0),
        (-.9, -.9),
        (-.9, 0),
        (-.9, .9),
        (0, -.9),
        (0, .9),
        (.9, -.9),
        (.9, 0),
        (.9, .9)
    ]

    plate = Plate(dual_gaussian_initial_temperature, size, t_max, alpha, q, spatial_step, nt)
    plate.run()
    plate.export_sparse(points=thermocouple_locations, filename="training_data/insulated_DG.csv", step=6, noise_std=noise_std)
    plate.export_random(filename="training_data/insulated_DG_val.csv", num_points=1000)

    plate.animate(points=thermocouple_locations)