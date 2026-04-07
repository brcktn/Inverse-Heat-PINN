import numpy as np
from tqdm import tqdm


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
        x_coords = np.linspace(-self.size/2, self.size/2, self.nx)
        xx, yy = np.meshgrid(x_coords, x_coords)
        self.coords = np.stack([xx, yy], axis=-1)

        # Precompute SDF values for the grid
        SDF_values = self.SDF(self.coords)
        self.shape_mask = SDF_values < self.dx # iterior + boundary
        self.interior_mask = SDF_values < -self.dx # iterior only
        self.boundary_mask = (np.abs(SDF_values) <= self.dx) & ~self.interior_mask

        # Initialize temperature array
        self.temperature = np.zeros((self.nt, self.nx, self.nx))
        self._initialize_temperature()


    def _initialize_temperature(self):
        """
        Initializes the temperature distribution based on the
        initial temperature function and the geometry defined by the SDF.
        """
        self.temperature[0][self.shape_mask] = self.initial_temp(self.coords[self.shape_mask])


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
        d2T_dx2 = (np.roll(T, -1, axis=1) - 2*T + np.roll(T, 1, axis=1)) / self.dx**2
        d2T_dy2 = (np.roll(T, -1, axis=0) - 2*T + np.roll(T, 1, axis=0)) / self.dx**2

        # Update interior points
        T_new[self.interior_mask] += self.alpha * self.dt * (d2T_dx2[self.interior_mask] + d2T_dy2[self.interior_mask])

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