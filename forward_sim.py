class Plate():
    def __init__(self, initial_temperature, size, t_max, alpha, nxy, dt):
        """
        Initialize a a 2D plate for forward simulation of the heat equation.

        Parameters
        ----------
        initial_temperature : function
            A function of the form initial_temperature(size, nxy) that returns a 2D array of shape (nxy, nxy) representing the initial temperature distribution on the plate.
        size : float
            The physical size of the plate (length of one side).
        t_max : float
            The maximum time for the simulation.
        alpha : float
            The thermal diffusivity of the material.
        nxy : int
            The number of grid points in each spatial dimension (the plate will be discretized into an nxy x nxy grid).
        dt : float
            The time step for the simulation.
        """
        self.initial_temperature = initial_temperature(size, nxy)
        self.size = size
        self.t_max = t_max
        self.alpha = alpha
        self.nxy = nxy
        self.dt = dt
        self.nt = int(t_max / dt)
        self.dxy = size / (nxy - 1)

