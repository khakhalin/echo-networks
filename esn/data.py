import numpy as np

class Data():
    @classmethod
    def create_source(cls, process='lorenz', params=None, fileName=None):
        """
        Factory method that creates custom chaotic time series generators.

        Parameters:
        process: string, type of the dynamical system (optional, default lorenz)
        params: parameters for ths system (optional)
        fileName: string, if time series needs to be loaded from a file (optional)
        """
        if process=='lorenz':
            return cls.Lorenz(params)
        raise ValueError('Chaotic process name not recognized.')


    class _DataSource(object):
        """Abstract class for data generators"""
        # abstractmethod
        def _run(self, n_points, seed, integration_step):
            """Returns 2 numpy arrays: x and y."""
            pass

        # decorator
        def generate(self, n_points, seed=None, integration_step=0.01, sampling_step=None):
            """Decorator that runs the model, then downsamples it."""
            if not sampling_step: # No need to downsample
                _, xy = self._run(n_points, seed, integration_step)
                return xy[:,0], xy[:,1]
            full_n = round(n_points * sampling_step / integration_step * 1.1)  # With some excess just in case
            if sampling_step < integration_step:
                raise ValueError('Integration step should be <= sampling_step')
            time,xy = self._run(full_n, seed, integration_step)
            # Now downsample
            ind = np.floor(time / sampling_step) # Steps
            ind = np.hstack(([0], ind[1:] - ind[:-1])).astype(bool) # Where steps change
            return xy[ind, 0][:n_points], xy[ind, 1][:n_points]     # Actual downsampling


    class Lorenz(_DataSource):
        """Lorenz system."""
        def __init__(self, params=None):
            if not params:
                params = (10, 8/3, 28) # Sigma, beta, rho
            self.sigma, self.beta, self.rho = params

        def _run(self, n_points=100, seed=None, integration_step=0.01):
            """Lorenz system, with manual resampling"""
            if not seed:
                seed = (1, 0, 1)
            if len(seed) != 3:
                seed = (0, 0, seed[0])
            (x, y, z) = seed
            history = np.zeros((n_points, 3))
            time = 0
            for i in range(n_points):
                x, y, z = (x + integration_step * self.sigma * (y - x),
                           y + integration_step * (x * (self.rho - z) - y),
                           z + integration_step * (x * y - self.beta * z))
                time += integration_step
                history[i, :] = (time, x, z)
            return (history[:, 0], history[:, 1:]) # time, then x and z together