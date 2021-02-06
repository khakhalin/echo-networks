import pytest
from esn import Data, utils
import numpy as np


def test_lorenz_zero_seed():
    params = (10, 8 / 3, 0)
    data = Data.Lorenz(params=params)
    seed = (0, 0, 0)
    x, y = data.generate(10, seed, integration_step=0.01)
    expected = np.zeros(10)
    assert isinstance(x, (np.ndarray, np.generic))
    assert isinstance(y, (np.ndarray, np.generic))
    assert x.size == 10
    assert y.size == 10
    assert expected == pytest.approx(x, rel=1e-3)
    assert expected == pytest.approx(y, rel=1e-3)

def test_lorenz_zero_attract():
    params = (10, 8 / 3, -1)
    data = Data.Lorenz(params=params)
    seed = (0, 1, 1.05)
    x, y = data.generate(10, seed, integration_step=0.01)
    expected_x = np.array([0.1000, 0.18900, 0.26791, 0.33757, 0.39877,
                           0.45225, 0.49867, 0.53868, 0.57285, 0.60171])
    expected_y = np.array([1.02200, 0.99574, 0.97103, 0.94772, 0.92566,
                           0.90469, 0.88471, 0.86560, 0.84726, 0.82960])
    assert isinstance(x, (np.ndarray, np.generic))
    assert isinstance(y, (np.ndarray, np.generic))
    assert x.size == 10
    assert y.size == 10
    assert np.allclose(expected_x, x)
    assert expected_y == pytest.approx(y, rel=1e-4)
    assert True

    ## test graphics in chaotic stage
    #def test_plot_lorenz_chaotic(self):
    #    params = (10, 8 / 3, 28)
    #    data = Data.Lorenz(params=params)
    #    seed = (0, 1, 1.05)
    #    x, y = data.generate(1000, seed, integration_step=0.01)
    #    fig = plot_data(x,y,title='Lorenz')
    #    fig.show()
    #    return fig
