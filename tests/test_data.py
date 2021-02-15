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

def test_lorenz_seed():
    data = Data.Lorenz()
    x1, y1 = data.generate(10, 0.0, integration_step=0.01) # Scalar seed
    x2, y2 = data.generate(10, 0.1, integration_step=0.01)
    assert not (x1 == x2).all() # Different seeds should give different values

def test_data_integration():
    data = Data.Lorenz()
    x1, y1 = data.generate(10, 0.0, integration_step=0.01)  # Scalar seed
    x2, y2 = data.generate(10, 0.0, integration_step=0.02)
    assert np.allclose(y1[8], y2[4], atol=0.01)

def test_data_sampling():
    data = Data.Lorenz()
    _, y1 = data.generate(16, 0.0, sampling_step=0.02)  # Scalar seed
    _, y2 = data.generate(8, 0.0, sampling_step=0.04)
    _, y3 = data.generate(4, 0.0, sampling_step=0.08)
    _, y4 = data.generate(2, 0.0, sampling_step=0.16)
    assert np.allclose(y1[-1], y2[-1], atol=0.01)
    assert np.allclose(y2[-1], y3[-1], atol=0.01)
    assert np.allclose(y3[-1], y4[-1], atol=0.01)

