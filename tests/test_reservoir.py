import pytest
from esn import Data, Reservoir, utils
import numpy as np


def test_reservoir_integration():
    """Integration test, minimal case, mostly with defaults."""
    data = Data.create_source('lorenz')
    x,y = data.generate(1000)     # Intentionally overfitting, so very short training
    model = Reservoir(100, l2=0)  # ... and no regularization
    points_to_skip = 200
    model.fit(x, y, skip=points_to_skip)
    z = model.predict(x)
    loss = utils.loss(y[points_to_skip:], z[points_to_skip:])
    assert loss > 0
    assert loss < 1  # A typical value with these settings is 1e-4 (overfitted)


def test_reservoir_predict():
    """Testing the part that predicts """
    data = Data.create_source('lorenz')
    x, y = data.generate(100)
    model = Reservoir(100)
    model.fit(x, y)
    z = model.predict(x)  # Default case: use x as input predict full length of y
    assert len(z) == len(x)
    z = model.predict(x, 50)  # Truncate early (only generate a few)
    assert len(z) == 50

    model.fit(x, x)  # Auto-generator mode (continuing the signal)
    z = model.predict(x, 500)  # Load x, then generate its continuation for 400 time points
    assert len(z) == 500