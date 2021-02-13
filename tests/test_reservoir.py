import pytest
from esn import Data, Reservoir
import numpy as np


def test_reservoir_integration():
    """Integration test, minimal case of actual reservoir use."""
    data = Data.create_source('lorenz')
    x,y = data.generate(1000)
    model = Reservoir(100)
    points_to_skip = 200
    model.fit(x, y, skip=points_to_skip)
    z = model.predict(x)
    loss = Data.loss(y[points_to_skip:], z[points_to_skip:])
    assert loss > 0
    assert loss < 1  # This one is a bit risky, but a typical value with these settings is 1e-4


def test_reservoir_predict():
    """Testing the part that predicts """
    data = Data.create_source('lorenz')
    x, y = data.generate(100)
    model = Reservoir(100)
    model.fit(x, y)
    z = model.predict(x)  # Default case, predict full x length
    assert len(z) == len(x)
    z = model.predict(x, 50)  # Truncate earlier
    assert len(z) == 50
    model.fit(x, x)            # Technically not necessary for this test, but that's how it needs to be used
    z = model.predict(x, 500)  # Load x, then generate some more
    assert len(z) == 500