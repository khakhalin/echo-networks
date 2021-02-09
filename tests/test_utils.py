import pytest
from esn import utils
import numpy as np


def test_loss():
    assert np.allclose(utils.loss([0, 0], [2, 1]) , 5/2)
    assert np.allclose(utils.loss([0, 0], np.array([1, 1])), 1)


def test_edges():
    assert utils.edges({0:[1,2], 1:[0]}) == {(0,1), (0,2), (1,0)}
