import pytest
from esn import utils
import numpy as np


def test_edges():
    assert utils.edges({0:[1,2], 1:[0]}) == {(0,1), (0,2), (1,0)}
