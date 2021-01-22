
from .reservoir import *
from .data import *
from .utils import *

"""
Target usage (the spirit of it):

model = esn.reservoir(n_nodes=100, network_type='ws', inhibition='spread')
x,y = esn.data.lorenz(t=100, start=0.0)
out_weights = model.fit(x, y)
prediction = model.predict(esn.data.lorenz(t=100, start=0.5))
print(prediction.quality)
"""