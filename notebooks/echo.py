"""Technical file to circumvent importing rules for Jupyter notebooks.
It allows to import 'esn' in notebooks using:
from echo import esn"""

import sys
sys.path.append('..')
import esn