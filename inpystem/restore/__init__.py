# -*- coding: utf-8 -*-
# flake8: noqa F401
"""
This package implements a variety of reconstruction algorithms for 2D
as 3D data.

These methods include:

* interpolation methods that are known to be fast but with low-quality
  results,
* regularized least-square methods which are a slower than interpolation
  but with higher quality,
* dictionary-learning methods which are very efficient at the price of
  long reconstruction procedures.


"""


# 2D / 3D methods.
from . import interpolation
from . import DL_ITKrMM
from . import DL_ITKrMM_matlab
from . import DL_BPFA

# 2D methods.
from . import LS_2D

# 3D methods.
from . import LS_3D
from . import LS_CLS
