# -*- coding: utf-8 -*-
"""
PyStem module.

Description here.
"""
# flake8: noqa F401

import logging

from .signals import *
from .dataset import *
from .dev import *

from . import version
from . import restore
from . import tools
from . import dev

# set up logging to file
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename='{}.log'.format(__name__),
#                     filemode='w')

# # Define a Handler which writes INFO messages or higher to the sys.stderr
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# # tell the handler to use this format
# console.setFormatter(formatter)
# # add the handler to the root logger
# logging.getLogger(__name__).addHandler(console)