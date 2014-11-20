# -*- coding: utf-8 -*-

"""A Python package for image fingerprinting and recognition via artificial
neural networks.

Provides scripts for training and classification. This package uses the
`imgpheno <https://github.com/naturalis/imgpheno>`_ package for image feature
extraction.
"""

from .config import conf, ANN_DEFAULTS
from .functions import open_config

__version__ = '0.1.0'
