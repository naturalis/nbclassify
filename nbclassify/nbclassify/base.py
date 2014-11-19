# -*- coding: utf-8 -*-

"""Base classes."""

from . import conf
from .functions import Struct

class Common(object):

    """Base class with common methods."""

    def __init__(self, config):
        """Set the configurations object `config`."""
        self.set_config(config)

    def set_config(self, config):
        """Set the configurations object `config`.

        Expects a configuration object as returned by
        :meth:`functions.open_config`.
        """
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, " \
                "not %s" % type(config))
        self.config = config

    def set_photo_count_min(self, count):
        """Set a minimum for photos count per photo classification.

        If `count` is a positive integer, only the classifications (i.e. genus,
        section, species combination) with a photo count of at least `count` are
        used to build the taxon hierarchy.

        This setting influences the output of :meth:`db.get_taxon_hierarchy`.
        """
        if not isinstance(count, int):
            raise TypeError("Value must be an integer")
        conf.photo_count_min = int(count)

    def get_photo_count_min(self):
        """Return the minimum for photos count per species."""
        return int(conf.photo_count_min)
