# -*- coding: utf-8 -*-

"""Base classes."""

from nbclassify.config import conf
from nbclassify.functions import Struct

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

        This setting is used by :meth:`db.get_taxon_hierarchy`, and indirectly by
        :meth:`get_classes_from_filter` (there exists an equivalent of this
        method in the :mod:`nbclassify.db` module, but that function does not
        filter by this criterion).
        """
        if not isinstance(count, int):
            raise TypeError("Value must be an integer")
        conf.photo_count_min = int(count)

    def get_photo_count_min(self):
        """Return the minimum for photos count per species."""
        return int(conf.photo_count_min)

    def readable_filter(self, filter_):
        """Return a human-readable description of a classification filter.

        Classification filters are those as returned by
        :meth:`classification_hierarchy_filters`.

        Example:

            >>> cmn.readable_filter({'where': {'section': 'Lorifolia',
            ... 'genus': 'Phragmipedium'}, 'class': 'species'})
            'species where section is Lorifolia and genus is Phragmipedium'
        """
        class_ = filter_['class']
        where = filter_.get('where', {})
        where_n = len(where)
        where_s = ""
        for i, (k,v) in enumerate(where.items()):
            if i > 0 and i < where_n - 1:
                where_s += ", "
            elif where_n > 1 and i == where_n - 1:
                where_s += " and "
            where_s += "%s is %s" % (k,v)

        if where_n > 0:
            return "%s where %s" % (class_, where_s)
        return "%s" % class_
