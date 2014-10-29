#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Global configurations."""

class ConfigManager(object):
    """Manage global configurations.

    An instance of this class provides access to a set of variables that need to
    be accessible across modules. By importing this module, one instance of this
    class is created. Subsequent imports in other modules provides access to
    that same instance.

    Configurations are set as attributes of an instance of this class. Getting
    an attribute that does not exist returns None, so this never raises an
    AttributeError.
    """
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None

# Create configurations singleton.
conf = ConfigManager()
