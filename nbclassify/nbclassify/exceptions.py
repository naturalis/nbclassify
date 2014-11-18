# -*- coding: utf-8 -*-

"""Custom exceptions that may be raised by this package."""

class FileExistsError(Exception):
    """Raised when a new file is going to be created, but the file already
    exists.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class ConfigurationError(Exception):
    """Raised when there is a configuration related error."""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)
