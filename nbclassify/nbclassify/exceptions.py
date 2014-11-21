# -*- coding: utf-8 -*-

"""Custom exceptions that may be raised by this package."""

class NBCException(Exception):
    """Base class for NBClassify exceptions."""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class FileExistsError(NBCException):
    """Raised when a new file is going to be created, but the file already
    exists.
    """
    pass

class ConfigurationError(NBCException):
    """Raised when there is a configuration related error."""
    pass

class DatabaseSessionError(NBCException):
    """Raised when there is a database session related error."""
    pass
