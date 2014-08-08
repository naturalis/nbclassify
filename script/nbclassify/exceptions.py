#!/usr/bin/env python
# -*- coding: utf-8 -*-

class FileExistsError(Exception):
    """Raised when a new file is going to be created, but the file already
    exists.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)
