# -*- coding: utf-8 -*-

"""Global configurations."""

# Default FANN training settings. These can be overwritten with the
# configurations file.
ANN_DEFAULTS = {
    'train_type': 'ordinary',
    'epochs': 100000,
    'desired_error': 0.00001,
    'training_algorithm': 'TRAIN_RPROP',
    'activation_function_hidden': 'SIGMOID_STEPWISE',
    'activation_function_output': 'SIGMOID_STEPWISE',
    'hidden_layers': 1,
    'hidden_neurons': 8,
    'learning_rate': 0.7,
    'connection_rate': 1,
    'max_neurons': 20,
    'neurons_between_reports': 1,
    'cascade_activation_steepnesses': [0.25, 0.50, 0.75, 1.00],
    'cascade_num_candidate_groups': 2,
}

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

    def __init__(self):
        # Switch to True whilst debugging. It is automatically set to True when
        # the environment variable PYTHONDEBUG is set. Setting this to True
        # prevents some exceptions from being caught by the scripts.
        self.debug = False

        # Force overwrite of files. If set to False, existing files are not
        # touched. When set to True, any existing files are overwritten without
        # warning.
        self.force_overwrite = False

        # File name of the meta data file. This file is created for an image
        # directory with the `meta` subcommand for the nbc-trainer script. This
        # file is needed for training and classification.
        self.meta_file = ".meta.db"

        # Prefix for output columns in training data. It is used for exporting
        # and reading training data.
        self.output_prefix = "OUT:"

        # If set to a positive integer, only the classifications (i.e. genus,
        # section, species combination) with a photo count of at least
        # `photo_count_min` are used to build the taxon hierarchy.
        self.photo_count_min = 0

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None

# Create configurations singleton.
conf = ConfigManager()
