# -*- coding: utf-8 -*-

"""ANN training routines."""

import logging
import os
import re
import subprocess
import sys

from pyfann import libfann
import yaml

from . import conf, ANN_DEFAULTS
from .base import Common, Struct
from .data import TrainData
from .exceptions import *
from .functions import (get_codewords, get_classification,
    classification_hierarchy_filters, readable_filter)
import nbclassify.db as db

class TrainANN(object):

    """Train artificial neural networks."""

    def __init__(self):
        """Set the default attributes."""
        self.ann = None
        self.train_data = None
        self.test_data = None

        # Set the default training settings.
        for option, val in ANN_DEFAULTS.iteritems():
            setattr(self, option, val)
        self.iterations_between_reports = self.epochs / 100

    def set_train_data(self, data):
        """Set the training data on which to train the network on.

        The training data `data` must be an instance of
        :class:`~nbclassify.data.TrainData`.
        """
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        if not data:
            raise ValueError("Train data is empty")
        self.train_data = data

    def train(self, data):
        """Train a neural network on training data `data`.

        Returns a FANN structure.
        """
        self.set_train_data(data)

        # Check some values.
        if not self.train_type in ('ordinary','cascade'):
            raise ValueError("Unknown training type `%s`" % self.train_type)
        if self.train_type == 'cascade':
            if not self.training_algorithm in ('TRAIN_RPROP','TRAIN_QUICKPROP'):
                raise ValueError("Expected TRAIN_RPROP or TRAIN_QUICKPROP "\
                    "as the training algorithm")

        # Get FANN train data object.
        fann_train_data = libfann.training_data()
        fann_train_data.set_train_data(self.train_data.get_input(),
            self.train_data.get_output())

        if self.train_type == 'ordinary':
            hidden_layers = [self.hidden_neurons] * self.hidden_layers
            layers = [self.train_data.num_input]
            layers.extend(hidden_layers)
            layers.append(self.train_data.num_output)

            self.ann = libfann.neural_net()
            self.ann.create_sparse_array(self.connection_rate, layers)

            # Set training parameters.
            self.ann.set_learning_rate(self.learning_rate)
            self.ann.set_activation_function_hidden(
                getattr(libfann, self.activation_function_hidden))
            self.ann.set_activation_function_output(
                getattr(libfann, self.activation_function_output))
            self.ann.set_training_algorithm(
                getattr(libfann, self.training_algorithm))

            sys.stderr.write("Ordinary training...\n")
            self.ann.print_parameters()

            # Ordinary training.
            self.ann.train_on_data(fann_train_data, self.epochs,
                self.iterations_between_reports, self.desired_error)

        if self.train_type == 'cascade':
            # This algorithm adds neurons to the neural network while training,
            # which means that it needs to start with an ANN without any hidden
            # layers.
            layers = [self.train_data.num_input, self.train_data.num_output]
            self.ann = libfann.neural_net()
            self.ann.create_shortcut_array(layers)

            # Set training parameters.
            self.ann.set_training_algorithm(
                getattr(libfann, self.training_algorithm))
            self.ann.set_activation_function_hidden(
                getattr(libfann, self.activation_function_hidden))
            self.ann.set_activation_function_output(
                getattr(libfann, self.activation_function_output))
            self.ann.set_cascade_activation_steepnesses(
                self.cascade_activation_steepnesses)
            self.ann.set_cascade_num_candidate_groups(
                self.cascade_num_candidate_groups)

            sys.stderr.write("Cascade training...\n")
            self.ann.print_parameters()

            # Cascade training.
            self.ann.cascadetrain_on_data(fann_train_data, self.max_neurons,
                self.neurons_between_reports, self.desired_error)

        return self.ann

    def test(self, data):
        """Test the trained neural network.

        Expects an instance of :class:`~nbclassify.data.TrainData`. Returns the
        mean square error on the test data `data`.
        """
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        if not self.ann:
            raise ValueError("No neural network was trained yet")
        if data.num_input != self.train_data.num_input:
            raise ValueError("Number of inputs of test data must be same as " \
                "train data")
        if data.num_output != self.train_data.num_output:
            raise ValueError("Number of output of test data must be same as " \
                "train data")

        fann_test_data = libfann.training_data()
        fann_test_data.set_train_data(data.get_input(), data.get_output())

        self.ann.reset_MSE()
        self.ann.test_data(fann_test_data)

        return self.ann.get_MSE()

class Aivolver(TrainANN):

    """Use Aivolver to train an artificial neural network."""

    def __init__(self):
        super(Aivolver, self).__init__()

    def train(self, data_path, config_path):
        """Train a neural network on training data from `data_path`.

        Returns a FANN structure.
        """
        # Load and update Aivolver configurations.
        with open(config_path, 'r') as fh:
            config = yaml.load(fh)

        if 'experiment' not in config:
            raise ValueError("Missing `experiment` configurations")
        if 'ann' not in config:
            raise ValueError("Missing `ann` configurations")

        config['ann']['error'] = self.desired_error
        config['ann']['epochs'] = self.epochs
        config['ann']['neurons'] = self.max_neurons

        sys.stderr.write("Training with Aivolver...\n")

        # Empty the working directory.
        workdir = config['experiment']['workdir']
        delete_temp_dir(workdir, recursive=True)
        os.makedirs(workdir)

        # Write new configurations file for Aivolver.
        new_config_path = os.path.join(workdir, 'config.yml')
        with open(new_config_path, 'w') as fh:
            yaml.dump(config, fh)

        # Execute Aivolver.
        ann_path = os.path.join(workdir, 'fittest.ann')
        subprocess.check_call([
            'aivolver',
            new_config_path,
            '-d', "file=%s" % data_path,
            '-o', ann_path,
        ])

        # Load the neural network from the temporary directory.
        self.ann = libfann.neural_net()
        self.ann.create_from_file(ann_path)

        return self.ann

class MakeAnn(Common):

    """Train an artificial neural network."""

    def __init__(self, config):
        """Set the configurations object `config`."""
        super(MakeAnn, self).__init__(config)
        self._train_method = 'default'
        self._aivolver_config_path = None

    def set_training_method(self, method, *args):
        methods = ('default', 'aivolver')
        if method not in methods:
            raise ValueError("Unknown method `%s`. Expected one of %s" % \
                method, methods)
        if method == 'aivolver':
            try:
                self._aivolver_config_path = str(args[0])
            except:
                raise ValueError("The Aivolver configuration path must be" \
                    "passed as the second argument")
        self._train_method = method

    def train(self, train_file, ann_file, config=None):
        """Train an artificial neural network.

        Loads training data from a TSV file `train_file`, trains a neural
        network `ann_file` with training paramerters ``ann`` from the
        configurations. If training parameters are provided with `config`,
        those are used instead.
        """
        if not os.path.isfile(train_file):
            raise IOError("Cannot open %s (no such file)" % train_file)
        if not conf.force_overwrite and os.path.isfile(ann_file):
            raise FileExistsError(ann_file)
        if config and not isinstance(config, Struct):
            raise TypeError("Expected an nbclassify.Struct instance for `config`")

        # Instantiate the ANN trainer.
        if self._train_method == 'default':
            trainer = TrainANN()
        if self._train_method == 'aivolver':
            trainer = Aivolver()

        # Set the training parameters.
        if not config:
            try:
                config = self.config.ann
            except:
                pass

        for option, value in ANN_DEFAULTS.iteritems():
            if config:
                value = getattr(config, option, value)
            setattr(trainer, option, value)

        trainer.iterations_between_reports = trainer.epochs / 100

        # Train the ANN.
        if self._train_method == 'default':
            try:
                dependent_prefix = self.config.data.dependent_prefix
            except:
                dependent_prefix = OUTPUT_PREFIX

            train_data = TrainData()
            train_data.read_from_file(train_file, dependent_prefix)

            ann = trainer.train(train_data)

        if self._train_method == 'aivolver':
            ann = trainer.train(train_file, self._aivolver_config_path)

        # Save the neural network to disk.
        ann.save(str(ann_file))
        logging.info("Artificial neural network saved to %s" % ann_file)

class BatchMakeAnn(MakeAnn):

    """Generate training data.

    Must be used within a database session scope.
    """

    def __init__(self, config):
        """Constructor for training data generator.

        Expects a configurations object `config`, a path to the root directory
        of the photos, and a path to the database file `meta_path` containing
        photo meta data.
        """
        super(BatchMakeAnn, self).__init__(config)

        self.taxon_hr = None

        # Get the taxonomic hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise ConfigurationError("classification hierarchy not set")

    def _load_taxon_hierarchy(self):
        """Load the taxon hierarchy.

        Must be separate from the constructor because
        :meth:`set_photo_count_min` influences the taxon hierarchy.
        """
        session, metadata = db.get_session_or_error()

        if not self.taxon_hr:
            self.taxon_hr = db.get_taxon_hierarchy(session, metadata)

    def batch_train(self, data_dir, output_dir):
        """Batch train neural networks.

        Training data is obtained from the directory `data_dir` and the
        neural networks are saved to the directory `output_dir`. Which training
        data to train on is set in the classification hierarchy of the
        configurations.
        """
        session, metadata = db.get_session_or_error()

        # Must not be loaded in the constructor, in case set_photo_count_min()
        # is used.
        self._load_taxon_hierarchy()

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Train an ANN for each path in the classification hierarchy.
        for filter_ in classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_.get('class'))
            train_file = os.path.join(data_dir, self.class_hr[level].train_file)
            ann_file = os.path.join(output_dir, self.class_hr[level].ann_file)
            if 'ann' not in self.class_hr[level]:
                config = None
            else:
                config = self.class_hr[level].ann

            # Replace any placeholders in the paths.
            where = filter_.get('where', {})
            for key, val in where.items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Get the classification categories from the database.
            classes = db.get_classes_from_filter(session, metadata, filter_)
            assert len(classes) > 0, \
                "No classes found for filter `%s`" % filter_

            # Skip train data export if there is only one class for this filter.
            if not len(classes) > 1:
                logging.debug("Only one class for this filter. Skipping " \
                    "training of %s" % ann_file)
                continue

            # Train the ANN.
            logging.info("Training network `%s` with training data " \
                "from `%s` ..." % (ann_file, train_file))
            try:
                self.train(train_file, ann_file, config)
            except FileExistsError as e:
                # Don't train if the file already exists.
                logging.warning("Skipping: %s" % e)

class TestAnn(Common):

    """Test an artificial neural network.

    Must be used within a database session scope.
    """

    def __init__(self, config):
        super(TestAnn, self).__init__(config)
        self.test_data = None
        self.ann = None
        self.re_photo_id = re.compile(r'([0-9]+)')
        self.classifications = {}
        self.classifications_expected = {}
        self.class_hr = None
        self.taxon_hr = None

    def test(self, ann_file, test_file):
        """Test an artificial neural network."""
        if not os.path.isfile(ann_file):
            raise IOError("Cannot open %s (no such file)" % ann_file)
        if not os.path.isfile(test_file):
            raise IOError("Cannot open %s (no such file)" % test_file)

        # Get the prefix for the classification columns.
        try:
            dependent_prefix = self.config.data.dependent_prefix
        except:
            dependent_prefix = OUTPUT_PREFIX

        self.ann = libfann.neural_net()
        self.ann.create_from_file(ann_file)

        self.test_data = TrainData()
        try:
            self.test_data.read_from_file(test_file, dependent_prefix)
        except IOError as e:
            logging.error("Failed to process the test data: %s" % e)
            exit(1)

        logging.info("Testing the neural network...")
        fann_test_data = libfann.training_data()
        fann_test_data.set_train_data(self.test_data.get_input(),
            self.test_data.get_output())

        self.ann.test_data(fann_test_data)

        mse = self.ann.get_MSE()
        logging.info("Mean Square Error on test data: %f" % mse)

    def export_results(self, filename, filter_, error=0.01):
        """Export the classification results to a TSV file.

        Export the test results to a tab separated file `filename`. The class
        name for a codeword is obtained from the database `db_path`, using the
        classification filter `filter_`. A bit in a codeword is considered on
        if the mean square error for a bit is less or equal to `error`.
        """
        session, metadata = db.get_session_or_error()

        if self.test_data is None:
            raise RuntimeError("Test data is not set")

        # Get the classification categories from the database.
        classes = db.get_classes_from_filter(session, metadata, filter_)
        assert len(classes) > 0, \
            "No classes found for filter `%s`" % filter_

        # Get the codeword for each class.
        codewords = get_codewords(classes)

        # Write results to file.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(['ID','Class','Classification','Match']) )

            total = 0
            correct = 0
            for label, input, output in self.test_data:
                total += 1
                row = []

                if label:
                    row.append(label)
                else:
                    row.append("")

                if len(codewords) != len(output):
                    raise ValueError("Codeword length ({0}) does not " \
                        "match output length ({1}). Is the classification " \
                        "filter correct?".format(len(codewords), len(output))
                    )

                class_expected = get_classification(codewords, output, error)
                class_expected = [class_ for mse,class_ in class_expected]
                assert len(class_expected) == 1, \
                    "The codeword for a class can only have one positive value"
                row.append(class_expected[0])

                codeword = self.ann.run(input)
                class_ann = get_classification(codewords, codeword, error)
                class_ann = [class_ for mse,class_ in class_ann]

                row.append(", ".join(class_ann))

                # Assume a match if the first items of the classifications match.
                if len(class_ann) > 0 and class_ann[0] == class_expected[0]:
                    row.append("+")
                    correct += 1
                else:
                    row.append("-")

                fh.write( "%s\n" % "\t".join(row) )

            # Calculate fraction correctly classified.
            fraction = float(correct) / total

            # Write correctly classified fraction.
            fh.write( "%s\n" % "\t".join(['','','',"%.3f" % fraction]) )

        print "Correctly classified: %.1f%%" % (fraction*100)
        print "Testing results written to %s" % filename

    def test_with_hierarchy(self, test_data_dir, ann_dir, max_error=0.001):
        """Test each ANN in a classification hierarchy and export results.

        Returns a 2-tuple ``(correct,total)``.
        """
        session, metadata = db.get_session_or_error()

        logging.info("Testing the neural networks hierarchy...")

        self.classifications = {}
        self.classifications_expected = {}

        # Get the taxonomic hierarchy from the database.
        self.taxon_hr = db.get_taxon_hierarchy(session, metadata)

        # Get the classification hierarchy from the configurations.
        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise ConfigurationError("classification hierarchy not set")

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Get the prefix for the classification columns.
        try:
            dependent_prefix = self.config.data.dependent_prefix
        except:
            dependent_prefix = OUTPUT_PREFIX

        # Get the expected and recognized classification for each sample in
        # the test data.
        for filter_ in classification_hierarchy_filters(levels, self.taxon_hr):
            logging.info("Classifying on %s" % readable_filter(filter_))

            level_name = filter_.get('class')
            level_n = levels.index(level_name)
            level = self.class_hr[level_n]
            test_file = os.path.join(test_data_dir, level.test_file)
            ann_file = os.path.join(ann_dir, level.ann_file)

            # Set the maximum error for classification.
            try:
                max_error = level.max_error
            except:
                pass

            # Replace any placeholders in the paths.
            where = filter_.get('where', {})
            for key, val in where.items():
                val = val if val is not None else '_'
                test_file = test_file.replace("__%s__" % key, val)
                ann_file = ann_file.replace("__%s__" % key, val)

            # Get the class names for this filter.
            classes = db.get_classes_from_filter(session, metadata, filter_)
            assert len(classes) > 0, \
                "No classes found for filter `%s`" % filter_

            # Get the codeword for each class.
            codewords = get_codewords(classes)

            # Load the ANN.
            if len(classes) > 1:
                ann = libfann.neural_net()
                ann.create_from_file(str(ann_file))

            # Load the test data.
            test_data = TrainData()
            test_data.read_from_file(test_file, dependent_prefix)

            # Test each sample in the test data.
            for label, input_, output in test_data:
                assert len(codewords) == len(output), \
                    "Codeword size mismatch. Codeword has {0} bits, but the " \
                    "training data has {1} output bits.".\
                    format(len(codewords), len(output))

                # Obtain the photo ID from the label.
                if not label:
                    raise ValueError("Test sample is missing a label with " \
                        "photo ID")

                try:
                    photo_id = self.re_photo_id.search(label).group(1)
                    photo_id = int(photo_id)
                except:
                    raise RuntimeError("Failed to obtain the photo ID from " \
                        "the sample label")

                # Skip classification if there is only one class for this
                # filter.
                if not len(classes) > 1:
                    logging.debug("Not enough classes for filter. Skipping " \
                        "testing of %s" % ann_file)

                    self.classifications[photo_id][level_name] = ['']
                    self.classifications_expected[photo_id][level_name] = ['']
                    continue

                # Set the expected class.
                class_expected = get_classification(codewords, output,
                    max_error)
                class_expected = [class_ for mse,class_ in class_expected]

                assert len(class_expected) == 1, \
                    "Class codewords must have one positive bit, found {0}".\
                    format(len(class_expected))

                # Get the recognized class.
                codeword = ann.run(input_)
                class_ann = get_classification(codewords, codeword,
                    max_error)
                class_ann = [class_ for mse,class_ in class_ann]

                # Save the classification at each level.
                if level_n == 0:
                    self.classifications[photo_id] = {}
                    self.classifications_expected[photo_id] = {}

                self.classifications[photo_id][level_name] = class_ann
                self.classifications_expected[photo_id][level_name] = class_expected

            ann.destroy()

        return self.get_correct_count()

    def export_hierarchy_results(self, filename):
        """Export classification results generated by :meth:`test_with_hierarchy`."""
        if not self.classifications or not self.classifications_expected:
            raise RuntimeError("Classifications not set")

        total = 0
        correct = 0

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Export the results.
        with open(filename, 'w') as fh:
            # Write the header.
            row = ['ID']
            row.extend(["%s (exp.)" % level for level in levels])
            row.extend(["%s (class.)" % level for level in levels])
            row.append('Match')
            fh.write( "%s\n" % "\t".join(row) )

            # Write the results.
            for photo_id, class_exp in self.classifications_expected.iteritems():
                # Check if we have a match.
                match = True
                for level in levels:
                    if level not in class_exp:
                        continue
                    expected = class_exp[level][0]
                    if expected not in self.classifications[photo_id][level]:
                        match = False
                        break

                # Construct the results row.
                row = [str(photo_id)]

                # Expected classification.
                for level in levels:
                    class_ = class_exp.get(level)
                    class_ = '' if class_ is None else class_[0]
                    row.append(class_)

                # Accuired classification.
                for level in levels:
                    class_ = self.classifications[photo_id].get(level, [])
                    class_ = ", ".join(class_)
                    row.append(class_)

                # Match or not.
                if match:
                    row.append("+")
                    correct += 1
                else:
                    row.append("-")

                fh.write("%s\n" % "\t".join(row))

                total += 1

        return (correct, total)

    def get_correct_count(self, level_filter=None):
        """Return number of correctly classified samples.

        The classifications are checked on the level names in the list
        `level_filter`. If `level_filter` is None, all levels are checked. A
        classification is considered correct if the classifications at all
        levels are correct. If the classification for a level returns multiple
        values, the level is considered correct if the expected value is one
        of the members.

        Returns a dictinary where the keys correspond to the level name, and
        "all" for the overall score. Each value is a 2-tuples
        ``(correct,total)``.
        """
        if not self.classifications or not self.classifications_expected:
            raise RuntimeError("Classifications not set")

        total = len(self.classifications_expected)
        correct = 0
        levels = [l.name for l in self.class_hr]

        # Count the number of correct classifications.
        for photo_id, class_exp in self.classifications_expected.iteritems():
            match = True
            for level in levels:
                if level_filter and level not in level_filter:
                    continue

                try:
                    expected = class_exp[level][0]
                except:
                    continue

                if expected not in self.classifications[photo_id][level]:
                    match = False
                    break

            if match:
                correct += 1

        return (correct, total)
