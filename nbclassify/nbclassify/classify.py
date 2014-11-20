# -*- coding: utf-8 -*-

"""Methods for image classification using artificial neural networks."""

import hashlib
import logging
import os
import sys

from pyfann import libfann

from .base import Common
from .data import Phenotyper
from .db import session_scope, get_taxon_hierarchy
from .exceptions import *
from .functions import (combined_hash, get_childs_from_hierarchy,
    get_classification, get_codewords, get_config_hashables)

class ImageClassifier(Common):

    """Classify an image."""

    def __init__(self, config, meta_path):
        super(ImageClassifier, self).__init__(config)
        self.error = 0.0001
        self.cache = {}
        self.roi = None
        self.set_meta_path(meta_path)

        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise ConfigurationError("classification hierarchy not set")

        # Get the taxon hierarchy from the database.
        with session_scope(meta_path) as (session, metadata):
            self.taxon_hr = get_taxon_hierarchy(session, metadata)

    def set_meta_path(self, path):
        """Set the path to the meta file."""
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.meta_path = path

    def set_error(self, error):
        """Set the default maximum error for classification."""
        if not 0 < error < 1:
            raise ValueError("Error must be a value between 0 and 1" % error)
        self.error = error

    def set_roi(self, roi):
        """Set the region of interest for the image.

        If a region of interest is set, only that region is used for image
        processing. The ROI must be a ``(y,y2,x,x2)`` coordinates tuple.
        """
        if roi is None:
            self.roi = roi
            return
        if roi and len(roi) != 4:
            raise ValueError("ROI must be a list of four integers")
        for x in roi:
            if not isinstance(x, int):
                raise TypeError("Found a non-integer in the ROI")
        y, y2, x, x2 = roi
        if not y < y2 or not x < x2:
            raise ValueError("ROI must be a (y,y2,x,x2) coordinates tuple")
        self.roi = roi

    def get_classification_hierarchy_levels(self):
        """Return the list of level names from the classification hierarchy."""
        return [l.name for l in self.class_hr]

    def classify_image(self, im_path, ann_path, config):
        """Classify an image file and return the codeword.

        Preprocess and extract features from the image `im_path` as defined
        in the configuration object `config`, and use the features as input
        for the neural network `ann_path` to obtain a codeword.
        """
        if not os.path.isfile(im_path):
            raise IOError("Cannot open %s (no such file)" % im_path)
        if not os.path.isfile(ann_path):
            raise IOError("Cannot open %s (no such file)" % ann_path)
        if 'preprocess' not in config:
            raise ConfigurationError("preprocess settings not set")
        if 'features' not in config:
            raise ConfigurationError("features settings not set")

        ann = libfann.neural_net()
        ann.create_from_file(str(ann_path))

        # Get the MD5 hash for the image.
        hasher = hashlib.md5()
        with open(im_path, 'rb') as fh:
            buf = fh.read()
            hasher.update(buf)

        # Get a hash that that is unique for this image/preprocess/features
        # combination.
        hashables = get_config_hashables(config)
        hash_ = combined_hash(hasher.hexdigest(),
            config.features, *hashables)

        if hash_ in self.cache:
            phenotype = self.cache[hash_]
        else:
            phenotyper = Phenotyper()
            phenotyper.set_image(im_path)
            if self.roi:
                y, y2, x, x2 = self.roi
                phenotyper.set_grabcut_roi((x, y, x2-x, y2-y))
            phenotyper.set_config(config)
            phenotype = phenotyper.make()

            # Cache the phenotypes, in case they are needed again.
            self.cache[hash_] = phenotype

        logging.debug("Using ANN `%s`" % ann_path)
        codeword = ann.run(phenotype)

        return codeword

    def classify_with_hierarchy(self, image_path, ann_base_path=".",
                                path=[], path_error=[]):
        """Start recursive classification.

        Classify the image `image_path` with neural networks from the
        directory `ann_base_path`. The image is classified for each level
        in the classification hierarchy ``classification.hierarchy`` set in
        the configurations file. Each level can use a different neural
        network for classification; the file names for the neural networks
        are set in ``classification.hierarchy[n].ann_file``. Multiple
        classifications are returned if the classification of a level in
        the hierarchy returns multiple classifications, in which case the
        classification path is split into multiple classifications paths.

        Returns a pair of tuples ``(classifications, errors)``, the list
        of classifications, and the list of errors for each classification.
        Each classification is a list of the classes for each level in the
        hierarchy, top to bottom. The list of errors has the same dimension
        of the list of classifications, where each value corresponds to the
        mean square error of each classification.
        """
        levels = self.get_classification_hierarchy_levels()
        paths = []
        paths_errors = []

        if len(path) == len(levels):
            return ([path], [path_error])
        elif len(path) > len(levels):
            raise ValueError("Classification hierarchy depth exceeded")

        # Get the level specific configurations.
        level = conf = self.class_hr[len(path)]

        # Replace any placeholders in the ANN path.
        ann_file = level.ann_file
        for key, val in zip(levels, path):
            val = val if val is not None else '_'
            ann_file = ann_file.replace("__%s__" % key, val)

        # Get the class names for this node in the taxonomic hierarchy.
        level_classes = get_childs_from_hierarchy(self.taxon_hr, path)

        # Some levels must have classes set.
        if level_classes == [None] and level.name in ('genus','species'):
            raise ValueError("Classes for level `%s` are not set" % level.name)

        if level_classes == [None]:
            # No need to classify if there are no classes for current level.
            classes = level_classes
            class_errors = [0.0]
        elif len(level_classes) == 1:
            # Also no need to classify if there is only one class.
            classes = level_classes
            class_errors = [0.0]
        else:
            # Get the codewords for the classes.
            class_codewords = get_codewords(level_classes)

            # Classify the image and obtain the codeword.
            ann_path = os.path.join(ann_base_path, ann_file)
            codeword = self.classify_image(image_path, ann_path, conf)

            # Set the maximum classification error for this level.
            try:
                max_error = level.max_error
            except:
                max_error = self.error

            # Get the class name associated with this codeword.
            classes = get_classification(class_codewords,
                codeword, max_error)
            if classes:
                class_errors, classes = zip(*classes)
            else:
                class_errors = classes = []

        # Print some info messages.
        path_s = '/'.join([str(p) for p in path])

        # Return the classification if classification failed on current level.
        if len(classes) == 0:
            logging.debug("Failed to classify on level `%s` at node `/%s`" % (
                level.name,
                path_s)
            )
            return ([path], [path_error])
        elif len(classes) > 1:
            logging.debug("Branching in level `%s` at node '/%s' into `%s`" % (
                level.name,
                path_s,
                ', '.join(classes))
            )
        else:
            logging.debug("Level `%s` at node `/%s` classified as `%s`" % (
                level.name,
                path_s,
                classes[0])
            )

        for class_, mse in zip(classes, class_errors):
            # Recurse into lower hierarchy levels.
            paths_, paths_errors_ = self.classify_with_hierarchy(image_path,
                ann_base_path, path+[class_], path_error+[mse])

            # Keep a list of each classification path and their
            # corresponding errors.
            paths.extend(paths_)
            paths_errors.extend(paths_errors_)

        assert len(paths) == len(paths_errors), \
            "Number of paths must be equal to the number of path errors"

        return paths, paths_errors
