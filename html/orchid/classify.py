#!/usr/bin/env python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import hashlib
import os

from pyfann import libfann
import sqlalchemy
import yaml

import imgpheno as ft
import nbclassify as nbc
import nbclassify.db as db

class ImageClassifier(nbc.Common):
    """Classify an image."""

    def __init__(self, config, db_path):
        super(ImageClassifier, self).__init__(config)

        self.set_db_path(db_path)
        self.error = 0.0001
        self.cache = {}
        self.roi = None

        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise nbc.ConfigurationError("missing `classification.hierarchy`")

        # Get the classification hierarchy from the database.
        with db.session_scope(db_path) as (session, metadata):
            self.taxon_hr = self.get_taxon_hierarchy(session, metadata)

    def set_db_path(self, path):
        """Set the path to the database file."""
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.db_path = path

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
                raise ValueError("Found a non-integer in the ROI")
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
            raise nbc.ConfigurationError("missing `preprocess`")
        if 'features' not in config:
            raise nbc.ConfigurationError("missing `features`")

        ann = libfann.neural_net()
        ann.create_from_file(str(ann_path))

        # Get the MD5 hash.
        hasher = hashlib.md5()
        with open(im_path, 'rb') as fh:
            buf = fh.read()
            hasher.update(buf)

        # Create a hash of the feature extraction options.
        hash_ = "%s.%s.%s" % (hasher.hexdigest(), hash(config.preprocess),
            hash(config.features))

        if hash_ in self.cache:
            phenotype = self.cache[hash_]
        else:
            phenotyper = nbc.Phenotyper()
            phenotyper.set_image(im_path)
            if self.roi:
                y, y2, x, x2 = self.roi
                phenotyper.set_grabcut_roi((x, y, x2-x, y2-y))
            phenotyper.set_config(config)
            phenotype = phenotyper.make()

            # Cache the phenotypes, in case they are needed again.
            self.cache[hash_] = phenotype

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
        level_classes = self.get_childs_from_hierarchy(self.taxon_hr, path)

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
            class_codewords = self.get_codewords(level_classes)

            # Classify the image and obtain the codeword.
            ann_path = os.path.join(ann_base_path, ann_file)
            codeword = self.classify_image(image_path, ann_path, conf)

            # Set the maximum classification error for this level.
            try:
                max_error = level.max_error
            except:
                max_error = self.error

            # Get the class name associated with this codeword.
            classes = self.get_classification(class_codewords,
                codeword, max_error)
            if classes:
                class_errors, classes = zip(*classes)
            else:
                class_errors = classes = []

        # Return the classification if classification failed on current level.
        if len(classes) == 0:
            return ([path], [path_error])

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
