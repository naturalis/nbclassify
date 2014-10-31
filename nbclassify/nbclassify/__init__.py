#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common code for classification scripts.

This package contains commmon code used by the training and classification
scripts. This package depends on Naturalis' `features
<https://github.com/naturalis/feature-extraction>`_ package for feature
extraction from digital images.
"""

from argparse import Namespace
import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import imgpheno as ft
import numpy as np
from pyfann import libfann
from sqlalchemy import func
from sqlalchemy import orm
from sqlalchemy.ext.automap import automap_base
import yaml

import nbclassify.db as db
from nbclassify.exceptions import *

# Default FANN settings.
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

def delete_temp_dir(path, recursive=False):
    """Delete a temporary directory.

    As a safeguard, this function only removes directories and files that are
    within the system's temporary directory (e.g. /tmp on Unix). Setting
    `recursive` to True also deletes its contents.
    """
    path = os.path.abspath(str(path))
    if os.path.isdir(path):
        if path.startswith(tempfile.gettempdir()):
            if recursive:
                shutil.rmtree(path)
            else:
                os.rmdir(path)
        else:
            raise ValueError("Cannot delete non-temporary directories")

def open_config(path):
    """Read a configurations file and return as a nested :class:`Struct` object.

    The configurations file is in the YAML format and is loaded from file path
    `path`.
    """
    with open(path, 'r') as f:
        config = yaml.load(f)
    return Struct(config)

def combined_hash(*args):
    """Create a combined hash from one or more hashable objects.

    Each argument must be an hashable object. Can also be used for configuration
    objects as returned by :meth:`open_config`. Returned hash is a negative or
    positive integer.

    Example::

        >>> a = Struct({'a': True})
        >>> b = Struct({'b': False})
        >>> combined_hash(a,b)
        6862151379155462073
    """
    hash_ = None
    for obj in args:
        if hash_ is None:
            hash_ = hash(obj)
        else:
            hash_ ^= hash(obj)
    return hash_

def test_classification_filter(f):
    """Test the validity of the classification filter `f`.

    Raises a ValueError if the filter is not valid.
    """
    if isinstance(f, dict):
        f = Struct(f)
    if 'class' not in f:
        raise ValueError("Attribute `class` not set")
    for key in vars(f):
        if key not in ('where', 'class'):
            raise ValueError("Unknown key `%s` in filter" % key)

class Struct(Namespace):

    """Return a dictionary as an object."""

    def __init__(self, d):
        if not isinstance(d, dict):
            raise TypeError("Expected a dictionary, got {0} instead".\
                format(type(d)))
        for key, val in d.iteritems():
            if isinstance(val, (list, tuple)):
                setattr(self, str(key), [self.__class__(x) if \
                    isinstance(x, dict) else x for x in val])
            elif isinstance(val, self.__class__):
                setattr(self, str(key), val)
            else:
                setattr(self, str(key), self.__class__(val) if \
                    isinstance(val, dict) else val)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __getitem__(self, key):
        return getattr(self, str(key))

    def as_dict(self):
        d = vars(self)
        for key, val in d.iteritems():
            if isinstance(val, (list, tuple)):
                d[key] = [x.as_dict() if \
                    isinstance(x, self.__class__) else x for x in val]
            elif isinstance(val, self.__class__):
                d[key] = val.as_dict()
            else:
                d[key] = val
        return d

class Common(object):

    """Collection of common functions.

    This class is used as a super class in several scripts that implement this
    package. It provides commonly used functions.

    .. note::

       Methods of this class take into account the minimum photo count per
       class, when set with :meth:`set_photo_count_min`. When a script needs to
       take into account this value, always use methods from this class. If an
       outside function is used to obtain records from the metadata database,
       always make sure that the results are somehow filtered by
       :meth:`get_photo_count_min`.
    """

    def __init__(self, config):
        """Set the configurations object `config`."""
        self.set_config(config)
        self._photo_count_min = 0

    def set_config(self, config):
        """Set the configurations object `config`.

        Expects a configuration object as returned by :meth:`open_config`.
        """
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, " \
                "not %s" % type(config))

        try:
            path = config.preprocess.segmentation.output_folder
        except:
            path = None
        if path and not os.path.isdir(path):
            logging.error("Found a configuration error")
            raise IOError("Cannot open %s (no such directory)" % path)

        self.config = config

    def set_photo_count_min(self, count):
        """Set a minimum for photos count per photo classification.

        If `count` is a positive integer, only the classifications (i.e. genus,
        section, species combination) with a photo count of at least `count` are
        used to build the taxon hierarchy. Other methods of this class that
        process metadata records also filter by this criterion, if set. As a
        result, any data processing is done exclusively on photos of
        classifications with enough photos.

        This setting is used by :meth:`get_taxon_hierarchy`, and indirectly by
        :meth:`get_classes_from_filter` (there exists an equivalent of this
        method in the :mod:`nbclassify.db` module, but that function does not
        filter by this criterion).
        """
        if not count >= 0:
            raise ValueError("Value must be an integer of 0 or more")
        self._photo_count_min = int(count)

    def get_photo_count_min(self):
        """Return the minimum for photos count per species."""
        return self._photo_count_min

    def get_codewords(self, classes, on=1, off=-1):
        """Return codewords for a list of classes.

        Takes a list of class names `classes`. The class list is sorted, and a
        codeword is created for each class. Each codeword is a list of
        ``len(classes)`` bits, where all bits have an `off` value, except for
        one, which has an `on` value. Returns a dictionary ``{class: codeword,
        ..}``. The same set of classes always returns the same codeword for
        each class.
        """
        n = len(classes)
        codewords = {}
        for i, class_ in enumerate(sorted(classes)):
            cw = [off] * n
            cw[i] = on
            codewords[class_] = cw
        return codewords

    def get_classification(self, codewords, codeword, error=0.01, on=1.0):
        """Return the human-readable classification for a codeword.

        Each bit in the codeword `codeword` is compared to the `on` bit in
        each of the codewords in `codewords`, which is a dictionary of the
        format ``{class: codeword, ..}``. If the mean square error for a bit
        is less than or equal to `error`, then the corresponding class is
        assigned to the codeword. So it is possible that a codeword is
        assigned to multiple classes.

        The result is returned as an error-sorted list of 2-tuples ``[(error,
        class), ..]``. Returns an empty list if no classes were found.
        """
        if len(codewords) != len(codeword):
            raise ValueError("Codeword size mismatch. The number of " \
                "codeword bits does not match the number of classes. " \
                "The classes in the meta data file must match the classes " \
                "used to train the neural networks.")
        classes = []
        for class_, word in codewords.items():
            for i, bit in enumerate(word):
                if bit == on:
                    mse = (float(bit) - codeword[i]) ** 2
                    if mse <= error:
                        classes.append((mse, class_))
                    break
        return sorted(classes)

    def path_from_filter(self, filter_, levels):
        """Return the path from a classification filter."""
        path = []
        for name in levels:
            try:
                path.append(filter_['where'][name])
            except:
                return path
        return path

    def get_taxon_hierarchy(self, session, metadata):
        """Return the taxanomic hierarchy for photos in the metadata database.

        Requires access to a database via an SQLAlchemy Session `session` and
        MetaData object `metadata`.

        The hierarchy is returned as a dictionary in the format
        ``{genus: {section: [species, ..], ..}, ..}``.

        Returned hierarchies can be used as input for methods like
        :meth:`classification_hierarchy_filters` and
        :meth:`get_childs_from_hierarchy`.
        """
        hierarchy = {}

        for genus, section, species, count in db.get_taxa_photo_count(session, metadata):
            if self._photo_count_min > 0 and count < self._photo_count_min:
                continue

            if genus not in hierarchy:
                hierarchy[genus] = {}
            if section not in hierarchy[genus]:
                hierarchy[genus][section] = []
            hierarchy[genus][section].append(species)
        return hierarchy

    def get_classes_from_filter(self, session, metadata, filter_):
        """Return the classes for a classification filter.

        Requires access to a database via an SQLAlchemy Session `session` and
        MetaData object `metadata`.

        The unique set of classes for the classification filter `filter_` are
        returned. Filters are those as returned by
        :meth:`classification_hierarchy_filters`.
        """
        levels = ['genus','section','species']
        path = self.path_from_filter(filter_, levels)
        hr = self.get_taxon_hierarchy(session, metadata)
        classes = self.get_childs_from_hierarchy(hr, path)
        return set(classes)

    def classification_hierarchy_filters(self, levels, hr, path=[]):
        """Return the classification filter for each path in a hierarchy.

        Returns the classification filter for each possible path in the
        hierarchy `hr`. The name of each level in the hierarchy must be set in
        the list `levels`. The list `path` holds the position in the
        hierarchy. An empty `path` means start at the root of the hierarchy.
        Filters that return no classes are not returned.

        Example:

            >>> hr = {
            ...     'Selenipedium': {
            ...         None: ['palmifolium']
            ...     },
            ...     'Phragmipedium': {
            ...         'Micropetalum': ['andreettae', 'besseae'],
            ...         'Lorifolia': ['boissierianum', 'brasiliense']
            ...     }
            ... }
            >>> levels = ['genus', 'section', 'species']
            >>> filters = cmn.classification_hierarchy_filters(levels, hr)
            >>> for f in filters:
            ...     print f
            ...
            {'where': {}, 'class': 'genus'}
            {'where': {'section': None, 'genus': 'Selenipedium'}, 'class': 'species'}
            {'where': {'genus': 'Phragmipedium'}, 'class': 'section'}
            {'where': {'section': 'Lorifolia', 'genus': 'Phragmipedium'}, 'class': 'species'}
            {'where': {'section': 'Micropetalum', 'genus': 'Phragmipedium'}, 'class': 'species'}

        These filters are used directly by methods like
        :meth:`db.get_filtered_photos_with_taxon` and
        :meth:`db.get_classes_from_filter`.
        """
        filter_ = {}

        # The level number that is being classfied (0 based).
        level_no = len(path)

        if level_no > len(levels) - 1:
            raise ValueError("Maximum classification hierarchy depth exceeded")

        # Set the level to classify on.
        filter_['class'] = levels[level_no]

        # Set the where fields.
        filter_['where'] = {}
        for i, class_ in enumerate(path):
            name = levels[i]
            filter_['where'][name] = class_

        # Get the classes for the current hierarchy path.
        classes = self.get_childs_from_hierarchy(hr, path)

        # Only return the filter if the classes are set.
        if classes != [None]:
            yield filter_

        # Stop iteration if the last level was classified.
        if level_no == len(levels) - 1:
            return

        # Recurse into lower hierarchy levels.
        for c in classes:
            for f in self.classification_hierarchy_filters(levels, hr,
                    path+[c]):
                yield f

    def get_childs_from_hierarchy(self, hr, path=[]):
        """Return the child node names for a node in a hierarchy.

        Returns a list of child node names of the hierarchy `hr` at node with
        the path `path`. The hierarchy `hr` is a nested dictionary, as
        returned by :meth:`get_taxon_hierarchy`. Which node to get the childs
        from is specified by `path`, which is a list of the node names up to
        that node. An empty list for `path` means the names of the nodes of
        the top level are returned.

        Example:

            >>> hr = {
            ...     'Phragmipedium': {
            ...         'Micropetalum': ['andreettae', 'besseae'],
            ...         'Lorifolia': ['boissierianum', 'brasiliense']
            ...     },
            ...     'Selenipedium': {
            ...         None: ['palmifolium']
            ...     }
            ... }
            >>> cmn.get_childs_from_hierarchy(hr)
            ['Selenipedium', 'Phragmipedium']
            >>> cmn.get_childs_from_hierarchy(hr, ['Phragmipedium'])
            ['Lorifolia', 'Micropetalum']
            >>> cmn.get_childs_from_hierarchy(hr, ['Phragmipedium','Micropetalum'])
            ['andreettae', 'besseae']
        """
        nodes = hr.copy()
        try:
            for name in path:
                nodes = nodes[name]
        except:
            raise ValueError("No such path `%s` in the hierarchy" % \
                '/'.join(path))

        if isinstance(nodes, dict):
            names = nodes.keys()
        elif isinstance(nodes, list):
            names = nodes
        else:
            raise ValueError("Incorrect hierarchy format")

        return names

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


class Phenotyper(object):
    """Extract features from a digital image and return as a phenotype.

    Uses the :mod:`imgpheno` package to extract features from the image. Use
    :meth:`set_image` to load an image and :meth:`set_config` to set a
    configuration object as returned by :meth:`open_config`. Then :meth:`make`
    can be called to extract the features as specified in the configurations
    object and return the phenotype. A single phenotypes is returned, which is
    a list of floating point numbers.
    """

    def __init__(self):
        """Set the default attributes."""
        self.path = None
        self.config = None
        self.img = None
        self.mask = None
        self.bin_mask = None
        self.grabcut_roi = None

    def set_image(self, path, roi=None):
        """Load the image from path `path`.

        If a region of interest `roi` is set, only that region is used for
        image processing. The ROI must be a 4-tuple ``(y,y2,x,x2)``. Image
        related attributes are reset. Returns the image object.
        """
        self.img = cv2.imread(path)
        if self.img is None or self.img.size == 0:
            raise IOError("Failed to read image %s" % path)
        if roi and len(roi) != 4:
            raise ValueError("ROI must be a list of four integers")

        # If a ROI was provided, use only that region.
        if roi:
            y, y2, x, x2 = roi
            self.img = self.img[y:y2, x:x2]

        self.path = path
        self.config = None
        self.mask = None
        self.bin_mask = None

        return self.img

    def set_grabcut_roi(self, roi):
        """Set the region of interest for the GrabCut algorithm.

        If GrabCut is set as the segmentation algorithm, then GrabCut is
        executed with this region of interest.

        The ROI must be a 4-tuple ``(x,y,width,height)``.
        """
        if len(roi) != 4:
            raise ValueError("ROI must be a list of four integers")
        self.grabcut_roi = roi

    def set_config(self, config):
        """Set the configurations object.

        Expects a configuration object as returned by :meth:`open_config`.
        """
        if not isinstance(config, Struct):
            raise TypeError("Expected a Struct instance, got {0} instead".\
                format(type(config)))
        if not 'features' in config:
            raise ConfigurationError("Features to extract not set. Missing " \
                "the `features` setting.")
        self.config = config

    def __grabcut(self, img, iters=5, roi=None, margin=5):
        """Wrapper for OpenCV's grabCut function.

        Runs the GrabCut algorithm for segmentation. Returns an 8-bit
        single-channel mask. Its elements may have the following values:

        * ``cv2.GC_BGD`` defines an obvious background pixel
        * ``cv2.GC_FGD`` defines an obvious foreground pixel
        * ``cv2.GC_PR_BGD`` defines a possible background pixel
        * ``cv2.GC_PR_FGD`` defines a possible foreground pixel

        The GrabCut algorithm is executed with `iters` iterations. The region
        of interest `roi` can be a 4-tuple ``(x,y,width,height)``. If the ROI
        is not set, the ROI is set to the entire image, with a margin of
        `margin` pixels from the borders.

        This method is indirectly executed by :meth:`make`.
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdmodel = np.zeros((1,65), np.float64)
        fgdmodel = np.zeros((1,65), np.float64)

        # Use the margin to set the ROI if the ROI was not provided.
        if not roi:
            roi = (margin, margin, img.shape[1]-margin*2, img.shape[0]-margin*2)

        cv2.grabCut(img, mask, roi, bgdmodel, fgdmodel, iters, cv2.GC_INIT_WITH_RECT)
        return mask

    def __preprocess(self):
        """Perform preprocessing steps as specified in the configurations.

        Preprocessing steps may be:

        * Resizing
        * Color correction
        * Segmentation

        This method is executed by :meth:`make`.
        """
        if self.img is None:
            raise RuntimeError("No image is loaded")
        if 'preprocess' not in self.config:
            return

        # Scale the image down if its perimeter exceeds the maximum (if set).
        rf = 1.0
        perim = sum(self.img.shape[:2])
        max_perim = getattr(self.config.preprocess, 'maximum_perimeter', None)
        if max_perim and perim > max_perim:
            logging.info("Scaling down...")
            rf = float(max_perim) / perim
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Perform color enhancement.
        color_enhancement = getattr(self.config.preprocess, 'color_enhancement', None)
        if color_enhancement:
            for method, args in vars(color_enhancement).iteritems():
                if method == 'naik_murthy_linear':
                    logging.info("Color enhancement...")
                    self.img = ft.naik_murthy_linear(self.img)
                else:
                    raise ConfigurationError("Unknown color enhancement method '%s'" % method)

        # Perform segmentation.
        try:
            segmentation = self.config.preprocess.segmentation.grabcut
        except:
            segmentation = {}

        if segmentation:
            logging.info("Segmenting...")
            iters = getattr(segmentation, 'iters', 5)
            margin = getattr(segmentation, 'margin', 1)
            output_folder = getattr(segmentation, 'output_folder', None)

            if self.grabcut_roi:
                # Account for the resizing factor when setting the ROI.
                self.grabcut_roi = [int(x*rf) for x in self.grabcut_roi]
                self.grabcut_roi = tuple(self.grabcut_roi)

            # Get the main contour.
            self.mask = self.__grabcut(self.img, iters, self.grabcut_roi, margin)
            self.bin_mask = np.where((self.mask==cv2.GC_FGD) + (self.mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
            contour = ft.get_largest_contour(self.bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contour is None:
                raise ValueError("No contour found for binary image")

            # Create a binary mask of the main contour.
            self.bin_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            cv2.drawContours(self.bin_mask, [contour], 0, 255, -1)

            # Save the masked image to the output folder.
            if output_folder and os.path.isdir(output_folder):
                img_masked = cv2.bitwise_and(self.img, self.img, mask=self.bin_mask)
                fname = os.path.basename(self.path)
                out_path = os.path.join(output_folder, fname)
                cv2.imwrite(out_path, img_masked)

    def make(self):
        """Return the phenotype for the loaded image.

        Performs any image preprocessing if necessary and the image features
        are extracted as specified in the configurations. Finally the
        phenotype is returned as a list of floating point values.
        """
        if self.img is None:
            raise ValueError("No image was loaded")
        if self.config is None:
            raise ValueError("Configurations are not set")

        # Perform preprocessing.
        self.__preprocess()

        logging.info("Extracting features...")

        # Construct the phenotype.
        phenotype = []

        for feature, args in vars(self.config.features).iteritems():
            if feature == 'color_histograms':
                logging.info("- Running color:histograms...")
                data = self.__get_color_histograms(self.img, args, self.bin_mask)
                phenotype.extend(data)

            elif feature == 'color_bgr_means':
                logging.info("- Running color:bgr_means...")
                data = self.__get_color_bgr_means(self.img, args, self.bin_mask)
                phenotype.extend(data)

            elif feature == 'shape_outline':
                logging.info("- Running shape:outline...")
                data = self.__get_shape_outline(args, self.bin_mask)
                phenotype.extend(data)

            elif feature == 'shape_360':
                logging.info("- Running shape:360...")
                data = self.__get_shape_360(args, self.bin_mask)
                phenotype.extend(data)

            else:
                raise ValueError("Unknown feature '%s'" % feature)

        return phenotype

    def __get_color_histograms(self, src, args, bin_mask=None):
        """Executes :meth:`features.color_histograms`."""
        histograms = []
        for colorspace, bins in vars(args).iteritems():
            if colorspace.lower() == "bgr":
                colorspace = ft.CS_BGR
                img = src
            elif colorspace.lower() == "hsv":
                colorspace = ft.CS_HSV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            elif colorspace.lower() == "luv":
                colorspace = ft.CS_LUV
                img = cv2.cvtColor(src, cv2.COLOR_BGR2LUV)
            else:
                raise ValueError("Unknown colorspace '%s'" % colorspace)

            hists = ft.color_histograms(img, bins, bin_mask, colorspace)

            for hist in hists:
                hist = cv2.normalize(hist, None, -1, 1, cv2.NORM_MINMAX)
                histograms.extend( hist.ravel() )
        return histograms

    def __get_color_bgr_means(self, src, args, bin_mask=None):
        """Executes :meth:`features.color_bgr_means`."""
        if self.bin_mask is None:
            raise ValueError("Binary mask cannot be None")

        # Get the contours from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contour is None:
            raise ValueError("No contour found for binary image")

        # Create a masked image.
        img = cv2.bitwise_and(src, src, mask=bin_mask)

        bins = getattr(args, 'bins', 20)
        output = ft.color_bgr_means(img, contour, bins)

        # Normalize data to range -1 .. 1
        return output * 2.0 / 255 - 1

    def __get_shape_outline(self, args, bin_mask):
        """Executes :meth:`features.shape_outline`."""
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        k = getattr(args, 'k', 15)

        # Obtain contours (all points) from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Get the outline.
        outline = ft.shape_outline(contour, k)

        # Compute the delta's for the horizontal and vertical point pairs.
        shape = []
        for x, y in outline:
            delta_x = x[0] - x[1]
            delta_y = y[0] - y[1]
            shape.append(delta_x)
            shape.append(delta_y)

        # Normalize results.
        shape = np.array(shape, dtype=np.float32)
        shape = cv2.normalize(shape, None, -1, 1, cv2.NORM_MINMAX)

        return shape.ravel()

    def __get_shape_360(self, args, bin_mask):
        """Executes :meth:`features.shape_360`."""
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        rotation = getattr(args, 'rotation', 0)
        step = getattr(args, 'step', 1)
        t = getattr(args, 't', 8)
        output_functions = getattr(args, 'output_functions', {'mean_sd': True})

        # Get the largest contour from the binary mask.
        contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Set the rotation.
        if rotation == 'FIT_ELLIPSE':
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])
        if not 0 <= rotation <= 179:
            raise ValueError("Rotation must be in the range 0 to 179, found %s" % rotation)

        # Extract shape feature.
        intersects, center = ft.shape_360(contour, rotation, step, t)

        # Create a masked image.
        if 'color_histograms' in output_functions:
            img_masked = cv2.bitwise_and(self.img, self.img, mask=bin_mask)

        # Run the output function for each angle.
        means = []
        sds = []
        histograms = []
        for angle in range(0, 360, step):
            for f_name, f_args in vars(output_functions).iteritems():
                # Mean distance + standard deviation.
                if f_name == 'mean_sd':
                    distances = []
                    for p in intersects[angle]:
                        d = ft.point_dist(center, p)
                        distances.append(d)

                    if len(distances) == 0:
                        mean = 0
                        sd = 0
                    else:
                        mean = np.mean(distances, dtype=np.float32)
                        if len(distances) > 1:
                            sd = np.std(distances, ddof=1, dtype=np.float32)
                        else:
                            sd = 0

                    means.append(mean)
                    sds.append(sd)

                # Color histograms.
                if f_name == 'color_histograms':
                    # Get a line from the center to the outer intersection point.
                    line = None
                    if len(intersects[angle]) > 0:
                        line = ft.extreme_points([center] + intersects[angle])

                    # Create a mask for the line, where the line is foreground.
                    line_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                    if line != None:
                        cv2.line(line_mask, tuple(line[0]), tuple(line[1]), 255, 1)

                    # Create histogram from masked + line masked image.
                    hists = self.get_color_histograms(img_masked, f_args, line_mask)
                    histograms.append(hists)

        # Normalize results.
        if 'mean_sd' in output_functions:
            means = cv2.normalize(np.array(means), None, -1, 1, cv2.NORM_MINMAX)
            sds = cv2.normalize(np.array(sds), None, -1, 1, cv2.NORM_MINMAX)

        # Group the means+sds together.
        means_sds = np.array(zip(means, sds)).flatten()

        return np.append(means_sds, histograms)

class TrainData(object):

    """Store and retrieve training data.

    An instance of this class is iterable, which returns 3-tuples ``(label,
    input_data, output_data)`` per iteration. Data can either be loaded from
    file with :meth:`read_from_file` or manually appended with :meth:`append`.
    """

    def __init__(self, num_input=None, num_output=None):
        """Set the number of input and output columns.

        Training data consists of input data columns, and output data columns.
        The number of input `num_input` and output `num_output` columns must
        be specified when manually adding data with :meth:`append`.

        If :meth:`read_from_file` is used to load training data from a file,
        the number of input and output columns is automatically set.
        """
        self.labels = []
        self.input = []
        self.output = []
        self.counter = 0
        self.num_input = num_input
        self.num_output = num_output

        if num_input:
            self.set_num_input(num_input)
        if num_output:
            self.set_num_output(num_output)

    def set_num_input(self, n):
        if not n > 0:
            raise ValueError("The number of input columns must be at least 1")
        self.num_input = n

    def set_num_output(self, n):
        if not n > 1:
            raise ValueError("The number of output columns must be at least 2")
        self.num_output = n

    def read_from_file(self, path, dependent_prefix="OUT:"):
        """Load training data from file.

        Data is loaded from a tab separated file at location `path`. The file
        must have a header row with column names, and columns with a name
        starting with `dependent_prefix` are used as output columns. Optionally,
        labels for the samples can be stored in a column with the name "ID". All
        remaining columns are used as input data.
        """
        with open(path, 'r') as fh:
            reader = csv.reader(fh, delimiter="\t")

            # Figure out the format of the data.
            self.num_input = 0
            self.num_output = 0
            input_start = None
            output_start = None
            label_idx = None
            header = reader.next()
            for i, field in enumerate(header):
                if field == "ID":
                    label_idx = i
                elif field.startswith(dependent_prefix):
                    if output_start == None:
                        output_start = i
                    self.num_output += 1
                else:
                    if input_start == None:
                        input_start = i
                    self.num_input += 1

            if not self.num_input > 0:
                raise ValueError("No input columns found in training data")
            if not self.num_output > 1:
                raise ValueError("Training data needs at least 2 output " \
                    "columns, found %d" % self.num_output)

            input_end = input_start + self.num_input
            output_end = output_start + self.num_output

            for row in reader:
                if label_idx is not None:
                    self.labels.append(row[label_idx])
                else:
                    self.labels.append(None)
                self.input.append(row[input_start:input_end])
                self.output.append(row[output_start:output_end])

            self.finalize()

    def __len__(self):
        return len(self.input)

    def __iter__(self):
        return self

    def next(self):
        if self.counter >= len(self.input):
            self.counter = 0
            raise StopIteration
        else:
            self.counter += 1
            i = self.counter - 1
            return (self.labels[i], self.input[i], self.output[i])

    def append(self, input, output, label=None):
        """Append a training data row.

        A data row consists of input data `input`, output data `output`, and
        an optional sample label `label`.
        """
        if isinstance(self.input, np.ndarray):
            raise ValueError("Cannot add data once finalized")
        if len(input) != self.num_input:
            raise ValueError("Incorrect input array length (expected " \
                "length of %d)" % self.num_input)
        if len(output) != self.num_output:
            raise ValueError("Incorrect output array length (expected " \
                "length of %d)" % self.num_output)

        self.labels.append(label)
        self.input.append(input)
        self.output.append(output)

    def finalize(self):
        """Transform input and output data to Numpy arrays."""
        self.input = np.array(self.input).astype(float)
        self.output = np.array(self.output).astype(float)

    def normalize_input_columns(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        """Normalize the input columns using :meth:`cv2.normalize`.

        This method can only be called after :meth:`finalize` was executed.
        """
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this " \
                "function")

        for col in range(self.num_input):
            tmp = cv2.normalize(self.input[:,col], None, alpha, beta, norm_type)
            self.input[:,col] = tmp[:,0]

    def normalize_input_rows(self, alpha, beta, norm_type=cv2.NORM_MINMAX):
        """Normalize the input rows using :meth:`cv2.normalize`.

        This method can only be called after :meth:`finalize` was executed.
        """
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Data must be finalized before running this " \
                "function")

        for i, row in enumerate(self.input):
            self.input[i] = cv2.normalize(row, None, alpha, beta,
                norm_type).reshape(-1)

    def round_input(self, decimals=4):
        """Rounds the input data to `decimals` decimals."""
        self.input = np.around(self.input, decimals)

    def get_input(self):
        """Return the input data."""
        return self.input

    def get_output(self):
        """Return the output data."""
        return self.output

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

        The training data `data` must be an instance of :class:`TrainData`.
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

            sys.stderr.write("Ordinary training:\n")
            sys.stderr.write("* Neuron layers: %s\n" % layers)
            sys.stderr.write("* Connection rate: %s\n" % self.connection_rate)
            if not self.training_algorithm in ('FANN_TRAIN_RPROP',):
                sys.stderr.write("* Learning rate: %s\n" % self.learning_rate)
            sys.stderr.write("* Activation function for the hidden layers: %s\n" % \
                self.activation_function_hidden)
            sys.stderr.write("* Activation function for the output layer: %s\n" % \
                self.activation_function_output)
            sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)

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

            # Ordinary training.
            self.ann.train_on_data(fann_train_data, self.epochs,
                self.iterations_between_reports, self.desired_error)

        if self.train_type == 'cascade':
            sys.stderr.write("Cascade training:\n")
            sys.stderr.write("* Maximum number of neurons: %s\n" % self.max_neurons)
            sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)
            sys.stderr.write("* Activation function for the hidden layers: %s\n" % \
                self.activation_function_hidden)
            sys.stderr.write("* Activation function for the output layer: %s\n" % \
                self.activation_function_output)
            sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)

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

            # Cascade training.
            self.ann.cascadetrain_on_data(fann_train_data, self.max_neurons,
                self.neurons_between_reports, self.desired_error)

        return self.ann

    def test(self, data):
        """Test the trained neural network.

        Expects an instance of :class:`TrainData`. Returns the mean square
        error on the test data `data`.
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

        sys.stderr.write("Aivolver:\n")
        sys.stderr.write("* Maximum number of neurons: %s\n" % \
            config['ann']['neurons'])
        sys.stderr.write("* Desired error: %s\n" % \
            config['ann']['error'])
        sys.stderr.write("* Activation function: %s\n" % \
            config['ann']['activation_function'])

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
