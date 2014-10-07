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
import sys

import cv2
import imgpheno as ft
import numpy as np
from pyfann import libfann
from sqlalchemy import func
from sqlalchemy import orm
from sqlalchemy.ext.automap import automap_base
import yaml

from exceptions import *

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


class Struct(Namespace):

    """Return a dictionary as an object."""

    def __init__(self, d):
        if not isinstance(d, dict):
            raise TypeError("Expected a dictionary, got {0} instead".format(type(d)))
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
        return getattr(self, key)

class Common(object):

    """Collection of common functions.

    This class is used as a super class in several scripts that implement this
    package. It provides commonly used functions.
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
            raise TypeError("Configurations object must be of type Struct, not %s" % type(config))

        try:
            path = config.preprocess.segmentation.output_folder
        except:
            path = None
        if path and not os.path.isdir(path):
            logging.error("Found a configuration error")
            raise IOError("Cannot open %s (no such directory)" % path)

        self.config = config

    def set_photo_count_min(self, count):
        """Set a minimum for photos count per species.

        This setting is used by :meth:`get_taxon_hierarchy`. If `count` is a
        positive integer, only the species with a photo count of at least
        `count` are used to build the taxon hierarchy.
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

    def get_photos(self, session, metadata):
        """Return photos from the database.

        Returns 2-tuples ``(id, path)``.
        """
        Base = automap_base(metadata=metadata)
        Base.prepare()
        Photos = Base.classes.photos
        q = session.query(Photos.id, Photos.path)
        return q

    def get_photos_with_class(self, session, metadata, filter_):
        """Return photos with corresponding class from the database.

        Photos obtained from the photo metadata database are queried by rules
        set in the classification filter `filter_`. Filters are those as
        returned by :meth:`classification_hierarchy_filters`. Returned rows
        are 3-tuples ``(photo_id, photo_path, class)``.
        """
        if 'class' not in filter_:
            raise ValueError("The filter is missing the 'class' key")
        if isinstance(filter_, dict):
            filter_ = Struct(filter_)
        for key in vars(filter_):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in filter" % key)

        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        stmt1 = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        stmt2 = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        stmt3 = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query(Photos.id, Photos.path, getattr(filter_, 'class')).\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3)

        # Add the WHERE clauses to the query.
        if 'where' in filter_:
            for rank, taxon in vars(filter_.where).items():
                if rank == 'genus':
                    q = q.filter(stmt1.c.genus == taxon)
                elif rank == 'section':
                    q = q.filter(stmt2.c.section == taxon)
                elif rank == 'species':
                    q = q.filter(stmt3.c.species == taxon)

        return q

    def get_photo_ids(self, session, metadata):
        """Return all photo IDs.

        This generator returns integers.
        """
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos

        # Construct the query.
        q = session.query(Photos.id)

        for row in q:
            yield row[0]

    def get_photo_ids_with_genus_section_species(self, session, metadata):
        """Return photo IDs with genus, section, and species class.

        This generator returns 4-tuples ``(photo_id, genus, section, species)``.
        """
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        q_genus = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        q_section = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        q_species = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query(Photos.id, 'genus', 'section', 'species').\
            join(q_genus).\
            outerjoin(q_section).\
            join(q_species)

        q = q.filter

        return q

    def get_classes_from_filter(self, session, metadata, filter_):
        """Return the classes for a classification filter.

        Requires access to a database via an SQLAlchemy Session `session` and
        MetaData object `metadata`.

        This is a generator that returns one class at a time. The unique set
        of classes for the classification filter `filter_` are returned.
        Filters are those as returned by
        :meth:`classification_hierarchy_filters`.
        """
        if 'class' not in filter_:
            raise ValueError("The filter is missing the 'class' key")
        if isinstance(filter_, dict):
            filter_ = Struct(filter_)
        for key in vars(filter_):
            if key not in ('where', 'class'):
                raise ValueError("Unknown key '%s' in filter" % key)

        levels = ['genus','section','species']
        path = []
        if 'where' in filter_:
            for level in levels:
                try:
                    path.append( getattr(filter_.where, level) )
                except:
                    if level == getattr(filter_, 'class'):
                        break
                    else:
                        raise ValueError("Incorrect filter: %s" % filter_)

        hr = self.get_taxon_hierarchy(session, metadata)
        classes = self.get_childs_from_hierarchy(hr, path)

        return set(classes)

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

        for genus, section, species, count in self.get_taxa(session, metadata):
            if self._photo_count_min > 0 and count < self._photo_count_min:
                continue

            if genus not in hierarchy:
                hierarchy[genus] = {}
            if section not in hierarchy[genus]:
                hierarchy[genus][section] = []
            hierarchy[genus][section].append(species)
        return hierarchy

    def get_taxa(self, session, metadata):
        """Return the taxa from the photo metadata database.

        Taxa are returned as 4-tuples ``(genus, section, species,
        photo_count)``.
        """
        Base = automap_base(metadata=metadata)
        Base.prepare()

        # Get the table classes.
        Photos = Base.classes.photos
        PhotosTaxa = Base.classes.photos_taxa
        Taxa = Base.classes.taxa
        Rank = Base.classes.ranks

        # Construct the sub queries.
        stmt1 = session.query(PhotosTaxa.photo_id, Taxa.name.label('genus')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'genus').subquery()
        stmt2 = session.query(PhotosTaxa.photo_id, Taxa.name.label('section')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'section').subquery()
        stmt3 = session.query(PhotosTaxa.photo_id, Taxa.name.label('species')).\
            join(Taxa).join(Rank).\
            filter(Rank.name == 'species').subquery()

        # Construct the query.
        q = session.query('genus', 'section', 'species',
                func.count(Photos.id)).\
            select_from(Photos).\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3).\
            group_by('genus', 'section', 'species')

        return q

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
        :meth:`get_photos_with_class` and :meth:`get_classes_from_filter`.
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
        class_ = filter_.get('class')
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
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        # Get the contours from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contour == None:
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

    def __init__(self, num_input=0, num_output=0):
        """Set the number of input and output columns.

        Training data consists of input data columns, and output data columns.
        The number of input `num_input` and output `num_output` columns must
        be specified when manually adding data with :meth:`append`.

        If :meth:`read_from_file` is used to load training data from a file,
        the number of input and output columns is automatically set.
        """
        self.num_input = num_input
        self.num_output = num_output
        self.labels = []
        self.input = []
        self.output = []
        self.counter = 0

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

            if self.num_input == 0:
                raise ValueError("No input columns found in training data")
            if self.num_output  == 0:
                raise ValueError("No output columns found in training data")

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
        self.connection_rate = 1
        self.learning_rate = 0.7
        self.hidden_layers = 1
        self.hidden_neurons = 8
        self.epochs = 500000
        self.iterations_between_reports = self.epochs / 100
        self.desired_error = 0.0001
        self.training_algorithm = 'TRAIN_RPROP'
        self.activation_function_hidden = 'SIGMOID_STEPWISE'
        self.activation_function_output = 'SIGMOID_STEPWISE'
        self.train_data = None
        self.test_data = None

    def set_train_data(self, data):
        """Set the training data on which to train the network on.

        The training data `data` must be an instance of :class:`TrainData`.
        """
        if not isinstance(data, TrainData):
            raise ValueError("Training data must be an instance of TrainData")
        if not data:
            raise ValueError("Train data is empty")
        self.train_data = data

    def train(self, train_data):
        self.set_train_data(train_data)

        hidden_layers = [self.hidden_neurons] * self.hidden_layers
        layers = [self.train_data.num_input]
        layers.extend(hidden_layers)
        layers.append(self.train_data.num_output)

        sys.stderr.write("Network layout:\n")
        sys.stderr.write("* Neuron layers: %s\n" % layers)
        sys.stderr.write("* Connection rate: %s\n" % self.connection_rate)
        if self.training_algorithm not in ('FANN_TRAIN_RPROP',):
            sys.stderr.write("* Learning rate: %s\n" % self.learning_rate)
        sys.stderr.write("* Activation function for the hidden layers: %s\n" % \
            self.activation_function_hidden)
        sys.stderr.write("* Activation function for the output layer: %s\n" % \
            self.activation_function_output)
        sys.stderr.write("* Training algorithm: %s\n" % self.training_algorithm)

        self.ann = libfann.neural_net()
        self.ann.create_sparse_array(self.connection_rate, layers)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_hidden(
            getattr(libfann, self.activation_function_hidden))
        self.ann.set_activation_function_output(
            getattr(libfann, self.activation_function_output))
        self.ann.set_training_algorithm(
            getattr(libfann, self.training_algorithm))

        fann_train_data = libfann.training_data()
        fann_train_data.set_train_data(self.train_data.get_input(),
            self.train_data.get_output())

        self.ann.train_on_data(fann_train_data, self.epochs,
            self.iterations_between_reports, self.desired_error)
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
