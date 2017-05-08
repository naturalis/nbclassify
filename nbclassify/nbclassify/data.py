# -*- coding: utf-8 -*-

"""Train data routines."""

from copy import deepcopy
from cPickle import dump, load, HIGHEST_PROTOCOL
import csv
import datetime
import logging
import os
import shelve
import sys

import cv2
import imgpheno as ft
import numpy as np
import scipy.cluster.vq as vq
from sklearn.preprocessing import MinMaxScaler

from . import conf
from .base import Common, Struct
from .exceptions import *
from .functions import (combined_hash, classification_hierarchy_filters,
    get_codewords, get_config_hashables, readable_filter,
    get_bowcode_from_surf_features)
import nbclassify.db as db

class PhenotypeCache(object):

    """Cache and retrieve phenotypes.

    Must be used within a database session scope.
    """

    def __init__(self):
        self._cache = {}

    def get_single_feature_configurations(self, config):
        """Return each configuration together with each feature separately.

        Normally, a configuration contains ``features``, a list of all the
        configurations for features to be extracted. This function returns each
        feature configuration separately, but still paired with its
        corresponding non-feature configurations (ann, preprocess, etc.).
        Configurations are obtained from the classification hierarchy `hr`. This
        is a generator that returns 2-tuples ``(hash, config)``.
        """
        seen_hashes = []

        # Get the classification hierarchy.
        try:
            hr = config.classification.hierarchy
        except:
            raise ConfigurationError("classification hierarchy not set")

        # Get a list of all configurations from the main configurations object.
        # Each item in the classification hierarchy is in itself also a
        # configurations object.
        configs = [config]
        configs.extend(hr)

        # Traverse the classification hierarchy for features that need to be
        # extracted.
        for c in configs:
            # Get the individual features settings.
            try:
                features = c.features
            except:
                raise ConfigurationError("features not set")

            for name, feature in vars(features).iteritems():
                # Construct a new configuration object with a single feature
                # which can be passed to Phenotyper. We want to store only one
                # feature per cache file.
                c = deepcopy(c)
                c.features = Struct({name: feature})

                # Get those configuration objects that are suitable for creating
                # hashes for the feature caches.
                hashables = get_config_hashables(c)

                # Create a hash from the configurations. The hash must be
                # different for different configurations.
                hash_ = combined_hash(feature, *hashables)

                if hash_ not in seen_hashes:
                    seen_hashes.append(hash_)
                    yield (hash_, c)

    def get_cache(self):
        """Return the cache as a nested dictionary.

        Cache is returned as a nested dictionary ``{feature_name: {'key':
        feature}, ..}``.
        """
        return self._cache

    def get_features(self, cache_dir, hash_):
        """Return features from cache for a given hash.

        Looks for a cache file in the directory `cache_dir` with the file name
        `hash_`. Returns None if the cache could not be found.
        """
        try:
            cache = shelve.open(os.path.join(cache_dir, str(hash_)))
            c = dict(cache)
            cache.close()
            return c
        except:
            return None

    def get_phenotype(self, key):
        """Return the phenotype for key `key`.

        The `key` is whatever was used as a key for storing the cache for each
        feature. Method :meth:`load_cache` must be called before calling this
        method.
        """
        if not self._cache:
            raise ValueError("Cache is not loaded")
        phenotype = []
        for name in sorted(self._cache.keys()):
            phenotype.extend(self._cache[name][str(key)])
        return phenotype

    def make(self, image_dir, cache_dir, config, update=False):
        """Cache features for an image directory to disk.

        One cache file is created for each feature configuration set in the
        configurations. The caches are saved in the target directory
        `cache_dir`. Each cache file is a Python shelve, a persistent,
        dictionary-like object. If `update` is set to True, existing features
        are updated. Method :meth:`get_phenotype` can then be used to retrieve
        these features and combined them to phenotypes.
        """
        session, metadata = db.get_session_or_error()

        phenotyper = Phenotyper()

        # Get a list of all the photos in the database.
        photos = db.get_photos(session, metadata)

        # Cache each feature for each photo separately. One cache per
        # feature type is created, and each cache contains the features
        # for all images.
        for hash_, c in self.get_single_feature_configurations(config):
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)

            cache_path = os.path.join(cache_dir, str(hash_))
            sys.stderr.write("Caching features in `%s`...\n" % cache_path)

            # Create a shelve for storing features. Empty existing shelves.
            cache = shelve.open(cache_path)

            # Set the new config.
            phenotyper.set_config(c)

            # Cache the feature for each photo.
            for photo in photos:
                # Skip feature extraction if the feature already exists, unless
                # update is set to True.
                if not update and str(photo.md5sum) in cache:
                    continue

                logging.info("Processing photo %s...", photo.id)
                
                # Extract a feature and cache it.
                im_path = os.path.join(image_dir, photo.path)
                phenotyper.set_image(im_path)
                cache[str(photo.md5sum)] = phenotyper.make()

            cache.close()

    def load_cache(self, cache_dir, config):
        """Load cache for a feature extraction configuration.

        Looks for cache files in the directory `cache_dir`. Expects to find the
        attribute ``features``, and optionally ``preprocess`` in the
        configurations object `config`. This function does not traverse the
        classification hierarchy, so this function must be called once for each
        features configuration before calling :meth:`get_phenotype`.
        """
        try:
            features = config.features
        except:
            raise ConfigurationError("features not set")

        self._cache = {}
        for name, feature in vars(features).iteritems():
            hashables = get_config_hashables(config)
            hash_ = combined_hash(feature, *hashables)
            cache = self.get_features(cache_dir, hash_)
            if not cache:
                raise IOError("Cache {0} not found".format(hash_))
            self._cache[name] = cache

class Phenotyper(object):
    """Extract features from a digital image and return as a phenotype.

    Uses the :mod:`imgpheno` package to extract features from the image. Use
    :meth:`set_image` to load an image and :meth:`set_config` to set a
    configuration object as returned by
    :meth:`~nbclassify.functions.open_config`. Then :meth:`make` can be called
    to extract the features as specified in the configurations object and return
    the phenotype. A single phenotypes is returned, which is a list of floating
    point numbers.
    """

    def __init__(self):
        """Set the default attributes."""
        self.path = None
        self.config = None
        self.img = None
        self.mask = None
        self.bin_mask = None
        self.roi = None
        self.scaler = None

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

        # Reset image related variables so one instance can be used for multiple
        # images.
        self.path = path
        self.mask = None
        self.bin_mask = None

        return self.img

    def set_config(self, config):
        """Set the configurations object.

        Expects a configuration object as returned by
        :meth:`~nbclassify.functions.open_config`.
        """
        if not isinstance(config, Struct):
            raise TypeError("Expected a Struct instance, got {0} instead".\
                format(type(config)))
        if not 'features' in config:
            raise ConfigurationError("Features to extract not set. Missing " \
                "the `features` setting.")

        # Set the normalization method.
        try:
            min_, max_ = config.data.normalize.min_max
            self.set_norm_minmax(min_, max_)
            logging.info("Normalizing features to range %s..%s", min_, max_)
        except AttributeError:
            self.scaler = None

        self.config = config

    def set_norm_minmax(self, a=0, b=1):
        """Standardize features by scaling each feature to a given range.

        Desired range of transformed data is given by the minimum `a` and
        maximum `b`. by For each normalization, the scaler is first fit to the
        correct feature range. For example, for color intensities of the BGR
        color space, the scaler is fit to range 0..255.
        """
        a = float(a)
        b = float(b)
        self.scaler = MinMaxScaler(copy=True, feature_range=(a, b))

    def set_roi(self, roi):
        """Set the region of interest for the image.

        If GrabCut is set as the segmentation algorithm, then GrabCut is
        executed with this region of interest.

        The ROI must be a 4-tuple ``(x, y, width, height)``.
        """
        if roi is not None:
            if len(roi) != 4:
                raise ValueError("ROI must be a list of four integers")
            for x in roi:
                if not (isinstance(x, int) and x >= 0):
                    raise ValueError("ROI must be a (x, y, w, h) tuple")
        self.roi = roi

    def __grabcut(self, img, iters=5, roi=None, margin=5):
        """Wrapper for OpenCV's grabCut function.

        Runs the GrabCut algorithm for segmentation. Returns an 8-bit single-
        channel mask. Its elements may have the following values:

        * ``cv2.GC_BGD`` defines an obvious background pixel
        * ``cv2.GC_FGD`` defines an obvious foreground pixel
        * ``cv2.GC_PR_BGD`` defines a possible background pixel
        * ``cv2.GC_PR_FGD`` defines a possible foreground pixel

        The GrabCut algorithm is executed with `iters` iterations. The region of
        interest `roi` can be a 4-tuple ``(x, y, width, height)``. If the ROI is
        not set, the ROI is set to the entire image, with a margin of `margin`
        pixels from the borders.

        This method is indirectly executed by :meth:`make`.
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdmodel = np.zeros((1,65), np.float64)
        fgdmodel = np.zeros((1,65), np.float64)

        # Use the margin to set the ROI if the ROI was not provided.
        if not roi:
            h, w = img.shape[:2]
            roi = (margin, margin, w - margin * 2, h - margin * 2)

        logging.debug("Going to run GrabCut")
        cv2.grabCut(img, mask, roi, bgdmodel, fgdmodel, iters,
            cv2.GC_INIT_WITH_RECT)

        return mask

    def __preprocess(self):
        """Perform preprocessing steps as specified in the configurations.

        Preprocessing steps may be:

        * Resizing
        * Color correction
        * Segmentation or cropping

        This method is executed by :meth:`make`.
        """
        if self.img is None:
            raise RuntimeError("No image is loaded")
        if 'preprocess' not in self.config:
            return

        # Scale the image down if its perimeter (width+height) exceeds the
        # maximum. If a ROI is set, use the perimeter of the ROI instead, or
        # else we might end up with a very small ROI.
        if self.roi:
            perim = sum(self.roi[2:4])
        else:
            perim = sum(self.img.shape[:2])

        rf = 1.0
        max_perim = getattr(self.config.preprocess, 'maximum_perimeter', None)
        if max_perim and perim > max_perim:
            logging.info("Scaling down...")
            rf = float(max_perim) / perim
            self.img = cv2.resize(self.img, None, fx=rf, fy=rf)

        # Account for the resizing factor if a ROI is set.
        if self.roi:
            self.roi = [int(x*rf) for x in self.roi]
            self.roi = tuple(self.roi)

        # Perform color enhancement.
        color_enhancement = getattr(self.config.preprocess,
            'color_enhancement', None)
        if color_enhancement:
            for method, args in vars(color_enhancement).iteritems():
                if method == 'naik_murthy_linear':
                    logging.info("Color enhancement...")
                    self.img = ft.naik_murthy_linear(self.img)
                else:
                    raise ConfigurationError("Unknown color enhancement "\
                        "method '%s'" % method)

        # Perform segmentation.
        try:
            segmentation = self.config.preprocess.segmentation.grabcut
        except:
            segmentation = {}

        if segmentation:            
            iters = getattr(segmentation, 'iters', 5)
            margin = getattr(segmentation, 'margin', 1)
            output_folder = getattr(segmentation, 'output_folder', None)
            logging.debug("Segmenting iters=%s margin=%s output_folder=%s" % ( iters, margin, output_folder ) )

            # Get the main contour.
            self.mask = self.__grabcut(self.img, iters, self.roi, margin)
            logging.debug("Ran GrabCut, have mask")
            self.bin_mask = np.where((self.mask==cv2.GC_FGD) + \
                (self.mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
            logging.debug("Made mask binary")
            try:
                contour = ft.get_largest_contour(self.bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                ( exc_type, exc_value, exc_traceback ) = sys.exc_info()
                logging.warning("Error: %s, %s" % ( exc_type, exc_value ) )
                logging.warning(exc_traceback.format_exc)
            logging.info("Computed contour")
            if contour is None:
                raise ValueError("No contour found for binary image")

            # Create a binary mask of the main contour.
            self.bin_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            logging.info("Created binary mask of main contour")
            cv2.drawContours(self.bin_mask, [contour], 0, 255, -1)
            logging.info("Drew contour")

            # Save the masked image to the output folder.
            if output_folder:
                img_masked = cv2.bitwise_and(self.img, self.img,
                    mask=self.bin_mask)
                logging.info("Masked image")

                out_path = os.path.join(output_folder, self.path)
                out_dir = os.path.dirname(out_path)
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)

                cv2.imwrite(out_path, img_masked)
                logging.info("Wrote image to file")
        else:
            # Crop image in stead of segmenting.
            try:
                crop = self.config.preprocess.crop
            except:
                crop = {}

            if crop:
                logging.info("Cropping image...")
                roi_pix = getattr(crop, 'roi_pix', None)
                roi_frac = getattr(crop, 'roi_frac', None)
                if roi_pix:
                    # roi_pix is like (x, y, w, h) in pixel units.
                    if len(roi_pix) != 4:
                        raise ValueError(
                            "roi_pix must be a list of four integers.")
                    for x in roi_pix:
                        if not (isinstance(x, int) and x >= 0):
                            raise ValueError(
                                "roi_pix must be a (x, y, w, h) tuple "
                                "of integers.")
                    self.roi = roi_pix
                elif roi_frac:
                    # roi_frac is like (x1, x2, y1, y2) in fractions
                    # of total img size.
                    if len(roi_frac) != 4:
                        raise ValueError(
                            "roi_frac must be a list of four floats.")
                    for x in roi_frac:
                        if not 0 <= x <= 1:
                            raise ValueError(
                                "roi_frac must be a (x1, x2, y1, y2) tuple, "
                                "where the values are floats between 0 and 1.")
                    if not (roi_frac[0] < roi_frac[1] and
                            roi_frac[2] < roi_frac[3]):
                        raise ValueError(
                            "roi_frac must be a (x1, x2, y1, y2) tuple, "
                            "where x1 < x2 and y1 < y2.")
                    # Make ROI like (x, y, w, h).
                    self.roi = (int(self.img.shape[1] * roi_frac[0]),
                                int(self.img.shape[0] * roi_frac[2]),
                                int(self.img.shape[1] * roi_frac[1]) -
                                int(self.img.shape[1] * roi_frac[0]),
                                int(self.img.shape[0] * roi_frac[3]) -
                                int(self.img.shape[0] * roi_frac[2]))
                else:
                    logging.warning("No ROI for cropping found. Proceed "
                                    "without cropping.")
                    self.roi = (0, 0, self.img.shape[1], self.img.shape[0])

                # Crop image to given ROI.
                self.img = self.img[self.roi[1]: self.roi[1] + self.roi[3],
                                    self.roi[0]: self.roi[0] + self.roi[2]]

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
        for name in sorted(vars(self.config.features).keys()):
            args = self.config.features[name]

            if name == 'color_histograms':
                logging.info("- Running color:histograms...")
                data = self.__get_color_histograms(self.img, args, self.bin_mask)
                phenotype.extend(data)

            elif name == 'color_bgr_means':
                logging.info("- Running color:bgr_means...")
                data = self.__get_color_bgr_means(self.img, args, self.bin_mask)
                phenotype.extend(data)

            elif name == 'shape_outline':
                logging.info("- Running shape:outline...")
                data = self.__get_shape_outline(args, self.bin_mask)
                phenotype.extend(data)

            elif name == 'shape_360':
                logging.info("- Running shape:360...")
                data = self.__get_shape_360(args, self.bin_mask)
                phenotype.extend(data)

            elif name == 'surf':
                logging.info("- Running feature:surf...")
                data = self.__get_surf_features(args, self.img)
                phenotype.extend(data)

            else:
                raise ValueError("Unknown feature `%s`" % name)

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

            # Get the color space ranges. Correct for the exclusive upper
            # boundaries.
            ranges = np.array(ft.CS_RANGE[colorspace]).astype(float) - [0,1]

            for i, hist in enumerate(hists):
                # Normalize the features if a scaler is set.
                if self.scaler:
                    self.scaler.fit(ranges[i])
                    hist = self.scaler.transform( hist.astype(float) )

                histograms.extend( hist.ravel() )

        return histograms

    def __get_color_bgr_means(self, src, args, bin_mask=None):
        """Executes :meth:`features.color_bgr_means`."""
        segmentation = getattr(self.config.preprocess, 'segmentation', False)
        if not segmentation:
            bin_mask = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if bin_mask is None:
            raise ValueError("Binary mask cannot be None")

        # Get the contours from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if contour is None:
            raise ValueError("No contour found for binary image")

        # Create a masked image.
        img = cv2.bitwise_and(src, src, mask=bin_mask)

        bins = getattr(args, 'bins', 20)
        hor_means, ver_means = ft.color_bgr_means(img, contour, bins)
        output = np.append(hor_means, ver_means).astype(float)

        # Normalize the features if a scaler is set.
        if self.scaler:
            self.scaler.fit([0.0, 255.0])
            output = self.scaler.transform( output )

        return output

    def __get_shape_outline(self, args, bin_mask):
        """Executes :meth:`features.shape_outline`."""
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        k = getattr(args, 'k', 15)

        # Obtain contours (all points) from the mask.
        contour = ft.get_largest_contour(bin_mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
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
        shape = np.array(shape, dtype=float)

        # Normalize the features if a scaler is set.
        if self.scaler:
            shape = self.scaler.fit_transform(shape)

        return shape

    def __get_shape_360(self, args, bin_mask):
        """Executes :meth:`features.shape_360`."""
        if self.bin_mask == None:
            raise ValueError("Binary mask cannot be None")

        rotation = getattr(args, 'rotation', 0)
        step = getattr(args, 'step', 1)
        t = getattr(args, 't', 8)
        output_functions = getattr(args, 'output_functions', {'mean_sd': True})

        # Get the largest contour from the binary mask.
        contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
        if contour == None:
            raise ValueError("No contour found for binary image")

        # Set the rotation.
        if rotation == 'FIT_ELLIPSE':
            box = cv2.fitEllipse(contour)
            rotation = int(box[2])
        if not 0 <= rotation <= 179:
            raise ValueError("Rotation must be in the range 0 to 179, "\
                "found %s" % rotation)

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
                    if intersects[angle]:
                        line = ft.extreme_points([center] + intersects[angle])

                    # Create a mask for the line, where the line is foreground.
                    line_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                    if line is not None:
                        cv2.line(line_mask, tuple(line[0]), tuple(line[1]),
                            255, 1)

                    # Create histogram from masked + line masked image.
                    hists = self.__get_color_histograms(img_masked, f_args,
                        line_mask)
                    histograms.append(hists)

        means = means.astype(float)
        sds = sds.astype(float)

        # Normalize the features if a scaler is set.
        if self.scaler and 'mean_sd' in output_functions:
            means = self.scaler.fit_transform(means)
            sds = self.scaler.fit_transform(sds)

        # Group the means+sds together.
        means_sds = np.array(zip(means, sds)).flatten()

        return np.append(means_sds, histograms)

    def __get_surf_features(self, args, src):
        """Executes :meth:`features.surf`."""
        threshVal = getattr(args, 'thresholdVal', None)
        maxVal = getattr(args, 'maxVal', None)
        ht = getattr(args, 'HessianThreshold', 400)
        mask = getattr(args, 'mask', None)

        if type(src) == np.ndarray:
            # Convert img to grayscale if it is in color.
            if len(src.shape) == 3:
                img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = src

            # Truncate img if preferred.
            if threshVal and maxVal:
                ret, thresh = cv2.threshold(img_gray, threshVal, maxVal,
                                        cv2.THRESH_TRUNC)
            else:
                thresh = img_gray

            # Get keypoints and descriptors from the SURF features.
            _kp, des = ft.surf_features(thresh, ht, mask)

            return des


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
        if not n > 0:
            raise ValueError("The number of output columns must be at least 1")
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
            if not self.num_output > 0:
                raise ValueError("Training data needs at least 1 output " \
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

class MakeTrainData(Common):

    """Generate training data.

    Must be used within a database session scope.
    """

    def __init__(self, config, cache_path):
        """Constructor for training data generator.

        Expects a configurations object `config` and the path to the directory
        where extracted image features are cached `cache_path`.
        """
        super(MakeTrainData, self).__init__(config)
        self.set_cache_path(cache_path)
        self.subset = None
        self.cache = PhenotypeCache()

    def set_cache_path(self, path):
        """Set the directory where the feature caches are stored."""
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.cache_path = path

    def set_subset(self, subset):
        """Set the sample subset that should be used for export.

        A subset of the samples can be exported by providing a list of sample
        IDs `subset` for the samples to be exported. If subset is None, all
        samples are exported.
        """
        if subset is not None and not isinstance(subset, set):
            subset = set(subset)
        self.subset = subset

    def export(self, filename, filter_, config=None, codebook_file=None):
        """Write the training data to `filename`.

        Images to be processed are obtained from the database. Which images are
        obtained and with which classes is set by the filter `filter_`. Image
        fingerprints are obtained from cache, which must have been created for
        configuration `config` or `self.config`.
        """
        session, metadata = db.get_session_or_error()

        if not conf.force_overwrite and os.path.isfile(filename):
            raise FileExistsError(filename)

        # Get the classification categories from the database.
        classes = db.get_classes_from_filter(session, metadata, filter_)
        assert len(classes) > 0, \
            "No classes found for filter `%s`" % filter_

        # Get the photos and corresponding classification using the filter.
        images = db.get_filtered_photos_with_taxon(session, metadata, filter_)
        images = images.all()

        if not images:
            logging.info("No images found for the filter `%s`", filter_)
            return

        if self.get_photo_count_min():
            assert len(images) >= self.get_photo_count_min(), \
                "Expected to find at least photo_count_min={0} photos, found " \
                "{1}".format(self.get_photo_count_min(), len(images))

        # Calculate the number of images that will be processed, taking into
        # account the subset.
        photo_ids = np.array([photo.id for photo, _ in images])

        if self.subset:
            n_images = len(np.intersect1d(list(photo_ids), list(self.subset)))
        else:
            n_images = len(images)

        logging.info("Going to process %d photos...", n_images)

        # Make a codeword for each class.
        codewords = get_codewords(classes)

        # Get the configurations.
        if not config:
            config = self.config

        # Load the fingerprint cache.
        self.cache.load_cache(self.cache_path, config)

        # Check if the BagOfWords alogrithm needs to be applied.
        use_bow = False
        for name in sorted(vars(self.config.features).keys()):
            if name == 'surf':
                use_bow = True
        if use_bow and codebook_file == None:
            codebook = self.__make_codebook(images, filename)
        elif use_bow:
            with open(codebook_file, "rb") as cb:
                codebook = load(cb)

        # Construct the header.
        if use_bow:
            n_clusters = len(codebook)
        else:
            n_clusters = None
        header_data, header_out = self.__make_header(len(classes), n_clusters)
        header = ["ID"] + header_data + header_out

        # Generate the training data.
        with open(filename, 'w') as fh:
            # Write the header.
            fh.write( "%s\n" % "\t".join(header) )

            # Set the training data.
            training_data = TrainData(len(header_data), len(classes))

            for photo, class_ in images:
                # Only export the subset if an export subset is set.
                if self.subset and photo.id not in self.subset:
                    continue

                logging.info("Processing `%s` of class `%s`...",
                    photo.path, class_)

                # Get phenotype for this image from the cache.
                phenotype = self.cache.get_phenotype(photo.md5sum)

                # If the BagOfWords algorithm is applied,
                # convert phenotype of SURF features to BOW-code.
                if use_bow:
                    surf_feat = []
                    bgr_feat = []
                    surf_locations = []
                    bgr_locations = []
                    for featnr in range(len(phenotype)):
                        # Check if phenotype is created with SURF or BGR.
                        if phenotype[featnr].shape == (128,):
                            surf_locations.append(featnr)
                            surf_feat.append(phenotype[featnr])
                        else:
                            bgr_locations.append(featnr)
                            bgr_feat.append(phenotype[featnr])

                    bowcode = get_bowcode_from_surf_features(surf_feat,
                                                               codebook)
                    if 0 in bgr_locations:
                        phenotype = list(bgr_feat)
                        phenotype.extend(bowcode)
                    else:
                        phenotype = list(bowcode)
                        phenotype.extend(bgr_feat)

                assert len(phenotype) == len(header_data), \
                    "Fingerprint size mismatch. According to the header " \
                    "there are {0} data columns, but the fingerprint has " \
                    "{1}".format(len(header_data), len(phenotype))

                training_data.append(phenotype, codewords[class_],
                    label=photo.id)

            training_data.finalize()

            if not training_data:
                raise ValueError("Training data cannot be empty")

            # Round feature data only if BOW is not applied.
            if not use_bow:
                training_data.round_input(6)

            # Write data rows.
            for photo_id, input_, output in training_data:
                row = [str(photo_id)]
                row.extend(input_.astype(str))
                row.extend(output.astype(str))
                fh.write("%s\n" % "\t".join(row))

        logging.info("Training data written to %s", filename)

    def __make_header(self, n_out, n_clusters):
        """Construct a header from features settings.

        Header is returned as a 2-tuple ``(data_columns, output_columns)``.
        """
        if 'features' not in self.config:
            raise ConfigurationError("missing `features`")

        data = []
        out = []

        # Always sort the features by name so that the headers match the
        # data column.
        features = sorted(vars(self.config.features).keys())

        for feature in features:
            args = self.config.features[feature]

            if feature == 'color_histograms':
                for colorspace, bins in vars(args).iteritems():
                    for ch, n in enumerate(bins):
                        for i in range(1, n+1):
                            data.append("%s:%d" % (colorspace[ch], i))

            if feature == 'color_bgr_means':
                bins = getattr(args, 'bins', 20)
                for i in range(1, bins+1):
                    for axis in ("HOR", "VER"):
                        for ch in "BGR":
                            data.append("BGR_MN:%d.%s.%s" % (i,axis,ch))

            if feature == 'shape_outline':
                n = getattr(args, 'k', 15)
                for i in range(1, n+1):
                    data.append("OUTLINE:%d.X" % i)
                    data.append("OUTLINE:%d.Y" % i)

            if feature == 'shape_360':
                step = getattr(args, 'step', 1)
                output_functions = getattr(args, 'output_functions', {'mean_sd': 1})
                for f_name, f_args in vars(output_functions).iteritems():
                    if f_name == 'mean_sd':
                        for i in range(0, 360, step):
                            data.append("360:%d.MN" % i)
                            data.append("360:%d.SD" % i)

                    if f_name == 'color_histograms':
                        for i in range(0, 360, step):
                            for cs, bins in vars(f_args).iteritems():
                                for j, color in enumerate(cs):
                                    for k in range(1, bins[j]+1):
                                        data.append("360:%d.%s:%d" % (i,color,k))

            if feature == 'surf':
                for i in range(1, n_clusters+1):
                    data.append("CL%d" % i)


        # Write classification columns.
        try:
            out_prefix = self.config.data.dependent_prefix
        except:
            out_prefix = OUTPUT_PREFIX

        for i in range(1, n_out + 1):
            out.append("%s%d" % (out_prefix, i))

        return (data, out)

    def __make_codebook(self, images, filename):
        # Create nparray with all descriptors of the SURF features.
        descr_array = np.zeros((len(images) * 1000, 128))
        position = 0
        for photo, class_ in images:
            # Only export the subset if an export subset is set.
            if self.subset and photo.id not in self.subset:
                continue

            # Get descriptors for this image from the cache.
            descriptors = self.cache.get_phenotype(photo.md5sum)
            
            # Convert list to nparray.
            surf_features = []
            for feat in descriptors:
		if feat.shape == (128,):
                    surf_features.append(feat)
            descriptors = np.asarray(surf_features)

            # Add descriptors to nparray.
            n_features = descriptors.shape[0]
            while position + n_features > descr_array.shape[0]:
                elongation = np.zeros_like(descr_array)
                descr_array = np.vstack((descr_array, elongation))
            descr_array[position: position + n_features] = descriptors
            position += n_features

        # Adjust size of nparray to number of descriptors.
        descr_array = np.resize(descr_array, (position, 128))

        # Get number of clusters.
        bow_clusters = getattr(self.config.features['surf'], 'bow_clusters', None)
        if str(bow_clusters).isdigit():
            n_clusters = int(bow_clusters)
        elif str(bow_clusters).lower() == 'root':
            n_clusters = int(np.sqrt(descr_array.shape[0]))
        else:
            logging.warning("No (valid) value for bow_clusters is set in "
                            "configurations. Default square root of total "
                            "number of features will be used.")
            n_clusters = int(np.sqrt(descr_array.shape[0]))

        logging.info("%d extracted features will now be clustered into "
                     "%d clusters to create a codebook (this will take "
                     "a while)...", descr_array.shape[0], 
                     n_clusters)
                     
        start = datetime.datetime.now().replace(microsecond=0)
        logging.info("\nStart creating codebook at: %s\n", start)
        codebook, _distortion = vq.kmeans(descr_array, n_clusters)
        end = datetime.datetime.now().replace(microsecond=0)
        
        # Check if the length of the codebook is correct.
        while len(codebook) != n_clusters:
            time = end - start
            logging.warning("%d clusters were created in stead of %d. "
                            "The codebook must be created again. This "
                            "will probably take the same time as last "
                            "time: %s (H:M:S), starting at %s", 
                            len(codebook), n_clusters, time,
                            datetime.datetime.now().replace(microsecond=0))
            start = datetime.datetime.now().replace(microsecond=0)
            codebook, _distortion = vq.kmeans(descr_array, n_clusters)
            end = datetime.datetime.now().replace(microsecond=0)
        
        time = end - start
        logging.info("\nThe codebook was succesfully created! It took %s "
                     "(H:M:S)\n", time)

        # Save the codebook.
        codebookfilename = filename + "_codebook.file"
        with open(codebookfilename, "wb") as f:
            dump(codebook, f, protocol=HIGHEST_PROTOCOL)

        logging.info("Codebook created and saved to %s", codebookfilename)

        return codebook


class BatchMakeTrainData(MakeTrainData):

    """Generate training data.

    Must be used within a database session scope.
    """

    def __init__(self, config, cache_path):
        """Constructor for training data generator.

        Expects a configurations object `config` and the path to the directory
        where extracted image features are cached `cache_path`.
        """
        super(BatchMakeTrainData, self).__init__(config, cache_path)

        self.taxon_hr = None

        # Set the classification hierarchy.
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

    def batch_export(self, target_dir, codebook_dir=None):
        """Batch export training data to directory `target_dir`."""
        self._load_taxon_hierarchy()

        # Get the name of each level in the classification hierarchy.
        levels = [l.name for l in self.class_hr]

        # Make training data for each path in the classification hierarchy.
        for filter_ in classification_hierarchy_filters(levels, self.taxon_hr):
            level = levels.index(filter_.get('class'))
            train_file = os.path.join(target_dir,
                self.class_hr[level].train_file)
                    
            config = self.class_hr[level]
            
            # Check if a codebook directory is given with existing codebook.
            codebook_file = ""
            if codebook_dir:
                codebook_file = os.path.join(codebook_dir, 
                    self.class_hr[level].train_file)

            # Replace any placeholders in the paths.
            where = filter_.get('where', {})
            for key, val in where.items():
                val = val if val is not None else '_'
                train_file = train_file.replace("__%s__" % key, val)
                codebook_file = codebook_file.replace("__%s__" % key, val)
            
            codebook_file = codebook_file + "_codebook.file"
            if not os.path.isfile(codebook_file):
                codebook_file = None

            # Generate and export the training data.
            logging.info("Exporting train data for classification on %s" % \
                readable_filter(filter_))
            try:
                self.export(train_file, filter_, config, codebook_file)
            except FileExistsError as e:
                # Don't export if the file already exists.
                logging.warning("Skipping: %s" % e)
