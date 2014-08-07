#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classify a digital photo using a committee of artificial neural networks.

The photo is classified on different levels in a classification hierarchy,
for example a taxanomic hierarchy. A different neural network may be used to
classify at each level in the hierarchy. The classification hierarchy is
set in a configuration file. See trainer.yml for an example configuration
file with the classification hierarchy set.

The neural networks on which this script depends are created by a separate
script, trainer.py. See `trainer.py batch-data --help` and
`trainer.py batch-ann --help` for more information.

The script also depends on an SQLite database file with meta data for a
collection of digital photographs. This database is created by
harvest-images.py, which is also responsible for compiling the collection of
digital photographs.

See the --help option for more information.
"""

import argparse
from contextlib import contextmanager
import logging
import os
import sys

import cv2
import numpy as np
from pyfann import libfann
import sqlalchemy
import sqlalchemy.orm as orm
from sqlalchemy.ext.automap import automap_base
import yaml

import nbclassify as nbc
# Import the feature extraction library.
# https://github.com/naturalis/feature-extraction
import features as ft

ANSI_COLOR = {
    'green': '\033[32m',
    'green_bold': '\033[1;32m',
    'red': '\033[31m',
    'red_bold': '\033[1;31m',
    'reset': '\033[0m',
}

def main():
    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Classify a digital " \
        "photograph using a committee of artificial neural networks.")

    parser.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a configurations file with the classification " \
        "hierarchy set.")
    parser.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser.add_argument("--anns", metavar="PATH", default=".",
        help="Path to a directory containing the neural networks.")
    parser.add_argument("--error", metavar="N", type=float, default=0.0001,
        help="The maximum error for classification at each level. Default " \
        "is 0.0001. If the maximum error for a level is set in the " \
        "classification hierarchy, then that value is used instead.")
    parser.add_argument("path", metavar="PATH", nargs='?',
        help="Path to the digital photograph to be classified.")
    parser.add_argument("--verbose", "-v", action='store_const',
        const=True, help="Increase verbosity.")

    # Parse arguments.
    args = parser.parse_args()

    # Print debug messages if the -d flag is set for the Python interpreter.
    if sys.flags.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level, format='%(levelname)s %(message)s')

    config = open_yaml(args.conf)
    classifier = ImageClassifier(config, args.db)
    classifier.set_error(args.error)

    try:
        classes, errors = classifier.classify_with_hierarchy(args.path, args.anns)
    except Exception as e:
        logging.error(e)
        return

    # Convert all values to string.
    for i, path in enumerate(classes):
        classes[i] = np.array(path, dtype=str)

    if len(classes) == 1:
        if len(classes[0]) == 0:
            print "Failed to classify the photo."
            return

        table = {
            'color': ANSI_COLOR['green_bold'],
            'reset': ANSI_COLOR['reset'],
            'class': '/'.join(classes[0]),
            'error': sum(errors[0]) / len(errors[0])
        }
        print "Image is classified as {color}{class}{reset} (mse: {error})".format(**table)

    elif len(classes) > 1:
        print "Multiple classifications were returned:"

        # Calculate the mean square error for each classification path.
        errors_classes = [(sum(e)/len(e),c) for e,c in zip(errors, classes)]

        # Print results sorted by error.
        for i, (error, class_) in enumerate(sorted(errors_classes), start=1):
            table = {
                'n': i,
                'color': ANSI_COLOR['green_bold'],
                'reset': ANSI_COLOR['reset'],
                'class': '/'.join(class_),
                'error': error
            }
            print "{n}. {color}{class}{reset} (mse: {error})".format(**table)

@contextmanager
def session_scope(db_path):
    """Provide a transactional scope around a series of operations."""
    engine = sqlalchemy.create_engine('sqlite:///%s' % os.path.abspath(db_path),
        echo=sys.flags.debug)
    Session = orm.sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.MetaData()
    metadata.reflect(bind=engine)
    try:
        yield (session, metadata)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def get_object(d):
    """Return an object from a dictionary."""
    if not isinstance(d, dict):
        raise TypeError("Argument 'd' is not a dictionary")
    return nbc.Struct(d)

def open_yaml(path):
    """Read a YAML file and return as an object."""
    with open(path, 'r') as f:
        config = yaml.load(f)
    return get_object(config)

class Common(object):
    def __init__(self, config):
        self.set_config(config)

    def set_config(self, config):
        """Set the YAML configurations object."""
        if not isinstance(config, nbc.Struct):
            raise TypeError("Configurations object must be of type Struct, not %s" % type(config))

        self.config = config

    def get_codewords(self, classes, on=1, off=-1):
        """Return codewords for a list of classes.

        The codewords are returned as a dictionary in the format ``{class:
        codeword, ..}``, where each class is assigned a codeword.
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

        The result is returned as a sorted 2-list ``[(mse, ..), (class,
        ..)]``. Returns a pair of empty tuples if no classes were found.
        """
        if len(codewords) != len(codeword):
            raise ValueError("Lenth of `codewords` must be equal to `codeword` length")
        classes = []
        for class_, word in codewords.items():
            for i, bit in enumerate(word):
                if bit == on:
                    mse = (float(bit) - codeword[i]) ** 2
                    if mse <= error:
                        classes.append((mse, class_))
                    break
        if len(classes) == 0:
            return [(),()]
        return zip(*sorted(classes))

    def get_taxon_hierarchy(self, session, metadata):
        """Return the taxanomic hierarchies for photos in the database.

        The hierarchy is returned as a dictionary in the format
        ``{genus: {section: [species, ..], ..}, ..}``.
        """
        hierarchy = {}
        taxa = self.get_taxa(session, metadata)

        for genus, section, species in taxa:
            if genus not in hierarchy:
                hierarchy[genus] = {}
            if section not in hierarchy[genus]:
                hierarchy[genus][section] = []
            hierarchy[genus][section].append(species)
        return hierarchy

    def get_taxa(self, session, metadata):
        """Return the taxa from the database.

        Taxa are returned as (genus, section, species) tuples.
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
        q = session.query(Photos.id, 'genus', 'section', 'species').\
            join(stmt1).\
            outerjoin(stmt2).\
            join(stmt3).\
            group_by('genus', 'section', 'species')

        for _, genus, section, species in q:
            yield (genus,section,species)

    def get_childs_from_hierarchy(self, hr, path=[]):
        """Return the child node names for a node in a hierarchy.

        Returns a list of child node names of the hierarchy `hr` at node
        with the path `path`. The hierarchy `hr` is a nested dictionary,
        where bottom level nodes are lists. Which node to get the childs
        from is specified by `path`, which is a list of the node names up
        to that node. An empty list for `path` means the names of the nodes
        of the top level are returned.
        """
        nodes = hr.copy()
        try:
            for name in path:
                nodes = nodes[name]
        except:
            raise ValueError("No such path `%s` in the hierarchy" % '/'.join(path))

        if isinstance(nodes, dict):
            names = nodes.keys()
        elif isinstance(nodes, list):
            names = nodes
        else:
            raise ValueError("Incorrect hierarchy format")

        return names

class ImageClassifier(Common):
    """Classify an image."""

    def __init__(self, config, db_path):
        super(ImageClassifier, self).__init__(config)

        self.set_db_path(db_path)
        self.error = 0.0001
        self.cache = {}

        try:
            self.class_hr = self.config.classification.hierarchy
        except:
            raise ValueError("The configuration file is missing object classification.hierarchy")

        # Get the classification hierarchy from the database.
        with session_scope(db_path) as (session, metadata):
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
            raise ValueError("Attribute `preprocess` not set in the configuration object")
        if 'features' not in config:
            raise ValueError("Attribute `features` not set in the configuration object")

        ann = libfann.neural_net()
        ann.create_from_file(str(ann_path))

        # Create a hash of the feature extraction options.
        hash_ = "%.8s-%.8s" % (hash(config.preprocess), hash(config.features))

        if hash_ in self.cache:
            phenotype = self.cache[hash_]
        else:
            phenotyper = nbc.Phenotyper()
            phenotyper.set_image(im_path)
            phenotyper.set_config(config)
            phenotype = phenotyper.make()

            # Cache the phenotypes, in case they are needed again.
            self.cache[hash_] = phenotype

        logging.info("Using ANN `%s`" % ann_path)
        codeword = ann.run(phenotype)

        return codeword

    def classify_with_hierarchy(self, image_path, ann_base_path=".", path=[], path_error=[]):
        """Start recursive classification.

        Classify the image `image_path` with neural networks from the
        directory `ann_base_path`. The image is classified for each level
        in the classification hierarchy ``classification.hierarchy`` set in
        the configurations file. Each level can use a different neural
        network for classification; the file names for the neural networks
        are set in ``classification.hierarchy[n].ann``. Multiple
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
        levels = [l.name for l in self.class_hr]
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
            # No need to classify if there are no classes set.
            classes = level_classes
            class_errors = (0.0,)
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
            class_errors, classes = self.get_classification(class_codewords,
                codeword, max_error)

            # Test branching.
            #if level.name == 'section':
                #classes += ('Coryopedilum','Brachypetalum')
                #class_errors += (0.0001,0.0002)
            #if level.name == 'species' and path[-1] == 'Paphiopedilum':
                #classes += ('druryi',)
                #class_errors += (0.0003,)
            #if level.name == 'species' and path[-1] == 'Coryopedilum':
                #classes = ()
                #class_errors = ()

        # Print some info messages.
        path_s = [str(p) for p in path]
        path_s = '/'.join(path_s)

        if len(classes) == 0:
            logging.info("Failed to classify on level `%s` at node `/%s`" % (
                level.name,
                path_s)
            )
            return ([path], [path_error])
        elif len(classes) > 1:
            logging.info("Branching in level `%s` at node '/%s' into `%s`" % (
                level.name,
                path_s,
                ', '.join(classes))
            )
        else:
            logging.info("Level `%s` at node `/%s` classified as `%s`" % (
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

if __name__ == "__main__":
    main()
