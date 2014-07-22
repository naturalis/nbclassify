#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def main():
    # Print debug messages if the -d flag is set for the Python interpreter.
    # Otherwise just show log messages of type INFO.
    if sys.flags.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Classify orchid photos.")
    subparsers = parser.add_subparsers(help="Specify which task to start.")

    # Create an argument parser for sub-command 'classify'.
    parser_classify = subparsers.add_parser('orchid',
        help="Classify the orchid species on a photo.",
        description="Classify an image.")
    parser_classify.add_argument("--conf", metavar="FILE", required=True,
        help="Path to a YAML file with classification configurations.")
    parser_classify.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser_classify.add_argument("--anns", metavar="PATH", default=".",
        help="Path to a directory containing the neural networks.")
    parser_classify.add_argument("--error", metavar="N", type=float,
        default=0.0001,
        help="The maximum error for classification. Default is 0.0001")
    parser_classify.add_argument("path", metavar="PATH", nargs='?',
        help="Path to photo to be classified.")

    # Parse arguments.
    args = parser.parse_args()

    if sys.argv[1] == 'orchid':
        config = open_yaml(args.conf)
        classifier = ImageClassifier(config, args.db)
        classifier.set_error(args.error)
        classification = classifier.classify(args.path, args.anns)
        logging.info("Image is classified as %s" % classification)

    sys.exit()

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
        """Return codewords for a list of classes."""
        n = len(classes)
        codewords = {}
        for i, class_ in enumerate(sorted(classes)):
            cw = [off] * n
            cw[i] = on
            codewords[class_] = cw
        return codewords

    def get_classification(self, codewords, codeword, error=0.01):
        """Return a human-readable classification from a codeword."""
        if len(codewords) != len(codeword):
            raise ValueError("Lenth of `codewords` must be equal to `codeword`")
        classes = []
        for cls, cw in codewords.items():
            for i, code in enumerate(cw):
                if code == 1.0 and (code - codeword[i])**2 < error:
                    classes.append((codeword[i], cls))
        classes = [x[1] for x in sorted(classes, reverse=True)]
        return classes

class ImageClassifier(Common):
    """Classify an image."""

    def __init__(self, config, db_path):
        super(ImageClassifier, self).__init__(config)

        self.set_db_path(db_path)
        self.error = 0.0001
        self.cache = {}

        # Get the classification hierarchy from the database.
        with session_scope(db_path) as (session, metadata):
            self.hierarchy = self.get_taxon_hierarchy(session, metadata)

    def set_db_path(self, path):
        if not os.path.isfile(path):
            raise IOError("Cannot open %s (no such file)" % path)
        self.db_path = path

    def set_error(self, error):
        if not 0 < error < 1:
            raise ValueError("Error must be a value between 0 and 1" % error)
        self.error = error

    def classify(self, image_path, ann_base_path="."):
        try:
            classification_hierarchy = self.config.classification.hierarchy
        except:
            raise ValueError("The configuration file is missing object classification.hierarchy")

        classification = {}
        for level in classification_hierarchy:
            # Replace any wildcards in the ANN path.
            ann_file = level.ann
            for key, val in classification.items():
                if val is None:
                    ann_file = ann_file.replace("__%s__" % key, '_')
                else:
                    ann_file = ann_file.replace("__%s__" % key, val)

            # Get the class names.
            classes = self.get_classes_from_hierarchy(level.name, **classification)

            # Section may not be set and is optional.
            if classes == None and level.name == 'section':
                logging.info("Rank '%s' not set" % level.name)
                classification['section'] = None
                continue
            else:
                assert classes != None, "Classes for level '%s' are not set" % level.name

            # Get the codewords for the classes.
            codewords = self.get_codewords(classes)

            # Classify the image and obtain the codeword.
            ann_path = os.path.join(ann_base_path, ann_file)
            codeword = self.run_ann(image_path, ann_path, level)

            # Get the associated class name for this codeword.
            class_ = self.get_classification(codewords, codeword, self.error)

            if len(class_) == 0:
                logging.info("Failed to classify on level '%s'" % level.name)
                break
            else:
                classification[level.name] = class_[0]
                logging.info("Rank '%s' classified as '%s'" % (level.name, class_[0]))

        return classification

    def get_classes_from_hierarchy(self, rank, genus=None, section=None, species=None):
        """Return a subset of classes from the classification hierarchy.

        Classes for rank `rank` are returned. All the names for the ranks
        up to rank `rank` must be set, for the order `genus`, `section`,
        `species`. Returns None if there are no classes set for a rank.
        """
        if rank == 'species':
            classes = self.hierarchy[genus][section]
        elif rank == 'section':
            classes = self.hierarchy[genus].keys()
        elif rank == 'genus':
            classes = self.hierarchy.keys()
        else:
            raise ValueError("Unknown value for rank")

        if classes == [None]:
            classes = None

        return classes

    def run_ann(self, im_path, ann_path, config):
        if not os.path.isfile(im_path):
            raise IOError("Cannot open %s (no such file)" % im_path)
        if not os.path.isfile(ann_path):
            raise IOError("Cannot open %s (no such file)" % ann_path)

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

            # Keep a record of the phenotypes, in case they are needed again.
            self.cache[hash_] = phenotype

        codeword = ann.run(phenotype)

        return codeword

    def get_taxon_hierarchy(self, session, metadata):
        """Return the taxanomic hierarchies for photos in the database.

        The hierarchy is returned as a dictionary in the format
        ``{genus: {section: [species, ..], ..}, ..}``.
        """
        hierarchy = {}
        q = self.get_taxa(session, metadata)

        for _, genus, section, species in q:
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

        return q

if __name__ == "__main__":
    main()
