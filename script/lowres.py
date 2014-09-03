#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Print a list of the photos with a low resolution."""

import argparse
from contextlib import contextmanager
import csv
import logging
import os
import sys

import cv2
import sqlalchemy
import sqlalchemy.orm as orm
from sqlalchemy.ext.automap import automap_base

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Setup the argument parser.
    parser = argparse.ArgumentParser(description="Print a list of the photos " \
        "with a low resolution.")
    parser.add_argument("path", metavar="PATH",
        help="Base directory where to look for photos.")
    parser.add_argument("--db", metavar="DB", required=True,
        help="Path to a database file with photo meta data.")
    parser.add_argument("--low", metavar="N", type=int, default=1000,
        help="Photo (width+height) threshold in pixels for a " \
        "low resolution photo. Default is 1000.")

    # Parse arguments.
    args = parser.parse_args()

    # Check resolution of each photo.
    with session_scope(args.db) as (session, metadata):
        photos = get_photos(session, metadata)

    # CSV writer.
    writer = csv.writer(sys.stdout, delimiter='\t', quotechar='"',
        quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["Photo ID", "Title", "Description", "Resolution"])
    for id_, path, title, descr in photos:
        path = os.path.join(args.path, path)
        img = cv2.imread(path)
        if img == None or img.size == 0:
            sys.stderr.write("Cannot open %s (no such file)\n" % path)
            sys.exit(1)

        img_shape = sum(img.shape[:2])
        reso = "{1}x{0}".format(*img.shape)
        if img_shape < args.low:
            writer.writerow([id_, title.encode('utf-8'), descr.encode('utf-8'), reso])

def get_photos(session, metadata):
    Base = automap_base(metadata=metadata)
    Base.prepare()

    # Get the table classes.
    Photos = Base.classes.photos

    # Construct the query.
    q = session.query(Photos.id, Photos.path, Photos.title,
        Photos.description).order_by(Photos.description).\
        filter(Photos.description != None)

    return q

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

if __name__ == "__main__":
    main()
