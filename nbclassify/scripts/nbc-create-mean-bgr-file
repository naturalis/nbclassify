#!/usr/bin/env python
# -*- conding: utf-8 -*-

"""
This program creates a tab separated file with mean bgr-values
of images.

The program loops over images in a given path and divides every image
in a number of horizontal and vertical bins. For each bin the mean
blue, green and red intensities are calculated, normalized and stored
in a tab separated file. It is possible to determine a
Region Of Interest (ROI) for the images, so only that region will be
used. This can be done in pixel units or by fractions of the total
image size.

The outputfile will be a tab separated file with the following columns:
image ID,
image title,
all horizontal blue bins (B.1.hor, B.2.hor, ...),
all vertical blue bins (B.1.ver, B.2.ver, ...),
all horizontal green bins (G.1.hor, G.2.hor, ...),
all vertical green bins (G.1.ver, G.2.ver, ...),
all horizontal red bins (R.1.hor, R.2.hor, ...),
all vertical red bins (R.1.ver, R.2.ver, ...).

Usage:

  python nbc_create_mean_bgr_file.py PATH OUTPUTFILE METAFILE
  [-h] [--verbose] [--bins N] [--roi_pix x,y,w,h]
  [--roi_frac x1,x2,y1,y2]

PATH is the path to the image directory,
OUTPUTFILE is the filename of the tab separated outputfile and
METAFILE is a database with metadata of the images.

Use the -h option for information about the optional arguments.
"""

import argparse
import cv2
import os
import sys

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import configure_mappers

import imgpheno as ft
import nbclassify.db as db
import numpy as np


def main():
    # Create argument parser.
    parser = argparse.ArgumentParser(
        description="Image histogram maker.\nImages are devided in "
                    "horizontal and vertical bins. The mean blue, green and red "
                    "intensities are calculated for each bin and this data is "
                    "stored in a tab separated file.")

    parser.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where the images are stored.")
    parser.add_argument(
        "outputfile",
        metavar="OUTFILE",
        help="Output filename. An existing file with the same name will be "
             "overwritten.")
    parser.add_argument(
        "meta_file",
        metavar="METAFILE",
        help="File name of the metadata file (e.g. meta.db).")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        const=True,
        help="Explain what is being done.")
    parser.add_argument(
        "--bins",
        metavar="N",
        default=50,
        type=int,
        help="Number of bins the image is devided in horizontally and "
             "vertically. Defaults to 50 if omitted.")
    parser.add_argument(
        "--roi_pix",
        metavar="x,y,w,h",
        help="Region Of Interest, expressed as X,Y,Width,Height in pixel "
             "units. If both roi-pix and roi-frac are given with valid "
             "values, only roi-frac will be used.")
    parser.add_argument(
        "--roi_frac",
        metavar="x1,x2,y1,y2",
        help="Region Of Interest, expressed as X1,X2,Y1,Y2 in fractions of "
             "total image size. X1 and Y1 start in left upper corner, X2 goes "
             "to the right, Y2 goes to the bottom. If both roi-pix and "
             "roi-frac are given with valid values, only roi-frac will be "
             "used."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Make the image directory path absolute.
    args.imdir = os.path.realpath(args.imdir)

    # Check if the image directory exists.
    if not os.path.exists(args.imdir):
        raise IOError("The given path does not exist.")

    # Check if metadata file exists.
    if not os.path.isfile(args.meta_file):
        raise IOError("The given metadata file does not exist.")

    # Check if (a valid) roi argument is given.
    what_roi = check_roi(args)

    # Create a new outputfile.
    create_output_file(args)

    # Connect to database.
    with db.session_scope(args.meta_file) as (session, metadata):
        make_histogram(args, what_roi, session, metadata)


def check_roi(args):
    """
    Check if a valid ROI is given to the argument parser.

    Expects the result of an argument parser.

    If the roi_frac argument is given, this argument is
    transformed to a list of floats in the function 'split_roi'.
    If the list contains four items, all items are between 0 and 1,
    the first item is smaller than the second and the third is
    smaller than the fourth, than the given value is valid and
    will be used to create the ROI. The string 'Fractions' is
    returned. If these conditions are not met, a message will
    be displayed.
    If the roi_pix argument is given, this argument is
    transformed to a list of integers in the function 'split_roi'.
    If the list contains four items, than the given value is valid and
    will be used to create the ROI. The string 'Pixels' is returned.
    If there are not four items in this list, a message will be
    displayed.
    If both arguments are omitted or if both don't have valid values,
    no ROI will be used and 'None' is returned.
    """
    sys.stderr.write("Check if a valid ROI is given...\n\n")
    if args.roi_frac:
        fractions = split_roi(args.roi_frac, True)
        if len(fractions) == 4 and \
                (0 <= fractions[0] <= 1) and (0 <= fractions[1] <= 1) and \
                (0 <= fractions[2] <= 1) and (0 <= fractions[3] <= 1) and \
                (fractions[0] < fractions[1]) and (fractions[2] < fractions[3]):
            sys.stderr.write("The ROI given in fractions will be used.\n\n")
            return "Fractions"
        else:
            sys.stderr.write("WARNING\n"
                             "Four fractions must be given, separated by a "
                             "comma,\nwhere the order is X1,X2,Y1,Y2.\n"
                             "Fractions must be at least 0 and at most 1.\n"
                             "Keep in mind that X1 < X2 and Y1 < Y2 and that\n"
                             "counting starts at the left upper corner.\n"
                             "These conditions weren't met,\nso the "
                             "ROI given in fractions can't be used.\n\n")
    if args.roi_pix:
        pixels = split_roi(args.roi_pix, False)
        if len(pixels) == 4:
            sys.stderr.write("The ROI given in pixels will be used.\n\n")
            return "Pixels"
        else:
            sys.stderr.write("WARNING\n"
                             "Four pixel-values must be given, separated by a"
                             " comma,\nwhere the order is X,Y,Width,Height.\n"
                             "These conditions weren't met,\nso the "
                             "ROI given in pixels can't be used.\n\n")
    sys.stderr.write("Images will be processed without a ROI.\n\n")
    return None


def split_roi(roistring, decimal):
    """
    Transform a string to a list of either integers
    or floats.

    Expects a string with numerical characters,
    separated by a comma and a boolean to determine
    if there are floats or integers in the string.

    The list with either integers or floats is returned.
    """
    roi = roistring.split(",")
    for x in range(len(roi)):
        if decimal:
            roi[x] = float(roi[x])
        else:
            roi[x] = int(roi[x])
    return roi


def create_output_file(args):
    """
    Create a new outputfile with a headerline in it.
    
    The parameter args is the result of an argument parser.
    
    If there was a file with the same filename, that file will
    be overwritten. Otherwise, a file will be created.
    The items in the header are separated by a tab. The header will
    be like: "ID Title B.1.hor B.2.hor .. B.1.ver B.2.ver .. G.1.hor
    G.2.hor .. G.1.ver G.2.ver .. R.1.hor R.2.hor .. R.1.ver R.2.ver"
    .. means continue for the given number of bins.
    """
    if '.' not in args.outputfile:
        args.outputfile += ".tsv"
    filename = open(args.outputfile, 'w')
    headerlist = ['ID', 'Title']
    for color in ['b', 'g', 'r']:
        for direction in ['hor', 'ver']:
            for binnr in range(1, args.bins + 1):
                title = str(color.upper() + "." + str(binnr) + "." + direction)
                # e.g. B.1.hor
                headerlist.append(title)
    header = "\t".join(headerlist)
    header += "\n"
    filename.write(header)
    filename.close()


def make_histogram(args, what_roi, session, metadata):
    """
    Make a histogram of every image and write the values
    to a file.
    
    Expects the result of an argument parser, a string of what kind
    of roi needs to be used and a connection
    to an existing metadata database via an SQLAlchemy Session
    instance 'session' and an SQLAlchemy MetaData instance
    'metadata' which describes the database tables.

    A connection to the database table 'Photos' is made.
    Every file that is an image is opened and the title of that
    image is taken from the database. A mask is created to isolate
    a part of the image (Region Of Interest (ROI)) and a histogram is
    made of that ROI. The values in the histogram-list are
    normalized and relevant data is written to the outputfile.
    """
    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()
    Photo = Base.classes.photos

    # Open outputfile.
    outputfile = open(args.outputfile, 'a')

    # Walk through files.
    for root, dirs, files in os.walk(args.imdir):
        for filename in files:
            sys.stderr.write("File %s is being processed...\n" % filename)

            # Make path to file.
            path = os.path.join(root, filename)

            # Open file and check datatype.
            img = cv2.imread(path, 1)
            if not isinstance(img, np.ndarray):
                sys.stderr.write("File is no image: will be skipped.\n")
                continue

            photo_id = filename.split(".")[0]

            # Get title of image from database.
            # Set default in case there is no database entry for it.
            title = photo_id
            for pic in session.query(Photo).filter(Photo.id == photo_id):
                title = photo_id if pic.title is None else pic.title

            img, contour = create_mask(img, args, what_roi)
            hist = ft.color_bgr_means(img, contour, bins=args.bins)
            means_norm = hist_means(hist)
            write_to_output_file(photo_id, title, means_norm, args, outputfile)

    # Close outputfile.
    outputfile.close()


def create_mask(img, args, what_roi):
    """
    Create a binary mask for an image.

    Expects an image, the result of an argument parser and a string
    of what kind of roi needs to be used.
    
    A binary mask is created for the image. If there is no (valid) ROI
    given, the whole image will be in foreground, otherwise only the
    ROI will be in foreground.

    The masked image and a contour of the ROI will be returned.
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    if what_roi == "Fractions":
        roi = split_roi(args.roi_frac, True)
        mask[int(img.shape[0] * roi[2]):
             int(img.shape[0] * roi[3]),
             int(img.shape[1] * roi[0]):
             int(img.shape[1] * roi[1])] = cv2.GC_FGD
    elif what_roi == "Pixels":
        roi = split_roi(args.roi_pix, False)
        mask[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]] = cv2.GC_FGD
    else:
        mask[0: img.shape[0], 0:img.shape[1]] = cv2.GC_FGD

    # Create a binary mask. Foreground is made white, background black.
    bin_mask = np.where((mask == cv2.GC_FGD) + (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Create a binary mask for the largest contour.
    contour = ft.get_largest_contour(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge the binary mask with the image.
    img_masked = cv2.bitwise_and(img, img, mask=bin_mask)

    return img_masked, contour


def hist_means(hist):
    """
    Normalizes the values in a list and returns a list
    with normalized values.

    Expects a list containing two lists with bgr values. The first
    inner list are values of the horizontal bins, the second
    inner list are values of the vertical bins. In the inner lists
    are the blue green and red values of bin one stored, than
    those of bin two etc.

    Every value is normalized and stored in a list with two items.
    The first is a dictionary for the horizontal bins, the second
    a dictionary for the vertical bins. In the dictionaries the
    colors are separated.
    
    The list with dictionaries is returned.
    """
    # hist looks like: [[hor1B, hor1G, hor1R, hor2B, hor2G, hor2R, etc.],
    #                   [ver1B, ver1G, ver1R, ver2B, ver2G, ver2R, etc.]]
    all_means_norm = [{'b': [], 'g': [], 'r': []},
                      {'b': [], 'g': [], 'r': []}]
    for i in range(len(hist)):
        # i = 0: hor bins; i = 1: ver bins
        # loop over bins
        for x in range(len(hist[i])):
            if x % 3 == 0:
                all_means_norm[i]['b'].append(float(
                    hist[i][x] * 2.0 / 255 - 1))
            elif x % 3 == 1:
                all_means_norm[i]['g'].append(float(
                    hist[i][x] * 2.0 / 255 - 1))
            elif x % 3 == 2:
                all_means_norm[i]['r'].append(float(
                    hist[i][x] * 2.0 / 255 - 1))
    return all_means_norm


def write_to_output_file(photo_id, title, data, args, outputfile):
    """
    One line is written to an already opened file.

    Parameters:
        photo_id:   Integer.
        title:      String.
        data:       List with dictionaries.
        args:       Result of argument parser.
        outputfile: Filename of a file that is opened.

    A string is made with values in correct order for the outputfile,
    separated by tabs, and written to the outputfile.
    """
    # data looks like: [{'b':[], 'g':[], 'r':[]},
    #                   {'b':[], 'g':[], 'r':[]}]
    valuelist = [photo_id, title]
    for color in ['b', 'g', 'r']:
        for direction in range(len(['hor', 'ver'])):
            for binnr in range(args.bins):
                # e.g. B.1.hor
                value = data[direction][color][binnr]
                valuelist.append(str(value))
    row = "\t".join(valuelist)
    row += "\n"

    # Write to outputfile.
    outputfile.write(row)


if __name__ == "__main__":
    main()
