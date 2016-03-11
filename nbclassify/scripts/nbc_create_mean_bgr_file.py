#!/usr/bin/env python
# -*- conding: utf-8 -*-

"""
This program creates a tab separated file with mean bgr-values
of images.

The program loops over images in a given path and divides every
image in a number of horizontal and vertical bins. For each bin
the 

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
        description="Image histogram maker.\nImages are devided in "\
        "horizontal and vertical bins. The mean blue, green and red "\
        "intensities are calculated for each bin and this data is "\
        "stored in a tab separated file.")

    parser.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where the images are stored.")
    parser.add_argument(
        "outputfile",
        metavar="OUTFILE",
        help="Output filename. An existing file with the same name will be "\
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
        "--roi",
        metavar="x,y,w,h",
        help="Region Of Interest, expressed as X,Y,Width,Height in pixel "\
        "units.")
    parser.add_argument(
        "--bins",
        metavar="N",
        default=50,
        type=int,
        help="Number of bins the image is devided in horizontally and "\
        "vertically. If this parameter is omitted, it defaults to 50.")

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
    
    create_output_file(args)

    # Connect to database.
    with db.session_scope(args.meta_file) as (session, metadata):
        make_histogram(args, session, metadata)

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
        args.outputfile = args.outputfile + ".tsv"
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

def make_histogram(args, session, metadata):
    """
    Make a histogram of every image and write the values
    to a file.
    
    Expects the result of an argument parser and a connection
    to an existing metadata database via an SQLAlchemy Session
    instance 'session' and an SQLAlchemy MetaData instance
    'metadata' which describes the database tables.

    A connection to the database table 'Photos' is made.
    Every file that is an image is opened and the title of that
    image is taken from the database. A mask is created to isolate
    a part of the image (region of interest (ROI)) and a histogram is
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
            for pic in session.query(Photo).filter(Photo.id == photo_id):
                title = photo_id if pic.title is None else pic.title
                
            img, contour = create_mask(img, args)
            hist = ft.color_bgr_means(img, contour, bins=args.bins)
            means_norm = hist_means(hist)
            write_to_output_file(photo_id, title, means_norm, args, outputfile)

    # Close outputfile.
    outputfile.close()

def create_mask(img, args):
    """
    Create a binary mask for an image.

    Expects an image and the result of an argument parser.
    
    A binary mask is created for the image. If there is no ROI given,
    the whole image will be in foreground, otherwise only the ROI will
    be in foreground.

    The masked image and a contour of the ROI will be returned.
    """
    mask = np.zeros(img.shape[:2], np.uint8)
    if args.roi and len(args.roi.split(",")) == 4:
        roi = args.roi.split(",")
        mask[int(roi[1]):int(roi[1])+int(roi[3]),
             int(roi[0]):int(roi[0])+int(roi[2])] = cv2.GC_FGD
    else:
        mask[0:img.shape[0], 0:img.shape[1]] = cv2.GC_FGD

    # Create a binary mask. Foreground is made white, background black.
    bin_mask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD),
                        255, 0).astype('uint8')

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
    all_means_norm = [{'b':[], 'g':[], 'r':[]},
                      {'b':[], 'g':[], 'r':[]}]
    for i in range(len(hist)):
        #i = 0: hor bins; i = 1: ver bins
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
    
