#!/usr/bin/env python
"""
Feature extractor with the use of the SURF-algorithm.

Features are extracted from all the images in a 
given path. All processed image filenames are saved
in a .txt file. A dictionary with all descriptors per 
image are saved in a .file file.

See the --help option for information about the 
possible arguments to be parsed.
"""

import argparse
import datetime
import os
import sys

from cPickle import dump, HIGHEST_PROTOCOL
import numpy as np

import cv2


def main():
    # Create argument parser.
    parser = argparse.ArgumentParser(
        description="Image feature extractor.\nFeatures are extracted "
                    "from images in the (subdirectories of the) given path "
                    "with the SURF algorithm of OpenCV.\n"
                    "A dictionary with all features per image is saved to a "
                    "binary '.file' file. A list of all processed images "
                    "is saved to a '.txt' file."
    )
    parser.add_argument(
        "imdir",
        metavar="PATH",
        help="Base directory where the images are stored."
    )
    parser.add_argument(
        "--dictfile",
        metavar="FILE",
        default="Descriptions_dictionary",
        help="Filename to store the feature dictionary in.\n"
             "Defaults to 'Descriptions_dictionary' if omitted.\n"
             "File is stored in current working directory by default. "
             "Any existing file with the same name will be overwritten."
    )
    parser.add_argument(
        "--listfile",
        metavar="FILE",
        default="Imagenames.txt",
        help="Filename to store a list of all processed images.\n"
             "Defaults to 'Imagenames' if omitted.\n"
             "File is stored in current working directory by default. "
             "Any existing file with the same name will be overwritten."
    )
    parser.add_argument(
        "--roi_pix",
        metavar="x,y,w,h",
        help="Region Of Interest, expressed as X,Y,Width,Height in pixel "
             "units. If both roi_pix and roi_frac are given with valid "
             "values, only roi_frac will be used."
    )
    parser.add_argument(
        "--roi_frac",
        metavar="x1,x2,y1,y2",
        help="Region Of Interest, expressed as X1,X2,Y1,Y2 in fractions of "
             "total image size. X1 and Y1 start in left upper corner, X2 goes "
             "to the right, Y2 goes to the bottom. If both roi_pix and "
             "roi_frac are given with valid values, only roi_frac will be "
             "used."
    )
    parser.add_argument(
        "--time",
        metavar="BOOL",
        choices=[True, False],
        default=True,
        help="Show duration of subprocesses and total running time. "
             "Defaults to 'True' if omitted."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Make the image directory path absolute.
    args.imdir = os.path.realpath(args.imdir)

    # Check if the image directory exists.
    if not os.path.exists(args.imdir):
        raise IOError("The given path does not exist.")

    what_roi = check_roi(args)

    # Display time and duration if wanted.
    if args.time:
        starttime = datetime.datetime.now()
        newtime = print_duration(starttime, starttime)

    # Extract features from images.
    descr_dict, file_list = feature_extraction(args, what_roi)
    
    # Display time and duration if wanted.
    if args.time:
        newtime = print_duration(starttime, newtime)

    # Write descriptors and filenames to files.
    print("Writing to files...")
    write_dictionary(args, descr_dict)
    write_list(args, file_list)
    
    # Display time and duration if wanted.
    if args.time:
        print_duration(starttime, newtime)
    print("Finished.")


def print_duration(starttime, previous):
    """
    The function takes two times in datetime format and
    displays the current time and running time of the program.
    The current time is returned.
    """
    current = datetime.datetime.now()
    print("Present time: %s" % current)
    print("Duration of step: %s" % (current - previous))
    print("Total time: %s" % (current - starttime))
    print("\n")
    return current


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


def feature_extraction(args, what_roi):
    """
    The function takes the result of the argument parser.
    For every image in the given path the image is read and the 
    filename is added to a list of processed images. If necessary
    the image is cropped. The image is turned to grayscale and
    truncated to fade the background. The SURF-algorithm is
    used to extract features and the descriptors of the processed
    filename are added to a dictionary.
    When all images are processed, the dictionary with descriptors
    and the list with processed image filenames are returned.
    """
    print("Start feature extraction...")
    descr_dict = {}
    file_list = []
    for root, dirs, files in os.walk(args.imdir):
        for n, filename in enumerate(files):
            if n == 0:
                print("\nCurrent directory: %s" % root)
            filepath = os.path.join(root, filename)
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

            if type(img) != np.ndarray:
                sys.stderr.write("--> %d File %s is not an img: will be skipped.\n" % (n, filename))
                continue
            else:
                print("%d Image %s is being processed..." % (n, filename))
                file_list.append(filename)

            img = apply_roi(args, what_roi, img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_TRUNC)
            des = surf_keypoint_detection(thresh)
            descr_dict[filename] = des

    print("\nAll images have been processed.")
    return descr_dict, file_list


def apply_roi(args, what_roi, img):
    """
    The function takes the result of an argument parser,
    a string to specify what type of roi is wanted and
    an image of type nd.nparray.
    If a valid roi argument is given by the user, the image
    is cropped to the given values.
    The (cropped) image is returned.
    """
    if what_roi == "Fractions":
        roi = split_roi(args.roi_frac, True)
        img = img[int(img.shape[0] * roi[2]):
                  int(img.shape[0] * roi[3]),
                  int(img.shape[1] * roi[0]):
                  int(img.shape[1] * roi[1])]
    elif what_roi == "Pixels":
        roi = split_roi(args.roi_pix, False)
        img = img[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]
    return img


def surf_keypoint_detection(img):
    """
    The function takes an image in grayscale of
    type nd.nparray.
    Features are extracted from the image by the
    SURF-algorithm and keypoints and descriptors are
    calculated. The descriptors are returned.
    """
    surf = cv2.xfeatures2d.SURF_create(510)
    kp, des = surf.detectAndCompute(img, None)
    return des


def write_dictionary(args, dictio):
    """
    The function takes the result of an argument parser 
    and a dictionary with descriptors per image filename.
    The dictionary is written to a given filename
    in a .file format.
    """
    if not args.dictfile.endswith(".file"):
        args.dictfile += ".file"
    with open(args.dictfile, "wb") as f:
        dump(dictio, f, protocol=HIGHEST_PROTOCOL)


def write_list(args, file_list):
    """
    The function takes the result of an argument parser
    and a list with filenames of processed images.
    The filenames in the list are written to a
    given filename in a .txt format.
    """
    if not args.listfile.endswith(".txt"):
        args.listfile += ".txt"
    outputfile = open(args.listfile, 'w')
    for name in file_list:
        outputfile.write(name)
        outputfile.write("\n")
    outputfile.close()

# Call the main function.
main()
