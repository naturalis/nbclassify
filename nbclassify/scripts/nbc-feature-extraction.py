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
        "--roi",
        metavar="x,y,w,h",
        help="Region Of Interest, expressed as X,Y,Width,Height in pixel "
             "units."
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

    # Display time and duration if wanted.
    if args.time:
        starttime = datetime.datetime.now()
        newtime = print_duration(starttime, starttime)

    # Extract features from images.
    descr_dict, file_list = feature_extraction(args)
    
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


def feature_extraction(args):
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

            img = check_roi(args, img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_TRUNC)
            des = surf_keypoint_detection(thresh)
            descr_dict[filename] = des

    print("\nAll images have been processed.")
    return descr_dict, file_list


def check_roi(args, img):
    """
    The function takes the result of an argument parser and
    an image of type nd.nparray.
    If the roi argument is given by the user, the image
    is cropped to the given values.
    The (cropped) image is returned.
    """
    if args.roi and len(args.roi.split(",")) == 4:
        roi = args.roi.split(",")
        img = img[int(roi[1]): int(roi[1]) + int(roi[3]),
                  int(roi[0]): int(roi[0]) + int(roi[2])]
    return img


def surf_keypoint_detection(img):
    """
    The function takes an image in grayscale of
    type nd.nparray.
    Features are extracted from the image by the
    SURF-algorithm and keypoints and descriptors are
    calculated. The descriptors are returned.
    """
    surf = cv2.SURF(510)
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
