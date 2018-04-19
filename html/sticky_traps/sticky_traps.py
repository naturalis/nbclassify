"""This is the main code used for the sticky trap project"""
# importing modules that are used. not all of them are used at the moment,
# but it is expected that they will be used in the final version.

# import argparse
import logging
import mimetypes
import os
import sys
import common
import cv2
import numpy as np
import yaml
import imgpheno
import time

'''
Code adapted from github.com/Naturalis/imgpheno/examples/train.py
This is a placeholder and will likely change when integrated with
web functionality for large scale testing.
'''
image_list = []
open(os.path.join(os.path.dirname(__file__),"foto_output.txt"), "w").close()
OUTPUT = open(os.path.join(os.path.dirname(__file__),"output.txt"), "a+")


def main():
    """start of program,
    creates parser to obtain the path to images to analyse.
    This code is a placehoder as of yet to accelerate the process of testing. this will change in the final version.
    """
    # Not needed when running as webapplication.
    # # TODO: Change the code that gets the path to the images to either be fully automatic
    # path = "sticky_traps/uploads/"
    # # destination = r"./without"
    # image_files = get_image_paths(path)
    # for img in image_files:
        # analyse_photo(img)


def analyse_photo(img):
    contours, trap, nocorners = find_insects(img)
    output = run_analysis(contours, img)
    if output != "":
        ellipse_img = trap.copy()
        for i in contours:
            if len(i) >= 5:
                ellipse = cv2.fitEllipse(i)
                cv2.ellipse(ellipse_img, ellipse, (0, 0, 255), 1)
        image_list.append(ellipse_img)
    return output, nocorners


def run_analysis(contours, filename):
    """
    does an analysis on the contours found, and returns relevant data
    """
    if contours == None:
        output = ""
    else:
        Foto_output_file = open(os.path.join(os.path.dirname(__file__),"foto_output.txt"), "a+") # creating temporary output file.

    # TODO: have this function automaticially make a file ready for further analysis with R.
    # possibly integrated directly with the webapp.

        properties = imgpheno.contour_properties(contours, ('Area', 'MajorAxisLength',))
        major_axes = [i['MajorAxisLength'] for i in properties]
        smaller_than_4 = [i for i in major_axes if i < 12]
        between_4_and_10 = [i for i in major_axes if i >= 12 and i < 40]
        larger_than_10 = [i for i in major_axes if i >= 40]

        areas = [i['Area'] for i in properties]
        average_area = np.mean(areas)
        number_of_insects = len(contours)

        Foto_output_file.write("""There are %s insects on the trap in %s.
The average area of the insects in %s is %d mm square.
The number of insects smaller than 4mm is %s
The number of insects between 4 and 10 mm is %s
the number of insects larger than 10mm is %s

""" % (number_of_insects, filename, filename, (average_area / 4), len(smaller_than_4),
        len(between_4_and_10), len(larger_than_10)))
        output = {"average_area": average_area/4, "smaller_than_4": len(smaller_than_4), "between_4_and_10": len(between_4_and_10), "larger_than_10": len(larger_than_10), "number_of_insects": number_of_insects}
    return output


def find_insects(img_file):
    """Call all functions in order to analyse the image."""
    nocorners = False
    img = read_img(img_file)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # converts to HSV colourspace for trap roi selection
    """
    note that when displaying hsv as an image using cv2.imshow
    the colours are distorted since imshow assumes bgr colourspace.
    h value of yellow is 30 here.
    """
    mask = hsv_threshold(hsv)  # calls the function that detects the trap based on the HSV image
    corners = imgpheno.find_corners(mask)  # finds the four corners based on an approximation of the contour of the mask.
    #TODO make a config file to edit these values more easily.
    width = 4 * yml.trap_dimensions.Trap_width
    height = 4 * yml.trap_dimensions.Trap_height
    # this height and width must be easily adjustable.
    points_new = np.array([[0, 0], [width, 0], [0, height], [width, height]], np.float32)
    trap = imgpheno.perspective_transform(img, corners, points_new)  # resizes the image.
    if trap is None:  # This code shows the corners returned by find_corners, in case not exactly 4 were returned.
        nocorners = True
        contours = None
        trap = None
        show_corners(corners, img, img_file)
    else:
        nocorners = False
    # after this the program needs to find the insects present on the trap.
        trap = cv2.bilateralFilter(trap, 50, 60, 100) #This eliminates fine texture from the image
        if yml.edges_to_crop:
            trap = crop_image(trap)

        r_channel = trap[:, :, 2]  # selects the channel with the highest contrast
        image_list.append(trap)  # displays the image at the end
        contours = find_contours(r_channel)

        contour_img = trap.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), -1)
        image_list.append(contour_img)

    return contours, trap, nocorners


def get_image_paths(path):
    """Return a list of all images present in the directory 'path'."""
    if not os.path.exists(path):
        logging.error("Cannot open %s (No such file or directory)", path)
        return 1

    images = []

    for item in os.listdir(path):
        imgpath = os.path.join(path, item)
        if os.path.isfile(imgpath):
            mime = mimetypes.guess_type(imgpath)[0]
            if mime and mime.startswith('image'):
                images.append(imgpath)

    if len(images) == 0:
        logging.error("No images found in %s", path)
        return 1
    return images


def read_img(path):
    """
    Read the images into an array generated by opencv2.
    the image is also resized if it is to large
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    perim = sum(img.shape[:2])
    if perim > 1000:
        ref = float(1000) / perim
        img = cv2.resize(img, None, fx=ref, fy=ref)
    return img


# could expand the code so it is more universal, possibly by having target and allowed deviance arguments.
def hsv_threshold(img):
    """The corner detection did not work, I switched to a contour
    finding algorithm. this return the outer contour,
    this will be the sticky trap. the next step will be to find the corners
    using the contour"""
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(img, lower_yellow, upper_yellow)
    return mask


# possible candidate, is only a small wrapper though, could almost be accomplished with one line of code.
def find_contours(image):
    """
    This function returns all contours found in an image using find contours following adaptive thresholding
    """
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 10)
    #finds the contours in the mask of the thresholded image.
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def crop_image(img):
    short_edge = yml.cropping_width.along_short_edges * 4
    long_edge = yml.cropping_width.along_long_edges * 4
    width, height = img.shape[0:2]
    roi = img[short_edge: width - short_edge, long_edge: height - long_edge]
    return roi

def show_corners(corners, img, img_file):
    "shows the corners found on the image."
    for i in corners:
        cv2.circle(img, tuple(i), 5, [0, 0, 255], -1)
    msg = "corners found in " + str(img_file)


def open_yaml(path):
    if not os.path.isfile(path):
        logging.error("Cannot open %s (no such file)" % path)
        return None

    f = open(path, 'r')
    yml = yaml.load(f)
    yml = common.DictObject(yml)
    f.close()

    return yml
yml = open_yaml(os.path.join(os.path.dirname(__file__),"sticky_traps.yml"))

if __name__ == "__main__":
    main()
    write_images("tests", image_list)
