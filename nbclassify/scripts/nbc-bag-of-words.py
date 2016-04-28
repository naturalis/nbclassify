
import argparse
from cPickle import dump, load, HIGHEST_PROTOCOL
import datetime
import os

import numpy as np
import scipy.cluster.vq as vq
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import configure_mappers

import nbclassify.db as db


def main():
    # Create argument parser.
    parser = argparse.ArgumentParser(
        description="Bag-of-words (BOW) creator.\nThe program takes a file "
                    "with a dictionary of image feature descriptors and a "
                    "database with metadata of these images.\n"
                    "The Bag-of-words model is applied for image "
                    "classification. The extracted features are used to "
                    "create a codebook. The vector of occurrence counts "
                    "of each feature of an image in the codebook is "
                    "calculated (=BOW).\nThe codebook is saved as a binary file. "
                    "The list with vectors is saved as a tab-separated file. "
                    "The list of files is used to keep the same order of "
                    "files in the tab-separated file as they appeared in "
                    "the processed folders."
    )
    parser.add_argument(
        "descr_dict",
        metavar="FILE.file",
        help="File with a dictionary of image feature descriptors per image."
    )
    parser.add_argument(
        "meta_file",
        metavar="METAFILE",
        help="Filename of the metadata file (e.g. meta.db)."
    )
    parser.add_argument(
        "--bow",
        metavar="BAG-OF-WORDS-FILE",
        default="BagOfWords",
        help="Output filename to store the vector of occurrence counts of "
             "the features of each image in the codebook. "
             "Defaults to 'BagOfWords' if omitted. "
             "File is stored in current working directory by default. "
             "An existing file with the same name will be overwritten."
    )
    parser.add_argument(
        "--clusters",
        metavar="N",
        help="Number of clusters to create the codebook with. If omitted, "
             "the sqrt of the total number of features will be used."
    )
    parser.add_argument(
        "--codebookfile",
        metavar="CODEBOOK-FILE",
        default="Codebook",
        help="Output filename to store the codebook. "
             "Defaults to 'Codebook' if omitted. "
             "File is stored in current working directory by default. "
             "An existing file with the same name will be overwritten. "
    )
    parser.add_argument(
        "--file_list",
        metavar="FILE.txt",
        help="File with a list of images in the dictionary. File will be "
             "used to order the images in the BOW-file."
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

    # Check if the dictionary file exists.
    if not os.path.isfile(args.descr_dict):
        raise IOError("The given dictionary file does not exist.")

    # Check if the metadata file exists.
    if not os.path.isfile(args.meta_file):
        raise IOError("The given metadata file does not exist.")

    if args.time:
        starttime = datetime.datetime.now()
        newtime = print_duration(starttime, starttime)

    descr_dict = open_dict_file(args)
    if args.time:
        newtime = print_duration(starttime, newtime)

    descr_array = dict2nparray(descr_dict)
    if args.time:
        newtime = print_duration(starttime, newtime)

    codebook = create_codebook(descr_array, args)
    if args.time:
        newtime = print_duration(starttime, newtime)

    bow = create_bow(descr_dict, codebook)
    if args.time:
        newtime = print_duration(starttime, newtime)

    save_codebook(args, codebook)
    if args.time:
        newtime = print_duration(starttime, newtime)

    imglist = open_listfile(args, descr_dict)
    if args.time:
        newtime = print_duration(starttime, newtime)

    save_bow(args, bow, len(codebook), imglist)
    if args.time:
        print_duration(starttime, newtime)

    print("Finished.")


def print_duration(starttime, previous):
    current = datetime.datetime.now()
    print("Present time: %s" % current)
    print("Duration of step: %s" % (current - previous))
    print("Total time: %s" % (current - starttime))
    print("\n")
    return current


def open_dict_file(args):
    print("Reading dictionary file...")
    dict_file = open(args.descr_dict, 'rb')
    descr_dict = load(dict_file)
    dict_file.close()
    return descr_dict


def dict2nparray(descr_dict):
    print("Converting dictionary to nparray...")
    nimages = len(descr_dict)
    descr_array = np.zeros((nimages * 1000, 128))
    position = 0
    for imagename in descr_dict.keys():
        descriptors = descr_dict[imagename]
        nfeatures = descriptors.shape[0]
        while position + nfeatures > descr_array.shape[0]:
            elongation = np.zeros_like(descr_array)
            descr_array = np.vstack((descr_array, elongation))
        descr_array[position: position + nfeatures] = descriptors
        position += nfeatures
    descr_array = np.resize(descr_array, (position, 128))
    return descr_array


def create_codebook(descr_array, args):
    total_features = descr_array.shape[0]
    print("Total number of features: %d" % total_features)
    if args.clusters and args.clusters.isdigit():
        nclusters = int(args.clusters)
    else:
        nclusters = int(np.sqrt(total_features))
    print("Number of clusters: %d" % nclusters)
    print("\nCreating codebook (this will take a while)...")
    codebook, distortion = vq.kmeans(descr_array, nclusters)
    return codebook


def create_bow(descr_dict, codebook):
    print("Creating Bag-of-words...")
    bag_of_words = {}
    for imagename in descr_dict:
        code, dist = vq.vq(descr_dict[imagename], codebook)
        word_hist, bin_edges = np.histogram(code,
                                            bins=range(codebook.shape[0] + 1),
                                            normed=True)
        bag_of_words[imagename] = list(word_hist)
    return bag_of_words


def save_codebook(args, codebook):
    print("Saving codebook...")
    if not args.codebookfile.endswith(".file"):
        args.codebookfile += ".file"
    with open(args.codebookfile, "wb") as f:
        dump(codebook, f, protocol=HIGHEST_PROTOCOL)


def open_listfile(args, descr_dict):
    # Check if the image list file exists.
    if args.file_list and os.path.isfile(args.file_list):
        print("Reading list file...")
        listfile = open(args.file_list, 'r')
        imglist = listfile.readlines()
        listfile.close()
        return imglist
    elif args.file_list:
        sys.stderr.write("The given image list file does not exist. "
                         "Proceeding without order in files.")
    print("Creating image list...")
    imglist = descr_dict.keys()
    return imglist


def save_bow(args, bow, nclusters, imglist):
    print("Saving bag-of-words...")
    if not args.bow.endswith(".tsv"):
        args.bow += ".tsv"
    outputfile = open(args.bow, "w")
    header = make_header(nclusters)
    outputfile.write(header)
    # Connect to database.
    with db.session_scope(args.meta_file) as (session, metadata):
        Base = automap_base(metadata=metadata)
        Base.prepare()
        configure_mappers()
        Photo = Base.classes.photos
        for filename in imglist:
            photo_id = filename.split(".")[0]
            title = photo_id
            for pic in session.query(Photo).filter(Photo.id == photo_id):
                title = photo_id if pic.title is None else pic.title
            valuelist = [photo_id, title]
            words = bow[filename.rstrip()]
            for item in words:
                valuelist.append(str(item))
            row = "\t".join(valuelist)
            row += "\n"
            outputfile.write(row)
    outputfile.close()


def make_header(nclusters):
    headerlist = ["ID", "Title"]
    for x in range(nclusters):
        clustername = "cl" + str(x)
        headerlist.append(clustername)
    header = "\t".join(headerlist)
    header += "\n"
    return header


main()
