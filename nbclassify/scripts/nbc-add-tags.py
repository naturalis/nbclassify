#!/usr/bin/env python2.7.11
# -*- coding: utf-8 -*-

"""
Saskia de Vetter
Naturalis Biodiversity Center
2016-02-16  Start building the script; make flickr connection
2016-02-17  Retrieve info from meta data file; get flickr photos
2016-02-18  Loop through flickr pages; add try-except
2016-02-22  Add documentation

This program is made to automatically add tags to pictures
on a flickr account. The information to add to the tags
are harvested from a comma separated file.

The first time this program is run, the webbrowser will
ask to accept the connection that is made with the flickr account.
After this is accepted once, it won't be asked again.

This program can be run in Windows by double clicking on the
filename.
This program can be run in Linux by the command:
    python nbc-add-tags.py
This program requires python 2.7.9 or a higer 2.7 version.
"""

import flickrapi
import sys
import xml.etree.ElementTree

# Flickr API Key for NBClassify.
FLICKR_API_KEY = "6ad8fc70959e3c50eff8339287dd38bd"
FLICKR_API_SECRET = "d7e3a0b378034fa6"

# File with metadata.
META_FILE = "Butterfly_identification_by_Jan_Moonen.csv"


def main():
    """
    The global key, global secret and local user ID are used to
    create a flickr object.
    The function 'retrieve_tag_info' is called to harvest meta data
    from a file.
    The function 'get_pictures' is called to get pictures from the
    flickr object and add tags to it.
    """
    # The Flickr user ID to harvest photos from.
    flickr_uid = "113733456@N06"

    api = FlickrCommunicator(FLICKR_API_KEY, FLICKR_API_SECRET, flickr_uid)
    meta_dict = retrieve_tag_info()
    get_pictures(api, meta_dict, flickr_uid)

def get_pictures(gp_api, gp_meta_dict, gp_flickr_uid):
    """
    Three parameters are passed into this function:
        gp_api:        Flickr object.
        gp_meta_dict:  Dictionary with text for tags.
        gp_flickr_uid: String with flickr user ID.
    In this function default variables are set for pagenumber (pagenr)
    and total number of pages (pages). While the pagenumber is less
    than the total number of pages, photo's of that pagenumber will
    be harvested from the flickr page in batches of 10 at a time.
    The function 'add_tags' is called, in which the tags will be
    added to the pictures of this batch.
    If the pagenumber is one, the correct total number of pages
    is retrieved from the information in the photo-batch.
    Variable 'pages' is now set by this number plus one, otherwise
    pictures in the last page will not be tagged.
    A message is displayed how many pictures were tagged in that
    pagenumber and how many pages there are in total.
    The pagenumber will be increased by one in order to go to
    the next page.
    """
    pagenr = 1
    pages = 2

    while pagenr < pages:
        photos = gp_api.flickr.photos.search(api_key=FLICKR_API_KEY,
                                             user_id=gp_flickr_uid,
                                             format='etree',
                                             per_page='10',
                                             page=pagenr)
        photonumber = add_tags(gp_api, photos, gp_meta_dict)
        if pagenr == 1:
            for information in photos.iter("photos"):
                pages = int(information.get('pages')) + 1
        print(str(photonumber) + " photo's in page " + str(pagenr) +
              " of "  + str(pages) + " pages have been tagged.")
        pagenr += 1
        

def add_tags(at_api, at_photos, at_meta_dict):
    """
    Three parameters are passed into this function:
        at_api:       Flickr object.
        at_photos:    Element object with pictures.
        at_meta_dict: Dictionary with text for tags.
    A variable is set to zero, to count the number of pictures
    to which tags are added.
    For every picture in the Element object, the title of that
    picture is checked to be in the keys of the dictionary.
    If that is the case, the photo-id and tags are put in
    variables and handed over to the flickr method 'addTags'.
    The tags will now be added to this picture and the
    photonumber will be increased by one.
    If the title is not in the keys of the dictionary,
    there are no tags available for that picture and an
    error message will be displayed with this information.
    The number of pictures that were tagged is returned.
    """
    photonumber = 0
    for photo in at_photos.iter("photo"):
        if photo.attrib['title'].startswith('ZMA.INS'):
            title = photo.attrib['title'][8:15]
            if title in at_meta_dict.keys():
                tags = at_meta_dict[title]
                photo_id = photo.attrib['id']
                at_api.flickr.photos.addTags(api_key=FLICKR_API_KEY,
                                             photo_id=photo_id,
                                             tags=tags)
                photonumber += 1
            else:
                sys.stderr.write(title + ' has no meta data.\n')
    return photonumber

def retrieve_tag_info():
    """
    In this function, an attempt will be made to open a .csv-file
    in read modus. The name of the file is global.
    The file is read line by line. For every line, a string will
    be made with the information of that line, combined with the
    header of every column. This string is added to a dictionary as
    a value, the key is the first item of the line: a registration
    number.
    Error messages are raised when the file doesn't exist, there are
    not enough columns in the file or there is no information in
    the file.
    When all went well, a message will be displayed.
    The dictionary with the data will be returned.
    """
    meta_dict = {}
    try:
        file = open(META_FILE, 'r')
        header = file.readline().split(',')
        info = file.readline()
        while info != "":
            info = info.split(',')
            # There was a comma in the name of the identifier,
            # so the split in that name has to be reversed.
            info[8] = str(info[8] + ',' + info.pop(9))[1:-1]
            tag_string = ""
            for item in range(len(header)):
                if not info[item].isspace() and info[item] != "":
                    new_information = make_tag_string(info[item], header[item])
                    tag_string += new_information
            meta_dict[info[0]] = tag_string
            info = file.readline()
        file.close()
    except IOError:
        raise IOError("The file '" + META_FILE +
                      "' for metadata was not found.")
    except IndexError:
        raise IndexError("'" + META_FILE + "' has not enough columns.")
    else:
        if meta_dict == {}:
            raise IndexError("There is no information in '" + META_FILE + "'.")
        else:
            print("The meta data was read successfully.")
    return meta_dict

def make_tag_string(mts_text, mts_header):
    """
    Two parameters are passed into this function:
        mts_text:   String with information.
        mts_header: String with header.
    In this function every '/' will be replaced by a '-' in the text.
    If there is a space in the text, double quotes must surround
    the entire tag information, so they will be added.
    The string with all the tag information is formed, in the
    format <key>:<value>. The key is the header information, the
    value the text. If necessary there will be double quotes
    surrounding the string. A space is put at the end of the string
    (outside the double quotes) to indicate the end of the tag.
    The string with the tag information is returned.
    """
    if '/' in mts_text:
        # To display the date like YYYY-MM-DD.
        mts_text = mts_text.replace('/', '-')
    if " " in mts_text.rstrip():
        # When a space needs to be in the tagvalue,
        # double quotes must surround the entire tag.
        start = '"'
        end = '" '
    else:
        start = ''
        end = ' '
    info_string = str(start + mts_header.rstrip() + ":" +
                      mts_text.rstrip() + end)
    return info_string


class FlickrCommunicator(object):
    """
    In this class, a flickr object is created.
    The authentication is done via browser. The first time
    this connection is made, a browser will open and ask
    to confirm this connection. Every next time the connection
    is made, there won't be a question again.
    """
    def __init__(self, key, secret, uid):
        self.key = key
        self.secret = secret
        self.uid = uid
        self.token = None
        self.frob = None
        self.flickr = flickrapi.FlickrAPI(key, secret)
        self.flickr.authenticate_via_browser(perms='write')


"""
Call the main function if this is the main program.
"""
if __name__ == "__main__":
    main()
