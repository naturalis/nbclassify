#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image harvester for downloading photos with meta data from Flickr.

The following classes are defined:

* FlickrDownloader: Download photos and meta data from Flickr.
* ImageHarvester: Organize photos and store meta data in a database.
"""

import argparse
import hashlib
import logging
import os
import re
import sys
from sqlite3 import dbapi2 as sqlite
import urllib
import xml.etree.ElementTree

import flickrapi

# Workaround for flickrapi issue #24
# https://bitbucket.org/sybren/flickrapi/issue/24/
try:
    logging.root.handlers.pop()
except IndexError:
    # Once the bug is fixed in the library the handlers list will be empty,
    # so we need to catch this error.
    pass

# API Key from Offlickr
FLICKR_API_KEY = '1391fcd0a9780b247cd6a101272acf71'
FLICKR_API_SECRET = 'fd221d0336de3b6d'

def main():
    # Print debug messages if the -d flag is set for the Python interpreter.
    # Otherwise just show INFO messages.
    if sys.flags.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # Create argument parser.
    parser = argparse.ArgumentParser(description='Flickr image harvester')
    parser.add_argument("flickr_uid", metavar="FLICKR_UID",
        help="The Flickr user ID to harvest photos from (e.g. 123456789@A12)")
    parser.add_argument("--output", "-o", metavar="PATH", default=".",
        help="Folder to put harvested photos in. Default is current folder.")
    parser.add_argument("--db", metavar="DB",
        help="Path to photos database file with meta data. If omitted, this "\
        "defaults to a file photos.db in the output folder. The database "\
        "file will be created if it doesn't exist.")
    parser.add_argument("--tags", metavar="TAGS",
        help="A space-delimited list of tags. Photos with one or more of " \
        "the tags listed will be harvested. You can exclude results that " \
        "match a term by prepending it with a - character (e.g. \"foo -bar\").")
    parser.add_argument("--tag-mode", metavar="MODE", default='all',
        help="Either 'any' for an OR combination of tags, or 'all' for an " \
        "AND combination. Defaults to 'all' if not specified.")
    parser.add_argument("--page", metavar="N", default=1,
        help="The page of results to return. If this argument is omitted, " \
        "it defaults to 1.")
    parser.add_argument("--per-page", metavar="N", default=100,
        help="Number of photos to return per page. If this argument is " \
        "omitted, it defaults to 100. The maximum allowed value is 500.")

    # Parse arguments.
    args = parser.parse_args()
    if args.db is None:
        args.db = os.path.join(args.output, "photos.db")

    # Initialize Flickr downloader.
    flickr = FlickrDownloader(FLICKR_API_KEY, FLICKR_API_SECRET, args.flickr_uid)

    # Flickr search options.
    search_options = {}
    if args.tag_mode is not None: search_options['tag_mode'] = args.tag_mode
    if args.tags is not None: search_options['tags'] = ",".join(args.tags.split())
    if args.page is not None: search_options['page'] = args.page
    if args.per_page is not None: search_options['per_page'] = args.per_page

    # Download and organize photos.
    harvester = ImageHarvester(flickr, args.db)
    n = harvester.archive_taxon_photos(args.output, **search_options)
    if n > 0:
        logging.info("Finished processing %d photos" % n)
    else:
        logging.info("No photos were found matching your search criteria")

class ImageHarvester(object):
    """Harvest images from a Flickr account."""

    def __init__(self, flickr, db_file):
        self.conn = None
        self.cursor = None
        self.re_filename_replace = re.compile(r'[\s]+')
        self.re_tags_ignore = re.compile(r'vision:')
        self.set_flickr_downloader(flickr)
        self.db_connect(db_file)

    def set_flickr_downloader(self, flickr):
        """Set the Flickr downloader."""
        if isinstance(flickr, FlickrDownloader):
            self.flickr = flickr
        else:
            raise TypeError("Expected an instance of FlickrDownloader")

    def db_connect(self, db_file):
        """Connect to the database, if one already exists.

        Otherwise, create a new database.
        """
        if os.path.isfile(db_file):
            self.conn = sqlite.connect(db_file)
            self.cursor = self.conn.cursor()
        else:
            self.create_new_db(db_file)

    def create_new_db(self, db_file):
        """Create a new database.

        This deletes any existing database file.
        """
        # Check if the folder exists. If not, create it.
        db_dir = os.path.dirname(db_file)
        if not os.path.exists(db_dir):
            logging.info("Creating directory %s" % db_dir)
            os.makedirs(db_dir)

        # Delete the current database file.
        if os.path.isfile(db_file):
            logging.info("Deleting existing database file...")
            self.remove_db()

        logging.info("Creating new database %s..." % db_file)

        # Create a new database.
        self.conn = sqlite.connect(db_file)
        self.cursor = self.conn.cursor()

        # Create the tables.
        self.cursor.executescript("""
        CREATE TABLE photos
        (
            id INTEGER,
            md5sum VARCHAR NOT NULL,
            path VARCHAR,
            title VARCHAR,
            description VARCHAR,

            PRIMARY KEY (id),
            UNIQUE (md5sum),
            UNIQUE (path)
        );

        CREATE TABLE ranks
        (
            id INTEGER,
            name VARCHAR NOT NULL,

            PRIMARY KEY (id),
            UNIQUE (name)
        );

        INSERT INTO ranks (name) VALUES ('domain');
        INSERT INTO ranks (name) VALUES ('kingdom');
        INSERT INTO ranks (name) VALUES ('phylum');
        INSERT INTO ranks (name) VALUES ('class');
        INSERT INTO ranks (name) VALUES ('order');
        INSERT INTO ranks (name) VALUES ('family');
        INSERT INTO ranks (name) VALUES ('genus');
        INSERT INTO ranks (name) VALUES ('subgenus');
        INSERT INTO ranks (name) VALUES ('section');
        INSERT INTO ranks (name) VALUES ('species');
        INSERT INTO ranks (name) VALUES ('subspecies');

        CREATE TABLE taxa
        (
            id INTEGER,
            rank_id INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            description VARCHAR,

            PRIMARY KEY (id),
            UNIQUE (rank_id, name),
            FOREIGN KEY (rank_id) REFERENCES ranks (id) ON DELETE RESTRICT
        );

        CREATE TABLE photos_taxa
        (
            id INTEGER,
            photo_id INTEGER NOT NULL,
            taxon_id INTEGER NOT NULL,

            PRIMARY KEY (id),
            UNIQUE (photo_id, taxon_id),
            FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
            FOREIGN KEY (taxon_id) REFERENCES taxa (id) ON DELETE RESTRICT
        );

        CREATE TABLE tags
        (
            id INTEGER,
            name VARCHAR NOT NULL,

            PRIMARY KEY (id),
            UNIQUE (name)
        );

        CREATE TABLE photos_tags
        (
            id INTEGER,
            photo_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,

            PRIMARY KEY (id),
            UNIQUE (photo_id, tag_id),
            FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE RESTRICT
        );
        """)

        # Commit the transaction.
        self.conn.commit()

    def remove_db(self, db_file, tries=0):
        """Remove existing database file.

        Raises an IOError after three failed deletion attempts.
        """
        if tries > 2:
            raise IOError("Failed to delete database file %s" % db_file)
        try:
            os.remove(db_file)
        except:
            tries += 1
            time.sleep(2)
            self.remove_db(db_file, tries)

    def db_insert_photo(self, photo_info, path, target='.'):
        """Set meta data for a photo in the database.

        Sets the meta data `photo_info` for a photo with file path `path`
        in the database, where `path` should be the path constructed from
        meta data. If `path` is not relative from the current directory,
        `target` should be set to the containing directory, which is
        prepended to `path` to get the photo's real path. By default
        `target` is set to the current directory.

        So if the photo's path is
        ``/path/to/Genus/Subgenus/Section/species/123.jpg``, then `path`
        should be ``Genus/Subgenus/Section/species/123.jpg`` and `target` is
        ``/path/to/``. Only `path` is stored in the database.

        This function checks whether an existing entry in the database
        matches the file's MD5 hash. If they don't match, the file entry is
        deleted and a new one is created.
        """
        real_path = os.path.join(target, path)
        if not os.path.isfile(real_path):
            raise IOError("Cannot open %s (no such file)" % real_path)
        if not xml.etree.ElementTree.iselement(photo_info):
            raise TypeError("Argument `photo_info` is not an xml.etree.ElementTree element")

        # Photo ID must be an integer.
        photo_id = int(photo_info.get('id'))

        # Get the MD5 hash.
        hasher = hashlib.md5()
        with open(real_path, 'rb') as fh:
            buf = fh.read()
            hasher.update(buf)

        # Check if the photo exists in the database.
        self.cursor.execute("SELECT md5sum FROM photos WHERE id=?;", [photo_id])
        md5sum = self.cursor.fetchone()

        # If the photo exists, but the MD5 sums mismatch, delete the record
        # from the database. Otherwise skip this photo.
        if md5sum:
            if md5sum[0] != hasher.hexdigest():
                # Remove the photo record if the MD5 sums don't match.
                logging.warning("MD5 sum mismatch for photo %d; photo record will be updated..." % photo_id)
                self.cursor.execute("DELETE FROM photos WHERE id=?;", [photo_id])
                self.conn.commit()
            else:
                return

        # Get dict of all ranks {name: id, ...}.
        self.cursor.execute("SELECT name, id FROM ranks;")
        ranks = self.cursor.fetchall()
        ranks = dict(ranks)

        # Get meta data.
        title = photo_info.find('title')
        title = None if title.text == '' else title.text
        description = photo_info.find('description')
        description = None if description.text == '' else description.text

        # Insert the photo into the database.
        self.conn.execute("INSERT INTO photos VALUES (?,?,?,?,?);",
            [photo_id, hasher.hexdigest(), path, title, description])

        # Process photo's taxon tags.
        tags = self.flickr_info_get_tags(photo_info, dict)
        for key, val in tags.items():
            # Check if key is a known rank. If not, skip tag.
            rank_id = ranks.get(key)
            if rank_id is None:
                continue

            # Insert the taxon if it doesn't exist.
            taxon_id = self.db_insert_taxon(rank_id, val)

            # Connect the photo to this taxon.
            self.conn.execute("INSERT INTO photos_taxa (photo_id, taxon_id) VALUES (?,?);",
                [photo_id, taxon_id])

        # Commit the transaction.
        self.conn.commit()

        # Set the tags for this photo.
        tags = self.flickr_info_get_tags(photo_info, list)
        self.db_set_photo_tags(photo_id, tags)

    def db_insert_taxon(self, rank_id, taxon_name):
        """Insert a taxon in the database.

        Returns the taxon ID for the rank/taxon combination.
        """
        self.cursor.execute("INSERT OR IGNORE INTO taxa (rank_id, name) VALUES (?,?);",
            [rank_id, taxon_name])
        self.conn.commit()

        self.cursor.execute("SELECT id FROM taxa WHERE rank_id=? AND name=?;",
            [rank_id, taxon_name])
        taxon_id = self.cursor.fetchone()
        return int(taxon_id[0])

    def db_set_photo_tags(self, photo_id, tags):
        """Sets the tags for a photo in the database.

        This method assumes that the tags from `tags` already exist in the
        database. The photo with ID `photo_id` will be linked to the tags.
        """
        self.cursor.execute("DELETE FROM photos_tags WHERE photo_id=?;",
            [photo_id])
        self.cursor.executemany("INSERT INTO photos_tags (photo_id, tag_id) SELECT ?,id FROM tags WHERE name=?;",
            [(photo_id, t) for t in tags])
        self.conn.commit()

    def db_set_tags(self):
        """Sets all Flickr user tags in the database.

        Run this method before calling :meth:`db_set_photo_tags`. Existing
        tags are kept, missing tags are added.
        """
        who = self.flickr.execute('tags.getListUserRaw')
        if who is False:
            raise RuntimeError("Failed to obtain user tags list from Flickr")

        # Construct list of raw tags and filter out unwanted tags.
        tags = [t.find('raw').text for t in who.find('tags')]
        tags = [t for t in tags if not self.re_tags_ignore.match(t)]

        self.cursor.executemany('INSERT OR IGNORE INTO tags (name) VALUES (?);',
            [(t,) for t in tags])
        self.conn.commit()

    def archive_taxon_photos(self, target, **kwargs):
        """Download taxon photos to a taxonomic directory hierarchy.

        Taxon photos are photos with tags with taxonomic information of the
        format ``rank:name``, where `rank` is either ``genus``,
        ``subgenus``, ``section``, or ``species``.

        Downloads photos matching criteria set in `kwargs`. Downloaded
        photos are stored in the following taxonomic directory structure:
        ``Genus/Subgenus/Section/species`` in the target directory
        `target`. Photo meta data is automatically stored in a database.

        Return the number of photos processed.
        """
        if not os.path.isdir(target):
            raise IOError("Cannot open %s (no such directory)" % target)
        if self.conn is None:
            raise RuntimeError("Not connected to a database")
        if self.cursor is None:
            raise RuntimeError("No database cursor set")

        photos = self.flickr.execute('photos.search', **kwargs)
        if photos is False:
            raise RuntimeError("Failed to obtain photo list from Flickr")

        # Make sure that all tags are set in the database.
        self.db_set_tags()

        # Download each photo and set meta data in the database.
        n = 0
        for n, photo in enumerate(photos, start=1):
            photo_id = int(photo.get('id'))
            info = self.flickr.execute('photos.getInfo', photo_id=photo_id)
            if info is False:
                raise RuntimeError("Failed to obtain photo info from Flickr")
            tags = self.flickr_info_get_tags(info, dict)
            ext = info.get('originalformat')

            # Construct the save path for the photo.
            photo_dir = os.path.join(
                tags.get('genus', 'genus_null'),
                tags.get('subgenus', 'subgenus_null'),
                tags.get('section', 'section_null'),
                tags.get('species', 'species_null')
            )

            filename = "%s.%s" % (photo_id, ext)
            filename = self.re_filename_replace.sub('-', filename)
            photo_path = os.path.join(photo_dir, filename)

            real_photo_dir = os.path.join(target, photo_dir)
            real_path = os.path.join(real_photo_dir, filename)

            # Check if the folder exists. If not, create it.
            if not os.path.exists(real_photo_dir):
                logging.info("Creating directory %s" % real_photo_dir)
                os.makedirs(real_photo_dir)

            # Download the photo.
            if not os.path.isfile(real_path):
                logging.info("Downloading photo %s to %s ..." % (photo_id, real_path))
                self.flickr.download_photo(photo_id, real_path)
            else:
                logging.info("Photo %s already exists. Skipping download." % (photo_id))

            # Insert the photo into the database.
            self.db_insert_photo(info, photo_path, target)

        return n

    def flickr_info_get_tags(self, info, format=list):
        """Returns the tags from a photo info object.

        If `format` is ``list``, the raw tag values are returned in a list.
        If `format` is ``dict``, the tags are returned as a dictionary. Tags
        of the format ``key:value`` are stored as a key:value pair in
        the dictionary. Otherwise the key will be the raw tag value, and the
        corresponding value will be None.
        """
        if not xml.etree.ElementTree.iselement(info):
            raise TypeError("Argument `info` is not an xml.etree.ElementTree element")
        if format is list:
            out = []
        elif format is dict:
            out = {}
        else:
            raise ValueError("Unkown format '%s'" % format)

        tags = info.find('tags')
        for tag in tags:
            raw = tag.get('raw').strip()
            if format is list:
                out.append(raw)
            else:
                e = raw.split(':')
                if len(e) == 2:
                    out[e[0]] = e[1]
                else:
                    out[raw] = None
        return out

class FlickrDownloader(object):
    """Download photos with metadata from Flickr."""

    def __init__(self, key, secret, uid):
        self.key = key
        self.secret = secret
        self.uid = uid
        self.token = None
        self.frob = None
        self.api = flickrapi.FlickrAPI(key, secret, format='etree')

        # Flickr's two-phase authentication.
        self.token, self.frob = self.api.get_token_part_one(perms='read')

        if self.token:
            # We have a token, but it might not be valid.
            logging.info("Flickr token found")
            try:
                self.api.auth_checkToken()
            except flickrapi.FlickrError:
                self.token = None

        if not self.token:
            # No valid token, so redirect to Flickr.
            logging.info("Please authorize this program via Flickr. Redirecting to Flickr...")
            raw_input("Press ENTER after you authorized this program")

        self.api.get_token_part_two((self.token, self.frob))
        logging.info("Flickr authorization success")

    def execute(self, method, *args, **kwargs):
        """Execute a method of the Flickr API.

        The method name `method` can be followed by the method specific
        arguments. Returns the result or True on success, False otherwise.

        The Flickr method arguments `api_key` and `user_id` are automatically
        added and can be omitted when calling this method.
        """
        if self.token is None:
            raise RuntimeError("Flickr token not set")
        if not isinstance(method, str):
            raise TypeError("Argument `method` must be a string")
        try:
            m = method.replace('.', '_')
            m = re.sub("^flickr_", "", m)
            m = getattr(self.api, m)
        except AttributeError:
            raise AttributeError("Flickr API method '%s' not found" % method)
        rsp = m(api_key=self.key, user_id=self.uid, *args, **kwargs)
        if rsp.get('stat') != 'ok':
            logging.error("Method '%' failed" % method)
            return False
        if len(rsp) > 0:
            return rsp[0]
        else:
            return True

    def get_photo_urls(self, photo_id):
        """Return the URLs for the photo.

        URLs are returned as a dictionary in the format ``{'label':
        'url', ..}``, where ``label`` is one of the available photo sizes as
        returned by ``flickr.photos.getSizes``.
        """
        urls = {}
        sizes = self.execute('photos.getSizes', photo_id=photo_id)
        if sizes is False:
            raise AttributeError("Failed to get sizes for photo %s" % photo_id)
        for size in sizes:
            label = size.get('label')
            source = size.get('source')
            if label is not None:
                urls[label] = source
        return urls

    def download_photo(self, photo_id, path, size='Original'):
        """Download a photo to a file.

        Downloads the photo with ID `photo_id` and size `size` to a file
        `path`. Returns None if the photo can't be downloaded.
        """
        urls = self.get_photo_urls(photo_id)
        url = urls.get(size)
        if url is None:
            logging.error("No URL found for photo %s with size '%s'" % (photo_id, size))
            return None
        tmpfile = "%s.part" % path
        urllib.urlretrieve(url, tmpfile)
        os.rename(tmpfile, path)

if __name__ == "__main__":
    main()
