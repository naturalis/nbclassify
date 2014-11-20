#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

from nbclassify import conf, open_config
import nbclassify.db as db

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONF_FILE  = os.path.join(BASE_DIR, "config.yml")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
META_FILE = os.path.join(IMAGE_DIR, conf.meta_file)

logging.basicConfig(level=logging.DEBUG)

#@unittest.skip("Debugging")
class TestDatabaseMethods(unittest.TestCase):

    """Unit tests for the database module."""

    @classmethod
    def setUpClass(cls):
        """Create a new metadata file if it does not exist.

        This is executed before any test is started.
        """
        config = open_config(CONF_FILE)
        if not os.path.isfile(META_FILE):
            sys.stdout.write("Initializing database...\n")
            db.make_meta_db(META_FILE)

            with db.session_scope(META_FILE) as (session, metadata):
                mkmeta = db.MakeMeta(config, IMAGE_DIR)
                mkmeta.make(session, metadata)

    def setUp(self):
        """Prepare the testing environment."""
        self.expected_taxa = {
            "40dde798989d9ea3b05140bc218d929a": ['Cypripedium','Obtusipetala','flavum'],
            "ae1cb63196cb236ae27accec4e7861cc": ['Cypripedium','Obtusipetala','flavum'],
            "370652c5f89eb65989c73b04aa9d1130": ['Cypripedium','Obtusipetala','flavum'],
            "9217e5000f2486fd32977b37b394153a": ['Cypripedium','Obtusipetala','flavum'],
            "d1b09d26b512b9c9d979053fdfc07a70": ['Cypripedium','Arietinum','plectrochilum'],
            "48a673b9080ba4bdb5c0581d3367ce8e": ['Cypripedium','Arietinum','plectrochilum'],
            "c900e0e3313b3770e04fea7dea7c56b0": ['Cypripedium','Arietinum','plectrochilum'],
            "7cd52621285f5ddf82cf90334ef12016": ['Cypripedium','Trigonopedia','sichuanense'],
            "e2fb04836e47726e9f46b877b6973e11": ['Cypripedium','Trigonopedia','sichuanense'],
            "5819583c06bfbf5053c3807f78957c5c": ['Cypripedium','Trigonopedia','sichuanense'],
            "0ad1d22dedb39e2e28af335f8de0bfeb": ['Cypripedium','Trigonopedia','fargesii'],
            "e806ebe794745107553f08e2fa223c73": ['Cypripedium','Trigonopedia','fargesii'],
            "b4c6ec5caa1aa340bd159c227de042c4": ['Cypripedium','Trigonopedia','fargesii'],
            "70072d2d0c0e42c9c1417c876895d904": ['Mexipedium',None,'xerophyticum'],
            "55daa2000961186caddd00d1ea48f818": ['Mexipedium',None,'xerophyticum'],
            "ec7e83650743c9dee465aafc5abc4871": ['Mexipedium',None,'xerophyticum'],
            "a7f66ae88910622b8484df23e4830c6d": ['Paphiopedilum','Brachypetalum','wenshanense'],
            "fa86742d507f914bd1dad82256a84c09": ['Paphiopedilum','Brachypetalum','wenshanense'],
            "c0a0b423f58491a386477a934f9889af": ['Paphiopedilum','Brachypetalum','wenshanense'],
            "af17cc52e35199b1bc3d98375e8add2b": ['Paphiopedilum','Brachypetalum','godefroyae'],
            "82562f872e7fc568785b8d5a13f38dff": ['Paphiopedilum','Brachypetalum','godefroyae'],
            "3023792b35e49861769939812eedd4e0": ['Paphiopedilum','Brachypetalum','godefroyae'],
            "11c006917100bcc6a64f0f396ade4abb": ['Paphiopedilum','Brachypetalum','godefroyae'],
            "fb11aa91dd5f11633ac608c51d86edc0": ['Selenipedium',None,'palmifolium'],
            "7f8f42421563be7dc9c753ab7b75453d": ['Selenipedium',None,'palmifolium'],
            "010b61e8a300f92b93536280bcae7657": ['Selenipedium',None,'palmifolium'],
            "b15523eb489824d2ac4be1838cd193bd": ['Phragmipedium','Micropetalum','besseae'],
            "ee014cb617f6a1fc7fd2a7cef261a75d": ['Phragmipedium','Micropetalum','besseae'],
            "59c8640b2494161db4062020b3525224": ['Phragmipedium','Micropetalum','besseae'],
            "2feb25b6467e06080b2df81507b10d0e": ['Phragmipedium','Micropetalum','besseae']
        }

    def test_get_photos_with_taxa(self):
        """Test the get_photos_with_taxa() method."""
        with db.session_scope(META_FILE) as (session, metadata):
            q = db.get_photos_with_taxa(session, metadata)
            ret = q.all()
            self.assertEqual(len(ret), len(self.expected_taxa))
            for photo, genus, section, species in ret:
                class_ = [genus, section, species]
                self.assertEqual(class_, self.expected_taxa[photo.md5sum])

    def test_get_taxon_hierarchy(self):
        """Test the get_taxon_hierarchy() method."""
        expected = {
            u'Paphiopedilum': {
                u'Brachypetalum': [u'godefroyae', u'wenshanense']
            },
            u'Selenipedium': {
                None: [u'palmifolium']
            },
            u'Mexipedium': {
                None: [u'xerophyticum']
            },
            u'Cypripedium': {
                u'Trigonopedia': [u'fargesii', u'sichuanense'],
                u'Obtusipetala': [u'flavum'],
                u'Arietinum': [u'plectrochilum']
            },
            u'Phragmipedium': {
                u'Micropetalum': [u'besseae']
            }
        }

        with db.session_scope(META_FILE) as (session, metadata):
            hier = db.get_taxon_hierarchy(session, metadata)

        self.assertEqual(str(hier), str(expected))

    def test_get_taxa_photo_count(self):
        """Test the get_taxa_photo_count() method."""
        expected = {
            'Cypripedium_Obtusipetala_flavum': 4,
            'Cypripedium_Arietinum_plectrochilum': 3,
            'Cypripedium_Trigonopedia_sichuanense': 3,
            'Cypripedium_Trigonopedia_fargesii': 3,
            'Mexipedium_None_xerophyticum': 3,
            'Paphiopedilum_Brachypetalum_wenshanense': 3,
            'Paphiopedilum_Brachypetalum_godefroyae': 4,
            'Selenipedium_None_palmifolium': 3,
            'Phragmipedium_Micropetalum_besseae': 4
        }

        with db.session_scope(META_FILE) as (session, metadata):
            q = db.get_taxa_photo_count(session, metadata)
            ret = q.all()
            self.assertEqual(len(ret), len(expected))
            for genus, section, species, count in ret:
                class_ = '_'.join([genus, str(section), species])
                self.assertEqual(count, expected[class_])

    def test_get_classes_from_filter(self):
        """Test the get_classes_from_filter() method."""
        filter_genera = {
            'class': 'genus'
        }

        filter_sections = {
            'class': 'section'
        }

        filter_mexi_section = {
            'class': 'section',
            'where': {
                'genus': 'Mexipedium'
            }
        }

        filter_trigo = {
            'class': 'species',
            'where': {
                'genus': 'Cypripedium',
                'section': 'Trigonopedia'
            }
        }

        with db.session_scope(META_FILE) as (session, metadata):
            classes = db.get_classes_from_filter(session, metadata,
                filter_genera)
            self.assertEqual(classes, set(['Cypripedium','Mexipedium',
                'Paphiopedilum','Selenipedium','Phragmipedium']))

            classes = db.get_classes_from_filter(session, metadata,
                filter_sections)
            self.assertEqual(classes, set(['Obtusipetala','Arietinum',
                'Trigonopedia','Brachypetalum','Micropetalum',None]))

            classes = db.get_classes_from_filter(session, metadata,
                filter_mexi_section)
            self.assertEqual(classes, set([None]))

            classes = db.get_classes_from_filter(session, metadata,
                filter_trigo)
            self.assertEqual(classes, set(['sichuanense','fargesii']))

    def test_get_filtered_photos_with_taxon(self):
        """Test the get_filtered_photos_with_taxon() method."""
        filter_mexi_species = {
            'class': 'species',
            'where': {
                'genus': 'Mexipedium',
                'section': None
            }
        }

        filter_mexi_section = {
            'class': 'section',
            'where': {
                'genus': 'Mexipedium'
            }
        }

        filter_trigo = {
            'class': 'species',
            'where': {
                'genus': 'Cypripedium',
                'section': 'Trigonopedia'
            }
        }

        filter_cypr = {
            'class': 'section',
            'where': {
                'genus': 'Cypripedium'
            }
        }

        filter_cypr_none = {
            'class': 'species',
            'where': {
                'genus': 'Cypripedium',
                'section': None
            }
        }

        filter_genera = {
            'class': 'genus'
        }

        with db.session_scope(META_FILE) as (session, metadata):
            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_mexi_species).all()
            self.assertEqual(len(q), 3)
            for photo, class_ in q:
                self.assertEqual(class_, self.expected_taxa[photo.md5sum][2])

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_mexi_section).all()
            self.assertEqual(len(q), 3)
            for photo, class_ in q:
                self.assertEqual(class_, self.expected_taxa[photo.md5sum][1])

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_trigo).all()
            self.assertEqual(len(q), 6)
            taxa = set([taxon for photo, taxon in q])
            self.assertEqual(taxa, set(['fargesii', 'sichuanense']))

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_cypr).all()
            self.assertEqual(len(q), 13)
            taxa = set([taxon for photo, taxon in q])
            self.assertEqual(taxa, set(['Arietinum','Obtusipetala','Trigonopedia']))

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_cypr_none).all()
            self.assertEqual(len(q), 0)

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_genera).all()
            self.assertEqual(len(q), len(self.expected_taxa))
            taxa = set([taxon for photo, taxon in q])
            self.assertEqual(taxa, set(['Cypripedium','Mexipedium',
                'Paphiopedilum','Selenipedium','Phragmipedium']))

if __name__ == '__main__':
    unittest.main()
