#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import tempfile
import unittest

from context import db
from context import nbc
from context import nbc_trainer

CONF_FILE  = "config.yml"
IMAGE_DIR = "images"

class TestCase(unittest.TestCase):

    """Super class for test cases."""

    def make_meta_db(self):
        """Create a metadata database for the image directory."""
        self.meta_file = os.path.join(IMAGE_DIR, '.meta.db')
        if os.path.isfile(self.meta_file):
            os.remove(self.meta_file)

        sys.argv = [
            'nbc-trainer.py',
            CONF_FILE,
            'meta',
            IMAGE_DIR
        ]
        ret = nbc_trainer.main()
        self.assertEqual(ret, 0)

class TestCommon(TestCase):

    """Unit tests for the Common class."""

    def setUp(self):
        """Prepare the testing environment."""
        config = nbc.open_config('config.yml')
        self.cmn = nbc.Common(config)

        self.meta_file = os.path.join(IMAGE_DIR, '.meta.db')
        if not os.path.isfile(self.meta_file):
            self.make_meta_db()

    def test_get_taxon_hierarchy(self):
        """Test the get_taxon_hierarchy() method."""
        expected = {
            u'Paphiopedilum': {
                u'Brachypetalum': [u'wenshanense']
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

        with db.session_scope(self.meta_file) as (session, metadata):
            hier = self.cmn.get_taxon_hierarchy(session, metadata)

        self.assertEqual(str(hier), str(expected))

class TestDatabaseMethods(TestCase):

    """Unit tests for the database module."""

    def setUp(self):
        """Prepare the testing environment."""
        self.meta_file = os.path.join(IMAGE_DIR, '.meta.db')
        if not os.path.isfile(self.meta_file):
            self.make_meta_db()

        self.expected_taxa = {
            "40dde798989d9ea3b05140bc218d929a": ['Cypripedium','Obtusipetala','flavum'],
            "ae1cb63196cb236ae27accec4e7861cc": ['Cypripedium','Obtusipetala','flavum'],
            "2da3ae62eb4411af53ac019c01b90039": ['Cypripedium','Obtusipetala','flavum'],
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
            "fb11aa91dd5f11633ac608c51d86edc0": ['Selenipedium',None,'palmifolium'],
            "7f8f42421563be7dc9c753ab7b75453d": ['Selenipedium',None,'palmifolium'],
            "010b61e8a300f92b93536280bcae7657": ['Selenipedium',None,'palmifolium'],
            "b15523eb489824d2ac4be1838cd193bd": ['Phragmipedium','Micropetalum','besseae'],
            "ee014cb617f6a1fc7fd2a7cef261a75d": ['Phragmipedium','Micropetalum','besseae'],
            "59c8640b2494161db4062020b3525224": ['Phragmipedium','Micropetalum','besseae']
        }

    def test_get_photos_with_taxa(self):
        """Test the get_photos_with_taxa() method."""
        with db.session_scope(self.meta_file) as (session, metadata):
            q = db.get_photos_with_taxa(session, metadata)
            ret = q.all()
            self.assertEqual(len(ret), len(self.expected_taxa))
            for photo, genus, section, species in ret:
                class_ = [genus, section, species]
                self.assertEqual(class_, self.expected_taxa[photo.md5sum])

    def test_get_taxa_photo_count(self):
        """Test the get_taxa_photo_count() method."""
        expected = {
            'Cypripedium_Obtusipetala_flavum': 3,
            'Cypripedium_Arietinum_plectrochilum': 3,
            'Cypripedium_Trigonopedia_sichuanense': 3,
            'Cypripedium_Trigonopedia_fargesii': 3,
            'Mexipedium_None_xerophyticum': 3,
            'Paphiopedilum_Brachypetalum_wenshanense': 3,
            'Selenipedium_None_palmifolium': 3,
            'Phragmipedium_Micropetalum_besseae': 3
        }

        with db.session_scope(self.meta_file) as (session, metadata):
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

        with db.session_scope(self.meta_file) as (session, metadata):
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

        with db.session_scope(self.meta_file) as (session, metadata):
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
            for photo, class_ in q:
                self.assertIn(class_, ('fargesii', 'sichuanense'))

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_cypr).all()
            self.assertEqual(len(q), 12)
            for photo, class_ in q:
                self.assertIn(class_, ('Arietinum','Obtusipetala','Trigonopedia'))

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_cypr_none).all()
            self.assertEqual(len(q), 0)

            q = db.get_filtered_photos_with_taxon(session, metadata,
                filter_genera).all()
            self.assertEqual(len(q), len(self.expected_taxa))
            for photo, class_ in q:
                self.assertIn(class_, ('Cypripedium','Mexipedium',
                    'Paphiopedilum','Selenipedium','Phragmipedium'))

if __name__ == '__main__':
    unittest.main()
