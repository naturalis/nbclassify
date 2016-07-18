# -*- coding: utf-8 -*-

"""Database routines.

Uses SQLAlchemy for object relational mapping.
"""

from contextlib import contextmanager
import hashlib
import os
import re
import sys

import cv2
import sqlalchemy
from sqlalchemy import Column, ForeignKey, Integer, Sequence, String, \
    UniqueConstraint
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, configure_mappers
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql import functions

from . import conf
from .exceptions import *
from .functions import Struct, get_childs_from_hierarchy, path_from_filter


def get_classes_from_filter(session, metadata, filter_):
    """Return the classes for a classification filter.

    The unique set of classes for the classification filter `filter_` are
    returned. Filters are those as returned by
    :meth:`~nbclassify.functions.classification_hierarchy_filters`.
    """
    if not isinstance(filter_, dict):
        ValueError("Expected a dict as filter")

    class_ = filter_.get('class')
    q = get_filtered_photos_with_taxon(session, metadata, filter_)
    q = q.group_by(class_)
    classes = [class_ for photo,class_ in q]
    return set(classes)

def get_filtered_photos_with_taxon(session, metadata, filter_):
    """Return photos with corresponding class for a filter.

    Returns all photos with corresponding taxon, as filterd by `filter_`. The
    taxon returned per photo is defined by the `class` attribute of the filter.
    Taxa to filter photos by is set in the `where` attribute of the filter.
    Filters are those as returned by
    :meth:`~nbclassify.functions.classification_hierarchy_filters`. Returned
    rows are 2-tuples ``(photo, taxon_name)``.
    """
    if not isinstance(filter_, dict):
        ValueError("Expected a dict as filter")

    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()

    # Get the table classes.
    Photo = Base.classes.photos
    Taxon = Base.classes.taxa
    Rank = Base.classes.ranks

    # Use a subquery because we want photos to be returned even if the don't
    # have a taxa for the given class.
    class_ = filter_.get('class')
    stmt_genus = session.query(Photo.id, Taxon.name.label('genus')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'genus').subquery()
    stmt_section = session.query(Photo.id, Taxon.name.label('section')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'section').subquery()
    stmt_species = session.query(Photo.id, Taxon.name.label('species')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'species').subquery()

    # Construct the main query.
    q = session.query(Photo, class_).\
        join(stmt_genus, stmt_genus.c.id == Photo.id).\
        outerjoin(stmt_section, stmt_section.c.id == Photo.id).\
        join(stmt_species, stmt_species.c.id == Photo.id)

    # Filter on each taxon in the where attribute of the filter.
    where = filter_.get('where', {})
    for rank_name, taxon_name in where.items():
        if rank_name == 'genus':
            q = q.filter(stmt_genus.c.genus == taxon_name)
        elif rank_name == 'section':
            q = q.filter(stmt_section.c.section == taxon_name)
        elif rank_name == 'species':
            q = q.filter(stmt_species.c.species == taxon_name)

    return q

def get_photos(session, metadata):
    """Return photo records from the database."""
    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()
    Photo = Base.classes.photos
    photos = session.query(Photo)
    return photos

def get_photos_with_taxa(session, metadata):
    """Return photos with genus, section, and species class.

    This generator returns 4-tuples ``(photo, genus, section, species)``.
    """
    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()

    Photo = Base.classes.photos
    Taxon = Base.classes.taxa
    Rank = Base.classes.ranks

    stmt_genus = session.query(Photo.id, Taxon.name.label('genus')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'genus').subquery()

    stmt_section = session.query(Photo.id, Taxon.name.label('section')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'section').subquery()

    stmt_species = session.query(Photo.id, Taxon.name.label('species')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'species').subquery()

    q = session.query(Photo, 'genus', 'section', 'species').\
        join(stmt_genus, stmt_genus.c.id == Photo.id).\
        outerjoin(stmt_section, stmt_section.c.id == Photo.id).\
        join(stmt_species, stmt_species.c.id == Photo.id)

    return q

def get_session_or_error():
    """Return the database session and metadata objects.

    Returns a ``(session, metadata)`` tuple. Raises a DatabaseSessionError
    exception if not in a database session is set.
    """
    if (conf.session and conf.metadata):
        return (conf.session, conf.metadata)
    else:
        raise DatabaseSessionError("Not in a database session")

def get_taxa_photo_count(session, metadata):
    """Return the photo count for each (genus, section, species) combination.

    Taxa are returned as 4-tuples ``(genus, section, species, photo_count)``.
    """
    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()

    Photo = Base.classes.photos
    Taxon = Base.classes.taxa
    Rank = Base.classes.ranks

    stmt_genus = session.query(Photo.id, Taxon.name.label('genus')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'genus').subquery()

    stmt_section = session.query(Photo.id, Taxon.name.label('section')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'section').subquery()

    stmt_species = session.query(Photo.id, Taxon.name.label('species')).\
        join(Photo.taxa_collection, Taxon.ranks).\
        filter(Rank.name == 'species').subquery()

    q = session.query('genus', 'section', 'species',
            functions.count(Photo.id).label('photos')).\
        select_from(Photo).\
        join(stmt_genus, stmt_genus.c.id == Photo.id).\
        outerjoin(stmt_section, stmt_section.c.id == Photo.id).\
        join(stmt_species, stmt_species.c.id == Photo.id).\
        group_by('genus', 'section', 'species')

    return q

def get_taxon_hierarchy(session, metadata):
    """Return the taxanomic hierarchy for photos in the metadata database.

    The hierarchy is returned as a dictionary in the format ``{genus: {section:
    [species, ..], ..}, ..}``. If the global configuration
    ``nbclassify.conf.photo_count_min`` is set to a positive value, only taxa
    with a minimum photo count of `photo_count_min` are used to construct the
    hierarchy.

    Returned hierarchies can be used as input for methods like
    :meth:`~nbclassify.functions.classification_hierarchy_filters` and
    :meth:`~nbclassify.functions.get_childs_from_hierarchy`.
    """
    hierarchy = {}
    for genus, section, species, count in get_taxa_photo_count(session, metadata):
        if conf.photo_count_min and count < conf.photo_count_min:
            continue
        if genus not in hierarchy:
            hierarchy[genus] = {}
        if section not in hierarchy[genus]:
            hierarchy[genus][section] = []
        hierarchy[genus][section].append(species)
    return hierarchy

def insert_new_photo(session, metadata, root, path, update=False, **kwargs):
    """Set meta data for a photo in the database.

    Sets meta data for a photo with `path` that is relative from the top image
    directory `root`. So ``os.path.join(root, path)`` should point to and actual
    image file. Only `path` is stored in the database. If `update` is set to
    True, an existing record for this photo is updated, otherwise a ValueError
    is raised.

    This function checks whether an existing photo entry in the database matches
    the file's MD5 hash. If one is found, the photo entry is deleted and a new
    one is created.

    Optional keyword arguments `title`, `description`, `taxa`, and `tags` can
    be passed to provide additional meta data. Here `title` and `description`
    provide a title and a description for the photo. Taxonomic classifications
    are provided as a dict `taxa` of the format ``{rank: taxon, ...}``. Photo
    tags can be provided as a list `tags`. If an argument `id` is provided,
    this is used as the unique identifier for the photo, which must be an
    integer.
    """
    real_path = os.path.join(root, path)
    if not os.path.isfile(real_path):
        raise IOError("Cannot open %s (no such file)" % real_path)

    # Get meta data from the arguments.
    photo_id = None
    title = None
    description = None
    taxa = {}
    tags = []
    for key, val in kwargs.items():
        if val is None:
            continue

        if key == 'id':
            photo_id = int(val)
        elif key == 'title':
            title = val
        elif key == 'description':
            description = val
        elif key == 'taxa':
            taxa = dict(val)
        elif key == 'tags':
            tags = list(val)
        else:
            ValueError("Unknown keyword argument `%s`" % key)

    # Get the database models.
    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()

    Photo = Base.classes.photos
    Rank = Base.classes.ranks
    Taxon = Base.classes.taxa
    Tag = Base.classes.tags

    # Get the MD5 hash.
    hasher = hashlib.md5()
    with open(real_path, 'rb') as fh:
        buf = fh.read()
        hasher.update(buf)

    # Check if a photo with the same MD5 sum exists in the database.
    try:
        photo = session.query(Photo).\
            filter(Photo.md5sum == hasher.hexdigest()).one()
    except NoResultFound:
        photo = None

    if photo:
        if not update:
            raise ValueError("Found existing photo {0} with matching MD5 sum {1}".\
                format(photo.path, hasher.hexdigest()))
        else:
            session.delete(photo)

    # Check if a photo with the same ID exists in the database.
    if photo_id:
        try:
            photo = session.query(Photo).\
                filter(Photo.id == photo_id).one()
        except NoResultFound:
            photo = None

        if photo:
            if not update:
                raise ValueError("Found existing photo {0} with matching ID {1}".\
                    format(photo.path, photo_id))
            else:
                session.delete(photo)

    # Insert the photo into the database.
    photo = Photo(
        md5sum=hasher.hexdigest(),
        path=path,
        title=title,
        description=description
    )

    # Overwrite the ID if the photo ID was provided.
    if photo_id:
        photo.id = photo_id

    # Save photo's taxa to the database.
    processed_ranks = []
    for rank_name, taxon_name in taxa.items():
        # Skip ranks that evaluate to False.
        if not rank_name:
            continue

        # Skip if the taxon is not set.
        if not taxon_name:
            continue

        # Get a rank instance.
        try:
            rank = session.query(Rank).\
                filter(Rank.name == rank_name).one()
        except NoResultFound:
            rank = Rank(name=rank_name)

        # Get a taxon instance.
        try:
            taxon = session.query(Taxon).\
                filter(Taxon.ranks == rank,
                       Taxon.name == taxon_name).one()
        except NoResultFound:
            taxon = Taxon(name=taxon_name, ranks=rank)

        # Add this taxon to the photo.
        photo.taxa_collection.append(taxon)

        # Keep track of processed ranks.
        processed_ranks.append(rank.name)

    # Make sure that the required ranks are set for each photo.
    if conf.required_ranks:
        assert set(conf.required_ranks).issubset(processed_ranks), \
            "Every photo must at least have the ranks {0}".\
                format(conf.required_ranks)

    # Set the tags for this photo.
    for tag_name in tags:
        if not tag_name:
            continue

        # Get a tag instance.
        try:
            tag = session.query(Tag).\
                filter(Tag.name == tag_name).one()
        except NoResultFound:
            tag = Tag(name=tag_name)

        # Add this tag to the photo.
        photo.tags_collection.append(tag)

    # Add photo to session.
    session.add(photo)

def make_meta_db(db_path):
    """Create a new metadata SQLite database `db_path`.

    This database holds records of the photos in a directory, along with
    their taxonomic classifications.
    """
    if os.path.isfile(db_path):
        raise FileExistsError(db_path)

    engine = sqlalchemy.create_engine('sqlite:///{0}'.format(db_path),
        echo=conf.orm_verbose)

    Base = declarative_base()

    class Photo(Base):

        """Photo records.

        SQL::

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
        """

        __tablename__ = 'photos'

        id = Column(Integer, Sequence("photos_id_seq"), primary_key=True)
        md5sum = Column(String(32), index=True, unique=True, nullable=False)
        path = Column(String(255), unique=True)
        title = Column(String(100))
        description = Column(String(255))

    class Rank(Base):

        """Taxonomic ranks.

        SQL::

            CREATE TABLE ranks
            (
                id INTEGER,
                name VARCHAR NOT NULL,

                PRIMARY KEY (id),
                UNIQUE (name)
            );
        """

        __tablename__ = 'ranks'

        id = Column(Integer, Sequence("ranks_id_seq"), primary_key=True)
        name = Column(String(50), index=True, unique=True, nullable=False)

    class Taxon(Base):

        """Taxonomic categories.

        Because taxonomic names can be used in multiple ranks, each taxon is
        linked to a specific rank.

        SQL::

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
        """

        __tablename__ = 'taxa'

        id = Column(Integer, Sequence("taxa_id_seq"), primary_key=True)
        rank_id = Column(Integer, ForeignKey('ranks.id', ondelete="RESTRICT"),
            nullable=False)
        name = Column(String(50), nullable=False)
        description = Column(String(255))

        UniqueConstraint('rank_id', 'name')

    class PhotoTaxon(Base):

        """Establish photos-taxa relationships.

        SQL::

            CREATE TABLE photos_taxa
            (
                photo_id INTEGER NOT NULL,
                taxon_id INTEGER NOT NULL,

                PRIMARY KEY (photo_id, taxon_id),
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
                FOREIGN KEY (taxon_id) REFERENCES taxa (id) ON DELETE RESTRICT
            );
        """

        __tablename__ = 'photos_taxa'

        photo_id = Column(Integer, ForeignKey('photos.id', ondelete="CASCADE"),
            primary_key=True, nullable=False)
        taxon_id = Column(Integer, ForeignKey('taxa.id', ondelete="RESTRICT"),
            primary_key=True, nullable=False)

    class Tag(Base):

        """Photo tags.

        SQL::

            CREATE TABLE tags
            (
                id INTEGER,
                name VARCHAR NOT NULL,

                PRIMARY KEY (id),
                UNIQUE (name)
            );
        """

        __tablename__ = 'tags'

        id = Column(Integer, Sequence("tags_id_seq"), primary_key=True)
        name = Column(String(50), index=True, unique=True, nullable=False)

    class PhotoTag(Base):

        """Establish photos-tags relationships.

        SQL::

            CREATE TABLE photos_tags
            (
                photo_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,

                PRIMARY KEY (photo_id, tag_id),
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE RESTRICT
            );
        """

        __tablename__ = 'photos_tags'

        photo_id = Column(Integer, ForeignKey('photos.id', ondelete="CASCADE"),
            primary_key=True, nullable=False)
        tag_id = Column(Integer, ForeignKey('tags.id', ondelete="RESTRICT"),
            primary_key=True, nullable=False)

    # Create the database.
    Base.metadata.create_all(engine)

@contextmanager
def session_scope(db_path):
    """Provide a transactional scope around a series of operations.

    This is a factory function for ``with`` statements that yields a 2-tuple
    ``(session, metadata)`` for the SQLite database `db_path`.
    """
    if conf.session:
        raise RuntimeError("Only one database session is allowed at a time")

    engine = sqlalchemy.create_engine('sqlite:///{0}'.format(db_path),
        echo=conf.orm_verbose)
    Session = sessionmaker(bind=engine)
    conf.session = Session()
    conf.metadata = sqlalchemy.MetaData()
    conf.metadata.reflect(bind=engine)
    try:
        yield (conf.session, conf.metadata)
        conf.session.commit()
    except:
        conf.session.rollback()
        raise
    finally:
        conf.session.close()
        conf.session = conf.metadata = None

def set_default_ranks(session, metadata):
    """Populate the ranks table."""
    ranks = ('domain', 'kingdom', 'phylum', 'class', 'order', 'family',
    'genus', 'subgenus', 'section', 'species', 'subspecies')

    Base = automap_base(metadata=metadata)
    Base.prepare()
    configure_mappers()
    Rank = Base.classes.ranks

    for name in ranks:
        rank = Rank(name=name)
        session.add(rank)

    session.commit()


class MakeMeta(object):

    """Populate a meta data database for an image directory.

    The images in the image directory must be stored in a directory hierarchy
    which corresponds to the directory hierarchy set in the configurations. The
    meta data is a single database file created in the image directory. If a
    meta data file already exists, a FileExistsError is raised.
    """

    def __init__(self, config, image_dir):
        """Expects a configurations object `config` and a path to the directory
        containing the images `image_dir`.
        """
        self.ranks = []
        self.set_config(config)
        self.set_image_dir(image_dir)

        try:
            hr = list(self.config.directory_hierarchy)
        except:
            raise ConfigurationError("directory hierarchy is not set")
        self.set_ranks(hr)

    def set_config(self, config):
        """Set the configurations object `config`."""
        if not isinstance(config, Struct):
            raise TypeError("Configurations object must be of type Struct, " \
                "not %s" % type(config))
        self.config = config

    def set_image_dir(self, path):
        """Set the image directory."""
        if not os.path.isdir(path):
            raise IOError("Cannot open %s (no such directory)" % path)
        self.image_dir = os.path.abspath(path)

    def set_ranks(self, hr):
        """Set the ranks from the directory hierarchy `hr`."""
        self.ranks = []
        for rank in hr:
            if rank == "__ignore__":
                rank = None
            self.ranks.append(rank)

    def get_image_files(self, root, ranks, classes=[]):
        """Return image paths and their classes.

        Images are returned as 2-tuples ``(path, {rank: class, ...})`` where
        each class is the directory name for each rank in `ranks`, a list of
        ranks. List `classes` is used internally to keep track of the classes.
        """
        if len(classes) > len(ranks):
            return

        for item in os.listdir(root):
            path = os.path.join(root, item)
            if os.path.isdir(path):
                # The current directory name is the class name.
                class_ = os.path.basename(path.strip(os.sep))
                if class_ in ("None", "NULL", "_"):
                    class_ = None
                for image in self.get_image_files(path, ranks, classes+[class_]):
                    yield image
            elif os.path.isfile(path) and classes:
                yield (path, dict(zip(ranks, classes)))

    def make(self, session, metadata):
        """Create the meta data database file `meta_path`."""
        sys.stdout.write("Setting taxonomic ranks...\n")
        set_default_ranks(session, metadata)

        sys.stdout.write("Setting meta data for images...\n")
        for path, classes in self.get_image_files(self.image_dir, self.ranks):
            # Get the path relative to self.image_dir
            path_rel = re.sub(self.image_dir, "", path)
            if path_rel.startswith(os.sep):
                path_rel = path_rel[1:]
            
            if type(cv2.imread(path)) == np.ndarray:
                # Save the meta data only if it is an image.
                insert_new_photo(session, metadata,
                    root=self.image_dir,
                    path=path_rel,
                    taxa=classes)
            else:
                sys.stdout.write("%s is not an image: will be skipped.\n" % path)
                continue

        sys.stdout.write("Done\n")
