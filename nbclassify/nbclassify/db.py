import os
import hashlib
import sys
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import Column, ForeignKey, Integer, Sequence, String, \
    UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.automap import automap_base

from exceptions import *

# Every photo must have the following ranks set in the meta data.
REQUIRED_RANKS = ['genus','species']

@contextmanager
def session_scope(db_path):
    """Provide a transactional scope around a series of operations.

    This is a decorator that yields a 2-tuple ``(session, metadata)`` for the
    SQLite database `db_path`.
    """
    engine = sqlalchemy.create_engine('sqlite:///{0}'.\
        format(os.path.abspath(db_path)),
        echo=sys.flags.debug)
    Session = sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.MetaData()
    metadata.reflect(bind=engine)
    try:
        yield (session, metadata)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def make_meta_db(db_path):
    """Create a new metadata SQLite database `db_path`.

    This database holds records of the photos in a directory, along with
    their taxonomic classifications.
    """
    if os.path.isfile(db_path):
        raise FileExistsError(db_path)

    engine = sqlalchemy.create_engine('sqlite:///{0}'.\
        format(os.path.abspath(db_path)),
        echo=sys.flags.debug)

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

        def __repr__(self):
           return "<{class}(id='{id}', title='{title}', path='{path}')>".\
                format({
                    'class': self.__class__,
                    'id': self.id,
                    'title': self.title,
                    'path': self.path
                })

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

        def __repr__(self):
           return "<{class}(id='{id}', name='{name}')>".\
                format({
                    'class': self.__class__,
                    'id': self.id,
                    'name': self.name
                })

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

        def __repr__(self):
           return "<{class}(id={id}, rank='{rank}', name='{name}')>".\
                format({
                    'class': self.__class__,
                    'id': self.id,
                    'rank': self.rank,
                    'name': self.name
                })

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

        def __repr__(self):
           return "<{class}(photo='{photo}', taxon='{taxon}')>".\
                format({
                    'class': self.__class__,
                    'photo': self.photo_id,
                    'taxon': self.taxon_id
                })

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

        def __repr__(self):
           return "<{class}(id={id}, name='{name}')>".\
                format({
                    'class': self.__class__,
                    'id': self.id,
                    'name': self.name
                })

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

        def __repr__(self):
           return "<{class}(photo='{photo}', tag='{tag}')>".\
                format({
                    'class': self.__class__,
                    'photo': self.photo_id,
                    'tag': self.tag_id
                })

    # Create the database.
    Base.metadata.create_all(engine)

    # Create some default ranks.
    with session_scope(db_path) as (session, metadata):
        ranks = ('domain', 'kingdom', 'phylum', 'class', 'order', 'family',
        'genus', 'subgenus', 'section', 'species', 'subspecies')

        for name in ranks:
            rank = Rank(name=name)
            session.add(rank)

def insert_new_photo(session, metadata, root, path, **kwargs):
    """Set meta data for a photo in the database.

    Sets meta data for a photo with `path` that is relative from the top image
    directory `root`. So ``os.path.join(root, path)`` should point to and actual
    image file. Only `path` is stored in the database.

    This function checks whether an existing photo entry in the database matches
    the file's MD5 hash. If one is found, the photo entry is deleted and a new
    one is created.

    Optional keyword arguments `title`, `description`, `taxa`, and `tags` can
    be passed to provide additional meta data. Here `title` and `description`
    provide a title and a description for the photo. Taxonomic classifications
    are provided as a dict `taxa` of the format ``{rank: taxon, ...}``. Photo
    tags can be provided as a list `tags`.
    """
    real_path = os.path.join(root, path)
    if not os.path.isfile(real_path):
        raise IOError("Cannot open %s (no such file)" % real_path)

    # Get meta data from the arguments.
    title = kwargs.get('title')
    description = kwargs.get('description')
    taxa = dict(kwargs.get('taxa', {}))
    tags = list(kwargs.get('tags', []))

    # Get the database models.
    Base = automap_base(metadata=metadata)
    Base.prepare()
    Photo = Base.classes.photos
    Rank = Base.classes.ranks
    Taxon = Base.classes.taxa
    Tag = Base.classes.tags

    # Get the MD5 hash.
    hasher = hashlib.md5()
    with open(real_path, 'rb') as fh:
        buf = fh.read()
        hasher.update(buf)

    # Check if the photo exists in the database. Delete the photo entry if it
    # exists.
    try:
        photo = session.query(Photo).\
            filter(Photo.md5sum == hasher.hexdigest()).one()
        session.delete(photo)
        session.commit()
    except NoResultFound:
        photo = None

    # Insert the photo into the database.
    photo = Photo(
        md5sum=hasher.hexdigest(),
        path=path,
        title=title,
        description=description
    )

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

    if REQUIRED_RANKS:
        assert set(REQUIRED_RANKS).issubset(processed_ranks), \
            "Every photo must at least have the ranks {0}".\
                format(REQUIRED_RANKS)

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

def get_photos(session, metadata):
    """Return photo records from the database."""
    Base = automap_base(metadata=metadata)
    Base.prepare()
    Photo = Base.classes.photos
    photos = session.query(Photo)
    return photos

def get_filtered_photos_with_taxon(session, metadata, filter_):
    """Return photos with corresponding class for a filter.

    Returns all photos with corresponding taxon, as filterd by `filter_`. The
    taxon returned per photo is defined by the `class` attribute of the filter.
    Taxa to filter photos by is set in the `where` attribute of the filter.
    Filters are those as returned by :meth:`classification_hierarchy_filters`.
    Returned rows are 2-tuples ``(photo, taxon_name)``.

    Note that only the photos for which there is the rank `filter_.class` set in
    the meta data are returned.
    """
    if 'class' not in filter_:
        raise ValueError("The filter is missing the 'class' key")
    for key in vars(filter_):
        if key not in ('where', 'class'):
            raise ValueError("Unknown key '%s' in filter" % key)

    Base = automap_base(metadata=metadata)
    Base.prepare()

    # Get the table classes.
    Photo = Base.classes.photos
    Taxon = Base.classes.taxa
    Rank = Base.classes.ranks

    # Construct the main query.
    q = session.query(Photo, Taxon.name).\
        join('taxa_collection', 'ranks').\
        filter(Rank.name==getattr(filter_, 'class'))

    # Apply taxon filters.
    try:
        where = vars(filter_.where)
    except:
        where = {}
    for rank_name, taxon_name in where.items():
        if not taxon_name:
            continue
        try:
            taxon = session.query(Taxon).join('ranks').\
                filter(Taxon.name == taxon_name,
                       Rank.name == rank_name).one()
        except NoResultFound:
            raise ValueError("No such taxon %s in rank %s" % \
                (taxon_name, rank_name))

        # Filter on this taxon.
        q = q.filter(Photo.taxa_collection.contains(taxon))

    return q
