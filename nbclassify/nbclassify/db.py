import os
import sys
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import Column, ForeignKey, Integer, Sequence, String, \
    UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref

from exceptions import *

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
        raise FileExistsError("file {0} exists".format(db_path))

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

        rank = relationship("Rank", backref=backref(__tablename__,
            order_by=id))

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
                id INTEGER,
                photo_id INTEGER NOT NULL,
                taxon_id INTEGER NOT NULL,

                PRIMARY KEY (id),
                UNIQUE (photo_id, taxon_id),
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
                FOREIGN KEY (taxon_id) REFERENCES taxa (id) ON DELETE RESTRICT
            );
        """

        __tablename__ = 'photos_taxa'

        id = Column(Integer, Sequence("photos_taxa_id_seq"), primary_key=True)
        photo_id = Column(Integer, ForeignKey('photos.id', ondelete="CASCADE"),
            nullable=False)
        taxon_id = Column(Integer, ForeignKey('taxa.id', ondelete="RESTRICT"),
            nullable=False)

        UniqueConstraint('photo_id', 'taxon_id')

        photo = relationship("Photo", backref=backref(__tablename__,
            order_by=id))
        taxon = relationship("Taxon", backref=backref(__tablename__,
            order_by=id))

        def __repr__(self):
           return "<{class}(id='{1}', photo='{photo}', taxon='{taxon}')>".\
                format({
                    'class': self.__class__,
                    'id': self.id,
                    'photo': self.photo,
                    'taxon': self.taxon
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
                id INTEGER,
                photo_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,

                PRIMARY KEY (id),
                UNIQUE (photo_id, tag_id),
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE RESTRICT
            );
        """

        __tablename__ = 'photos_tags'

        id = Column(Integer, Sequence("photos_tags_id_seq"), primary_key=True)
        photo_id = Column(Integer, ForeignKey('photos.id', ondelete="CASCADE"),
            nullable=False)
        tag_id = Column(Integer, ForeignKey('tags.id', ondelete="RESTRICT"),
            nullable=False)

        UniqueConstraint('photo_id', 'tag_id')

        photo = relationship("Photo", backref=backref(__tablename__,
            order_by=id))
        tag = relationship("Tag", backref=backref(__tablename__,
            order_by=id))

        def __repr__(self):
           return "<{class}(id='{1}', photo='{photo}', tag='{tag}')>".\
                format({
                    'class': self.__class__,
                    'id': self.id,
                    'photo': self.photo,
                    'tag': self.tag
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
