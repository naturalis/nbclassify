import os
import sys
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import Column, ForeignKey, Integer, Sequence, String, \
    UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref

def make_meta_db(db_path):
    """Create a new metadata SQLite database `db_path`."""
    engine = sqlalchemy.create_engine('sqlite:///{0}'.\
        format(os.path.abspath(db_path)),
        echo=sys.flags.debug)

    Base = declarative_base()

    class Photo(Base):

        """CREATE TABLE photos
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

        """CREATE TABLE ranks
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

        """CREATE TABLE taxa
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

        """CREATE TABLE photos_taxa
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

    # Create the database.
    Base.metadata.create_all(engine)

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
