-- The SQLite database schema for the image harvester (harvest-images.py).
-- Meta data for photos downloaded from Flickr are saved in this database.
-- The database is automatically created when running the script.

-- Photos.
CREATE TABLE photos
(
    id INTEGER, -- Flickr photo ID
    md5sum VARCHAR NOT NULL, -- MD5 hash of the image file
    path VARCHAR,
    title VARCHAR,
    description VARCHAR,

    PRIMARY KEY (id),
    UNIQUE (md5sum),
    UNIQUE (path)
);

-- Taxonomic ranks.
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

-- Taxonomic categories.
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

-- Link photos to taxa.
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

-- Photo tags used in Flickr.
CREATE TABLE tags
(
    id INTEGER,
    name VARCHAR NOT NULL,

    PRIMARY KEY (id),
    UNIQUE (name)
);

-- Link photos to tags.
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
