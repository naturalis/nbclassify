.. _json-api:

===============
OrchID JSON API
===============

:Author: Serrano Pereira
:Release: |release|
:Date: |today|

This describes the JSON API for OrchID. The JSON API is used by the OrchID
web application to upload and identify orchid photos. This JSON API can also
be used for other server-client setups, like a mobile application.

Schema
======

API access is over HTTP, though HTTPS is probably required once some form of
authentication (e.g. token authentication) is implemented. The API can be
accessed via the ``/api/`` subdirectory of the OrchiD web application (e.g.
``example.com/api/``). Data is sent and received as JSON.

Root Endpoint
=============

Send a GET request to the API root to obtain a hyperlinked list of all endpoints
that the API supports::

    $ curl -i http://example.com/api/
    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, HEAD, OPTIONS
    Transfer-Encoding: chunked
    Content-Type: application/json

    {
        "photos": "http://example.com/api/photos/",
        "identities": "http://example.com/api/identities/"
    }

Parameters
==========

Some API endpoints take optional parameters. For POST, PATCH, PUT, and DELETE
requests, the parameters must be encoded as JSON with a Content-Type of
"application/json"::

    $ curl -X PATCH -H "Content-Type: application/json" -d '{"roi": "186,55,117,218"}' http://example.com/api/photos/1/

    {
        "id": 1,
        "image": "http://example.com/media/orchid/uploads/2015/01/16/798232f20a.jpg",
        "roi": "186,55,117,218",
        "identities": [
            1,
            2
        ]
    }

Client Errors
=============

Sending invalid JSON will result in a ``400 Bad Request`` response::

    HTTP/1.1 400 BAD REQUEST
    Content-Type: application/json

    {"detail": "JSON parse error - Expecting object: line 1 column 24 (char 23)"}

Sending the wrong type of JSON values will result in a ``400 Bad Request``
response::

    HTTP/1.1 400 BAD REQUEST
    Content-Type: application/json

    {"roi": ["The ROI must have the format `x,y,width,height`"]}

Sending invalid fields will result in a ``400 Bad Request`` response::

    HTTP/1.1 400 BAD REQUEST
    Content-Type: application/json

    {"image": ["The submitted data was not a file. Check the encoding type on the form."]}


HTTP Verbs
==========

Where possible, the API strives to use the appropriate HTTP verbs for each
action.

======  =============
 Verb    Description
======  =============
HEAD    Can be issued against any resource to get just the HTTP header info.
GET     Used for retrieving resources.
POST    Used for creating resources.
PATCH   Used for updating resources with partial JSON data.
PUT     Used for replacing resources or collections.
DELETE  Used for deleting resources.
======  =============

Authentication
==============

The JSON API is currently at an early development stage and no form of user
authentication has been implemented so far. This means that users are currently
able to view and edit photos uploaded by other users.

Once user authentication is implemented, permissions can be set so that users
can only edit their own photos.

Pagination
==========

Requests that return multiple items will be paginated to 30 items. You can
specify further pages with the ``?page`` parameter::

    curl http://example.com/api/identities/?page=2

Note that page numbering is 1-based and that omitting the ``?page`` parameter
will return the first page. The resource will also contain multiple properties
to make navigation easier for the client::

    $ curl http://example.com/api/photos/

    {
        "count": 68,
        "next": "http://example.com/api/photos/?page=2",
        "previous": null,
        "results": [
            {
                "id": 1,
                "image": "http://example.com/media/orchid/uploads/2015/01/16/798232f20a.jpg",
                "roi": "186,55,117,218",
                "identities": [
                    1,
                    2
                ]
            },
            ...
        ]
    }

The possible properties are as follows.

=========  ===========
Name       Description
=========  ===========
count      The total number of items for the results.
next       Shows the URL of the immediate next page of results.
previous   Shows the URL of the immediate previous page of results.
results    The list of items for the given page.
=========  ===========


Photos list
===========

::

    GET /api/photos/

Should return only the user's photos once authentication is implemented.

Response
--------

::

    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, POST, HEAD, OPTIONS
    Content-Type: application/json

    {
        "count": 68,
        "next": "http://example.com/api/photos/?page=2",
        "previous": null,
        "results": [
            {
                "id": 1,
                "image": "http://example.com/media/orchid/uploads/2015/01/16/798232f20a.jpg",
                "roi": "186,55,117,218",
                "identities": [
                    1,
                    2
                ]
            },
            ...
        ]
    }


Get a single photo
==================

::

    GET /api/photos/:id/

Response
--------

::

    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, PUT, PATCH, DELETE, HEAD, OPTIONS
    Content-Type: application/json

    {
        "id": 1,
        "image": "http://example.com/media/orchid/uploads/2015/01/16/798232f20a.jpg",
        "roi": "186,55,117,218",
        "identities": [
            1,
            2
        ]
    }

Response fields
---------------

===========  =============
Field name   Description
===========  =============
id           The photo ID.
image        URL for the photo.
roi          The region of interest (ROI) in the format ``x,y,width,height``
             pixels.
             The ROI is set by the client when the flower is selected in the
             image.
identities   List of identifications made for the photo. This only lists the
             IDs for the identities.
===========  =============


Upload a photo
==============

::

    POST /api/photos/

Example::

    curl -F image=@Mexipedium_xerophyticum.jpg http://example.com/api/photos/

Response
--------

::

    HTTP/1.1 201 CREATED
    Vary: Accept,Cookie
    Allow: GET, POST, HEAD, OPTIONS
    Content-Type: application/json

    {
        "id": 26,
        "image": "http://example.com/media/orchid/uploads/2015/02/16/915995be75.jpg",
        "roi": null,
        "identities": []
    }


Update a photo
==============

::

    PATCH /api/photos/:id/

Example::

    curl -X PATCH -H 'Content-Type: application/json' -d '{"roi": "0,0,300,300"}' http://example.com/api/photos/26/

Response
--------

::

    HTTP/1.1 201 CREATED
    Vary: Accept,Cookie
    Allow: GET, POST, HEAD, OPTIONS
    Content-Type: application/json

    {
        "id": 26,
        "image": "http://example.com/media/orchid/uploads/2015/02/16/915995be75.jpg",
        "roi": "0,0,300,300",
        "identities": []
    }


Identify a photo
================

::

    GET /api/photos/:id/identify/
    POST /api/photos/:id/identify/

Example::

    curl http://example.com/api/photos/26/identify/

Example with modified region of interest (ROI)::

    curl -X POST -H 'Content-Type: application/json' -d '{"roi": "30,92,764,812"}' http://example.com/api/photos/26/identify/

Response
--------

::

    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, POST, HEAD, OPTIONS
    Content-Type: application/json

    {
        "id": 26,
        "image": "http://example.com/media/orchid/uploads/2015/02/16/915995be75.jpg",
        "roi": "30,92,764,812",
        "identities": [
            108,
            109
        ]
    }


List identities for a photo
===========================

List all the identities for a given photo::

    GET /api/photos/:id/identities/

Example::

    curl http://example.com/api/photos/26/identities/

Response
--------

::

    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, HEAD, OPTIONS
    Content-Type: application/json

    {
        "identities": [
            {
                "id": 108,
                "photo": 26,
                "genus": "Mexipedium",
                "section": "Mexipedium",
                "species": "xerophyticum",
                "error": 2.325400737519419e-14
            },
            {
                "id": 109,
                "photo": 26,
                "genus": "Paphiopedilum",
                "section": null,
                "species": null,
                "error": 0.000002519105043899469
            }
        ]
    }


Delete a photo
==============

Deleting a photo also causes the related identities to be deleted::

    DELETE /api/photos/:id/

Example::

    curl -X DELETE http://example.com/api/photos/26/

Response
--------

::

    HTTP/1.1 204 NO CONTENT
    Vary: Accept,Cookie
    Allow: GET, PUT, PATCH, DELETE, HEAD, OPTIONS
    Content-Length: 0


Identities list
===============

List all the identities::

    GET /api/identities/

Should return only the user's identities once authentication is implemented.

Response
--------

::

    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, HEAD, OPTIONS
    Content-Type: application/json

    {
        "count": 72,
        "next": "http://example.com/api/identities/?page=2",
        "previous": null,
        "results": [
            {
                "id": 1,
                "photo": 1,
                "genus": "Phragmipedium",
                "section": "Phragmipedium",
                "species": "lindenii",
                "error": 7.434628591867027e-08
            },
            ...
        ]
    }


Get a single photo identity
===========================

::

    GET /api/identities/:id/

Response
--------

::

    HTTP/1.1 200 OK
    Vary: Accept,Cookie
    Allow: GET, DELETE, HEAD, OPTIONS
    Content-Type: application/json

    {
        "id": 1,
        "photo": 1,
        "genus": "Phragmipedium",
        "section": "Phragmipedium",
        "species": "lindenii",
        "error": 7.434628591867027e-08
    }

Response fields
---------------

===========  =============
Field name   Description
===========  =============
id           The identity ID.
photo        The ID of the photo for which this identity was made.
genus        The name of the genus.
section      The name of the section.
species      The name of the species.
error        The mean square error (MSE) value for this classification.
===========  =============


Get taxon information for an identity
=====================================

This fetches taxon information from the Encyclopedia of Life::

    GET /api/identities/:id/info/

Response
--------

See http://eol.org/api/docs/pages for response format.

HTML response is also supported for use in the OrchID web application::

    <div class="hidden" id="species-name"><em>Phragmipedium lindenii</em> (Lindl.) Dressler &amp; N.H.Williams</div>
        <div class="alert alert-success" role="alert">IUCN threat status: <a href="http://www.iucnredlist.org/apps/redlist/details/43324857" class="alert-link">Least Concern (LC)</a></div>
    <div class="container-fluid">
        <a href="http://eol.org/data_objects/29868742"><img class="img-rounded" src="http://media.eol.org/content/2014/07/09/02/28667_98_68.jpg" alt="El Valle - Phragmipedium Lindenii Orchid"></a>
        <a href="http://eol.org/data_objects/30893623"><img class="img-rounded" src="http://media.eol.org/content/2014/09/27/01/18678_98_68.jpg" alt="El Valle - Phragmipedium Lindenii Orchid"></a>
        <a href="http://eol.org/data_objects/31488128"><img class="img-rounded" src="http://media.eol.org/content/2014/10/20/10/41492_98_68.jpg" alt="File:Phragmipedium lindenii Orchi 030.jpg"></a>
        <a href="http://eol.org/data_objects/31926505"><img class="img-rounded" src="http://media.eol.org/content/2012/06/15/16/20655_98_68.jpg" alt="File:Phragmipedium lindenii Orchi 066.jpg"></a>
        <a href="http://eol.org/data_objects/31926506"><img class="img-rounded" src="http://media.eol.org/content/2012/06/14/21/18370_98_68.jpg" alt="File:Phragmipedium lindenii Orchi 068.jpg"></a>
    </div>
        <h3>Range Description</h2>
        <p><em>Phragmipedium lindenii</em> is a large terrestrial, lithophytic, or epiphytic orchid. It has been reported from mountainous areas of Venezuela, Colombia and Ecuador (Dressler and Williams 1975, Coz and Bravo 2007, Villafuerte and Christenson 2007). New records have recently been found in Peru, which represents a large range extension (Coz and Bravo 2007, Villafuerte and Christenson 2007).</p>
    <span class="pull-right"><a href="http://eol.org/1135011"><img src="/static/orchid/images/eol_logo_100.png" height="25px" title="More info on the Encyclopedia of Life" alt="EOL.org"></a></span>

Delete a photo identity
=======================

Users should only be able to delete their own photo identities once
authentication is implemented.

::

    DELETE /api/identities/:id/

Example::

    curl -X DELETE http://example.com/api/identities/1/

Response
--------

::

    HTTP/1.1 204 NO CONTENT
    Vary: Accept,Cookie
    Allow: GET, DELETE, HEAD, OPTIONS
    Content-Length: 0
