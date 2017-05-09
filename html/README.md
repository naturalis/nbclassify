OrchID web application
----------------------

The folder structures below this directory are for the OrchID web application.

- [webapp](webapp) contains generic files to bootstrap OrchID as a Django/WSGI MVC app.
- [orchid](orchid) contains the actual Model (as in: MVC) files for the app.
- [templates](templates) contains only an HTML template

As a longer term goal, this folder structure should:

- have comprehensive unit tests.
- be scooped out of its contained folders and be migrated to a separate repository.
- be re-factored to comply with whatever the best practices are for distributing Django 
  apps.
