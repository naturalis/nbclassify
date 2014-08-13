dependencies
============
* python2.7
* Django 1.6.4 (sudo pip install Django==1.6.4)
* ImageMagick (c library) and Image::Magick (perl bindings)
* FANN (c library) version 2.2.0 and AI::FANN (perl bindings)
* Bio::Phylo (for logging)

How to start the app locally?
=============================
* Start the command line
* Go to the directory which contains manage.py
* Type `python manage.py runserver`
* Open a webbrowser (not Internet Explorer!) and go to <http://127.0.0.1:8000/> (or use the link generated in the command line)

How to start the app in a network?
==================================
* Find the ip-adres of the local machine
* Start the command line
* Go to the direcory which contains manage.py
* Type `python manage.py runserver ip-adres:8000`
* Open a webbrowser (not Internet Explorer!) and go to <http://ip-adres:8000/> (This will work for every computer in the same network)
