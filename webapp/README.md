dependencies
============
* python2.7
* Django 1.6.4 (sudo pip install Django==1.6.4)
* convert using ImageMagick and jpegscr


How to install ImageMagick and jpegscr?
========================================

download jpegscr here:http://www.ijg.org/files/jpegsrc.v8c.tar.gz  

downlaod ImageMagick here: http://www.imagemagick.org/download/

then cd to the download location of jpegscr and run the next commands:

tar xvfz jpeg-8c.tar.gz

cd jpeg-8c

./configure --enable-shared --prefix=$CONFIGURE_PREFIX

make

sudo make install (Password required!)

after this cd to the download location of ImageMagick and run the following commands:

tar xvfz ImageMagick-6.6.9-5.tar.gz (Use the numbers of your version!)

cd ImageMagick-6.6.9-5 (Use the numbers of your version!)

export CPPFLAGS=-I/usr/local/include

export LDFLAGS=-L/usr/local/lib

./configure --prefix=/usr/local --disable-static --with-modules --without-perl --without-magick-plus-plus --with-quantum-depth=8 --disable-openmp

make

sudo make install


How to start the app locally?
=============================
* Start the command line
* Go to the directory which contains manage.py
* Type python manage.py runserver
* Open a webbrowser (not Internet Explorer!) and go to 127.0.0.1:8000 (or use the link generated in the command line)

How to start the app in a network?
==================================
* Find the ip-adres of the local machine
* Start the command line
* Go to the direcory which contains manage.py
* Type python manage.py runserver ip-adres:8000
* Open a webbrowser (not Internet Explorer!) and go to ip-adres:8000 (This will work for every computer in the same network)
