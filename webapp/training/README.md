Requirements for the different scripts
=======================================

Required for Offlickr.py
------------------------
* libxml2 (brew install --with-python libxml2. When this doesn't work: 1) brew edit libxml2 change "--without-python" into "--with-python" 2) run the command again)
* flickrapi (download zip form pypi.python.org/pypi/flickrapi, unzip the file, in the terminal go to the directory, run: python setup.py install)

Required for prepare_training.sh
----------------------------------
* convert using ImageMagick and jpegscr
download jpegscr here:http://www.ijg.org/files/jpegsrc.v8c.tar.gz  
downlaod ImageMagick here: http://www.imagemagick.org/download/
then cd to the download location of jpegscr and run the next commands:
tar xvfz jpeg-8c.tar.gz
cd jpeg-8c
./configure --enable-shared --prefix=$CONFIGURE_PREFIX
make
sudo make install (Password required!)

after this cd to the download location of ImageMagick and run the following commands:
cd /usr/local/src
tar xvfz ImageMagick-6.6.9-5.tar.gz
cd ImageMagick-6.6.9-5
export CPPFLAGS=-I/usr/local/include
export LDFLAGS=-L/usr/local/lib
./configure --prefix=/usr/local --disable-static --with-modules --without-perl --without-magick-plus-plus --with-quantum-depth=8 --disable-openmp
make
sudo make install
