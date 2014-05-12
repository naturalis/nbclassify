#!/usr/bin/env bash

#This script is running when the website is used
#Because of this, print statemens are not used.

#The first argument is the ip-adress
ip="$1"

#Go into the uploaded_files directory
cd static/uploaded_files/
#Create a directory with the name of the ip-adress
mkdir $ip
#Search for the picture with the correct ip
#If it is a .jpg convert it to a .png
for jpgs in $ip*.jpg
do
    echo "JPGS: $jpgs"
    #Create a variable with the character to split on
    IFS=.
    #Split the name of the picture
    set $jpgs
    #The first argument is the name of the picture
    #without the extension

    #Create a variable with the filename with extension png
    png="$1.png"
    #Conver from jpg to png
    convert "$jpgs" "$png"
    #Remove the jpg
    rm "$jpgs"
    #in <ip>.filename.txt chagne the extensie to png.
    #The posible extensions are .jpg, .JPG and .jpeg.
    sed -i -e 's/.jpg/.png/g' "../../$ip.filename.txt"
    sed -i -e 's/.JPG/.png/g' "../../$ip.filename.txt"
    sed -i -e 's/.jpeg/.png/g' "../../$ip.filename.txt"
    #On mac the sed command will create a file named <ip>.filename.txt-e
    #So remove this file
    rm ../../*-e
done

#Search for the picture with the correct ip
#If it is a .png move it to the directory with
#The corresponding ip
for pngs in $ip*.png
do
	mv "$pngs" ./$ip
done
