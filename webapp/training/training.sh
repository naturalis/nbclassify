#Clear the screen.
clear

#===================================================================================================#
#                                   Download pictures from Flickr                                   #
#===================================================================================================#

#Download pictures from Flickr
python Offlickr.py -p -n -i 113733456@N06 -d .

#List all xml files and save it in a
ls | egrep xml > xml_files.txt

#Get the tags for every picture
python get_tags.py

#remove all xml files and the list of xml files saved in xml_files.txt
rm *.xml xml_files.txt


#Clear the screen, ask the user to check manualy if al pictures are download correctly. If not, download them by hand.
clear
echo "Pictures downloaded"
echo "Please check the downloaded pictures."
echo "Download all broken pictures by hand."
echo "log in on Flickr with the shared naturalis account and go to:"
echo "https://www.flickr.com/photos/113733456@N06/<picture_id>/sizes/o/"
echo "Press enter to continue"
read

#===================================================================================================#
#                               Divide pictures between Flower and Tuber                            #
#===================================================================================================#

#Create the required directories
mkdir Flower Tuber Tuber/LOblong Tuber/LSpur Tuber/LRound Tuber/Oblong Tuber/Round Tuber/Spur

#Loop through every jpg file
for i in *.jpg
do
    name=(${i//./ }$0)
    tags="$name""_tags.txt"
    #Conver from jpg to png
    echo "Convert $i to $name.png"
    convert $i $name.png
    content=$(cat $tags)
    echo "---------------------------------------------"
    #Divide the pictiure and tags between Tuber and Flower
    if [[ $content == *Tuber* ]]
    then
        mv $name.png Tuber
        mv $tags Tuber
    elif [[ $content == *Flower* ]]
    then
        mv $name.png Flower
        mv $tags Flower
    else
        echo "No correct tag found"
        echo "=================================================="
    fi
done

#Remove the jpg that will not be used anymore
rm *.jpg

#Clear the screen, print a message and wait for input from the user.
clear
echo "Pictures divided between flower and tuber"
echo "Press enter to continue"
read

#===================================================================================================#
#        Divide the pictures in Flower between the different genera and the different species       #
#===================================================================================================#

#go into the Flower directory
cd Flower

#Loop through the png files
for files in *.png
do
    name=(${files//./ }$0)
    tags="$name""_tags.txt"
    #Create variables that will be used to create directories.
    content=$(cat $tags)
    sectioni=$(sed -n '3p' < $tags)
    speciesi=$(sed -n '4p' < $tags)
    section=(${sectioni##*:})
    species=(${speciesi##*:})
    #Create a direcory with the name of the section
    mkdir $section
    #Move the picure and tag file to the the correct directory
    mv $files $section
    mv $tags $section
    #Go into the section directory and create a directory wiht the name of the species
    cd $section
    mkdir $species
    #Move the picture and tag file to the correct directory
    mv $files $species
    mv $tags $species
    #Go back to the Flower directory
    cd ..
done
#After looping through the png files in the Flower directory go out of this directory
cd ..

#Clear the screen, print a message and wait for input from the user
clear
echo "Flower pictures divided"
echo "Press enter to continue"
read

#===================================================================================================#
#                       Divide the pictures in Tuber between shape(L-a-L)                           #
#===================================================================================================#

#Go into the Tuber Directory
cd Tuber

for files in *.png
do
    #Create the variables to divide the pictures.
    name=(${files//./ }$0)
    tags="$name""_tags.txt"
    content=$(cat $tags)
    #Divide the pictures to the correct directory
    if [[ $content == *Look-a-Like_round* ]]
    then
        mv $name.png  LRound/
        mv $tags LRound/
    elif [[ $content == *Look-a-Like_oblong* ]]
    then
        mv $name.png LOblong/
        mv $tags    LOblong/
    elif [[ $content == *Look-a-Like_spur* ]]
    then
        mv $name.png LSpur/
        mv $tags LSpur/
    elif [[ $content == *Round* ]]
    then
        mv $name.png Round/
        mv $tags Round/
    elif [[ $content == *Oblong* ]]
    then
        mv $name.png Oblong/
        mv $tags Oblong/
    elif [[ $content == *Spur* ]]
        then
        mv $name.png Spur/
        mv $tags Spur/
    else
        echo "No correct tag found"
    fi
done
#After looping through the png files in the Tuber directory go out of this directory
cd ..

#Clear the screen, print a message and ask the user if he wants to split the Tuber pictures.
clear
echo "Tuber pictures divided"
read -p "Do you want to split the pictures of Tuber? (y or n)" split

#===================================================================================================#
#                               Split all pictures of Tuber                                         #
#===================================================================================================#

# When the input starts y or a Y split the Tuber pictures
case "$split" in
    y*|Y*)
        #Go into the Tuber directory
        cd Tuber

        #Loop through all directories in this folder
        for directories in $(ls -d */)
        do
            directory=${directories%%/}
            #Set standard parameter values
            size=10000
            t=0.65
            #Some directories requires other values for size or for t
            if [[ $directory == *LO* ]]
            then
                size=50000
            elif [[ $directory == *LS* ]]
            then
                size=50000
            elif [[ $directory == R* ]]
            then
                t=0.6
            fi
            #Loop through all pictures
            for files in ./$directory/*.png
            do
                #Split the picture, using the given parameter values
                echo "Splitting $files"
                perl ../splitter.pl -t $t -i $files
                #Remove the noise pictures using the file size
                for FILENAME in *,*.png
                do
                    FILESIZE=$(stat -f%z $FILENAME)
                    if (( FILESIZE > size ))
                    then
                    mv $FILENAME ./$directory
                    else
                    rm $FILENAME
                    fi
                done
            done
        done
        #When this step is finished go out of the Tuber directory
        cd ..

        #Clear the screen and print a message
        clear
        echo "Tuber pictures splited" ;;

    #If the input is something different print a message and don't split the pictures
    *)
        #Print a message that the pictures will not be split
        echo "The pictures wouldn't be split" ;;
esac

#Print a finish message
echo "The program is finished"
echo "To create traindata"
echo "Please run sh create_traindata.sh"