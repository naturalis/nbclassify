clear

#===================================================================================================#
#                                   Download pictures from Flickr                                   #
#===================================================================================================#

#Download pictures from Flickr
python Offlickr.py -p -n -i 113733456@N06 -d .

#List all xml files and save it in a
ls | egrep xml > a

#Get the tags for every picture
python get_tags.py

#remove all xml files and the list of xml files saved in a
rm *.xml a

clear
echo "Pictures downloaded"
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
#echo "File: $i"
    var=(${i//./ }$0)
#   echo "Var: $var"
    tags="$var""_tags.txt"
#   echo "Tag: $tags"
    #Conver from jpg to png
    echo "Convert $i to $var.png"
    convert $i $var.png
    content=$(cat $tags)
#   echo "Content: $content"
    echo "---------------------------------------------"
    #Divide the pictiure and tags between Tuber and Flower
    if [[ $content == *Tuber* ]]
    then
        mv $var.png Tuber
        mv $tags Tuber
    elif [[ $content == *Flower* ]]
    then
        mv $var.png Flower
        mv $tags Flower
    else
        echo "No correct tag found"
        echo "=================================================="
    fi
    #rm $tags
done

#Remove the jpg that will not be used anymore
rm *.jpg

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
for f in *.png
do
#echo "File: $f"
    var=(${f//./ }$0)
#echo "Var: $var"
    tags="$var""_tags.txt"
#   echo "Tag: $tags"
    #Create variables that will be used to create directories.
    content=$(cat $tags)
    genusi=$(sed -n '3p' < $tags)
    speciesi=$(sed -n '4p' < $tags)
    genus=(${genusi##*:})
    species=(${speciesi##*:})
#echo "Speciesi: $speciesi"
#   echo "Species: $species"
    #Create a direcory with the name of the genus
    mkdir $genus
    #Move the picure and tag file to the the correct directory
    mv $f $genus
    mv $tags $genus
    #Go into the genus directory and create a directory wiht the name of the species
    cd $genus
    mkdir $species
    #Move the picture and tag file to the correct directory
    mv $f $species
    mv $tags $species
    #Go back to the Flower directory
    cd ..
    #    rm $tags
done
#After looping through the png files in the Flower directory go out of this directory
cd ..
clear
echo "Flower pictures divided"
echo "Press enter to continue"
read

#===================================================================================================#
#                       Devide the pictures in Tuber between shape(L-a-L)                           #
#===================================================================================================#

#Go into the Tuber Directory
cd Tuber

for f in *.png
do
#echo "File: $f"
    #Create the variables to divide the pictures.
    var=(${f//./ }$0)
#   echo "Var: $var"
    tags="$var""_tags.txt"
#   echo "Tag: $tags"
    content=$(cat $tags)
#   echo "Content: $content"
    #Divide the pictures to the correct directory
    if [[ $content == *Look-a-Like_round* ]]
    then
        mv $var.png  LRound/
        mv $tags LRound/
    elif [[ $content == *Look-a-Like_oblong* ]]
    then
        mv $var.png LOblong/
        mv $tags    LOblong/
    elif [[ $content == *Look-a-Like_spur* ]]
    then
        mv $var.png LSpur/
        mv $tags LSpur/
    elif [[ $content == *Round* ]]
    then
        mv $var.png Round/
        mv $tags Round/
    elif [[ $content == *Oblong* ]]
    then
        mv $var.png Oblong/
        mv $tags Oblong/
    elif [[ $content == *Spur* ]]
        then
        mv $var.png Spur/
        mv $tags Spur/
    else
        echo "No correct tag found"
    fi
    #rm $tags
done
#After looping through the png files in the Tuber directory go out of this directory
cd ..
clear
echo "Tuber pictures divided"
echo "Press enter to continue"
read

#===================================================================================================#
#                               Split all pictures of Tuber                                         #
#===================================================================================================#

#Go into the Tuber directory
cd Tuber

#Loop through all directories in this folder
for i in $(ls -d */)
do
    y=${i%%/}
#           echo "Y1: $y"
    #Set standard parameter values
    size=10000
    t=0.65
    #Some directories requires other values for size or for t
    if [[ $y == *LO* ]]
    then
        size=50000
    elif [[ $y == *LS* ]]
    then
        size=50000
    elif [[ $y == R* ]]
    then
        t=0.6
    fi
    #Loop through all pictures
    for f in ./$y/*.png
    do
        #Split the picture, using the given parameter values
        echo "Splitting $f"
        #       echo "-t: $t"
        #       echo "Size: $size"
#               pwd
#               echo "+++++++++++++++++++++++++++++++++++++++++++++"
        perl ../splitter.pl -t $t -i $f
        #Remove the noise pictures using the file size
        for FILENAME in *,*.png
        do
            FILESIZE=$(stat -f%z $FILENAME)
            #           echo "$FILENAME: $FILESIZE"
            if (( FILESIZE > size ))
            then
            mv $FILENAME ./$y
            else
            rm $FILENAME
            fi
        done
    done
done
cd ..

clear
echo "Tuber pictures splited"
#echo "Press enter to continue"
#read

#cd Flower
#Loop through all directories in this folder
#for i in $(ls -d */)
#do
#    y=${i%%/}
#	
	#Set parameter values
#	size=4000
#	t=0.85
	
#	if [[ $y == *Barbata* ]]
#	then
#		t=0.85
#		size=10000
#	elif [[ $y == *Cochlopetalum* ]]
#	then
#		t=0.7
#	elif [[ $y == *Paphiopedilum* ]]
#	then
#		t=0.75
#	fi
	
#	echo "Y1: $y"
#	echo "Size: $size"
#	echo "-t: $t"
#	echo "--------------------------------------------"
	
#	if [[ $y == *Brachypetalum* ]]
#	then
#		continue
#	else
#		cd $y
#		if [[ $y == *Barbata* || $y == *Coryopedilum* || $y == *Pardalopetalum* ]]
#		then
#			echo "Y2: $y moet helemaal"
#			echo "+++++++++++++++++++++++++++++++++++++"
#			for a in $(ls -d */)
#			do
#				b=${a%%/}
#				echo "B1: $b"
#				echo "_____________________________________"
#				echo "-t: $t"
#				echo "============================================="
#				cd $b
#				mkdir original
#				for file in *.png
#				do
#					echo "perl ../../../splitter.pl -t $t -i $file"
#					perl ../../../splitter.pl -t $t -i $file
#					echo "mv $file ./original"
#					mv $file ./original
#					for FILENAME in *,*.png
#					do
#						FILESIZE=$(stat -f%z $FILENAME)
						#           echo "$FILENAME: $FILESIZE"
#						if (( FILESIZE < size ))
#						then
#						rm $FILENAME
#						else
#						continue
#						fi
#					done
#				done
#				cd ..
#			done
#		elif [[ $y == *Cochlopetalum* ]]
#		then
#			cd glaucophyllum
#			ls
#			mkdir original
#			echo "perl ../../../splitter.pl -t $t -i 13239075545.png"
#			perl ../../../splitter.pl -t $t -i 13239075545.png
#			echo "mv 13239075545.png ./original"
#			mv 13239075545.png ./original
#			cd ..
#		elif [[ $y == *Paphiopedilum* ]]
#		then
#			cd villosum
#			ls
#			mkdir original
#			echo "perl ../../../splitter.pl -t $t -i 13239599774.png"
#			perl ../../../splitter.pl -t $t -i 13239599774.png
#			echo "mv 13239599774.png ./original"
#			mv 13239599774.png ./original
#			cd ..
#		elif [[ $y == *Parvisepalum* ]]
#		then
#			cd armeniacum
#			ls
#			mkdir original
#			echo "perl ../../../splitter.pl -t $t -i 13218412065.png"
#			perl ../../../splitter.pl -t $t -i 13218412065.png
#			echo "mv 13218412065.png ./original"
#			mv 13218412065.png ./original
#			echo "perl ../../../splitter.pl -t $t -i 13218413845.png"
#			perl ../../../splitter.pl -t $t -i 13218413845.png
#			echo "mv 13218413845.png ./original"
#			mv 13218413845.png ./original
#			cd ../vietnamense
#			ls
#			mkdir original
#			echo "perl ../../../splitter.pl -t $t -i 13218572413.png"
#			perl ../../../splitter.pl -t $t -i 13218572413.png
#			echo "mv 13218572413.png ./original"
#			mv 13218572413.png ./original
#			cd ..
#		cd ..
#		fi
		
#		cd ..
#	fi
#done

#clear
#echo "Flower pictures splited"