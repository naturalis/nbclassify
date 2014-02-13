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

rm *.xml a

echo "Done"

clear

#===================================================================================================#
#                               Divide pictures between Flower and Tuber                            #
#===================================================================================================#

mkdir Flower Tuber Tuber/LOblong Tuber/LSpur Tuber/LRound Tuber/Oblong Tuber/Round Tuber/Spur

for i in *.jpg
do
    echo "File: $i"
    var=(${i//./ }$0)
    echo "Var: $var"
    tags="$var""_tags.txt"
    echo "Tag: $tags"
    echo "Convert $i to $var.png"
    convert $i $var.png
    content=$(cat $tags)
    echo "Content: $content"
    echo "---------------------------------------------"
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

#Remove the files that will not be used anymore
rm *.jpg

echo "Done"

clear

#===================================================================================================#
#        Divide the pictures in Flower between the different genera and the different species       #
#===================================================================================================#

cd Flower

for f in *.png
do
    echo "File: $f"
    var=(${f//./ }$0)
    echo "Var: $var"
    tags="$var""_tags.txt"
    echo "Tag: $tags"
    content=$(cat $tags)
    genusi=$(sed -n '3p' < $tags)
    speciesi=$(sed -n '4p' < $tags)
    genus=(${genusi##*:})
    species=(${speciesi##*:})
    echo "Speciesi: $speciesi"
    echo "Species: $species"
    mkdir $genus
    mv $f $genus
    mv $tags $genus
    cd $genus
    mkdir $species
    mv $f $species
    cd ..
    #    rm $tags
done
cd ..
clear

#===================================================================================================#
#                       Devide the pictures in Tuber between shape(L-a-L)                           #
#===================================================================================================#

cd Tuber

for f in *.png
do
    echo "File: $f"
    var=(${f//./ }$0)
    echo "Var: $var"
    tags="$var""_tags.txt"
    echo "Tag: $tags"
    content=$(cat $tags)
    echo "Content: $content"
    if [[ $content == *Look-a-Like_round* ]]
    then
        mv $var.png  LRound/$var.png
    elif [[ $content == *Look-a-Like_oblong* ]]
    then
        mv $var.png LOblong/$var.png
    elif [[ $content == *Look-a-Like_spur* ]]
    then
        mv $var.png LSpur/$var.png
    elif [[ $content == *Round* ]]
    then
        mv $var.png Round/$var.png
    elif [[ $content == *Oblong* ]]
    then
        mv $var.png Oblong/$var.png
    elif [[ $content == *Spur* ]]
        then
        mv $var.png Spur/$var.png
    else
        echo "No correct tag found"
    fi
    #rm $tags
done
cd ..

clear

#===================================================================================================#
#                               Split alle foto's (Tuber en Flower)                                 #
#===================================================================================================#

for d in $(ls -d */)
do
    echo "D: $d"

#======================================================================================#
#                               Tuber
    if [[ $d == *Tuber* ]]
    then
#pwd
#       echo "**********************************************************"
#       echo "Nu in Tuber"
        cd ${d%%/}
        for i in $(ls -d */)
        do
            y=${i%%/}
#           echo "Y1: $y"
            size=10000
            t=0.65
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
            for f in ./$y/*.png
            do
                echo "Splitting $f"
                #       echo "-t: $t"
                #       echo "Size: $size"
#               pwd
#               echo "+++++++++++++++++++++++++++++++++++++++++++++"
                perl ../splitter.pl -t $t -i $f
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

#======================================================================================#
#                               Flower
    elif [[ $d == Flower* ]]
    then
        cd ${d%%/}
        for i in $(ls -d */)
        do
            size=10000
            t=0.85
            q=$t
#  echo "Y2: $i"
#           echo "Size: $size"
#           echo "-t: $t"
            if [[ $i == *C* ]]
            then
#                echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                t=0.9
            fi

            cd ${i%%/}
            for x in $(ls -d */)
            do
                y=${x%%/}
#               echo "Y3: $y"
                if [[ $y == *vie* ]]
                then
#                   echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                    t=0.9
                elif [[ $y == *vicotria* ]]
                then
                    size=7000
                elif [[ $y == *druryi* ]]
                then
                    size=20000
                elif [[ $y == *tigrinum* ]]
                then
                    size=20000
                elif [[ $y == *delenatii* ]]
                then
                    t=0.95
                else
                    size=10000
                fi
                for g in ./$y/*.png
                do
                    if [[ $g == *12342093963* ]]
                    then
#                       echo "1"
                        q=0.9
                    elif [[ $g == *12342327174* ]]
                    then
#                       echo "2"
                        q=0.95
                    elif [[ $g == *12342204914* ]]
                    then
#                       echo "3"
                        q=0.9
                    elif [[ $g == *12341889003* ]]
                    then
#                       echo "4"
                        q=0.9
                    elif [[ $g == *12341158664* ]]
                    then
#                       echo "5"
                        q=0.9
                    elif [[ $g == *12340702965* ]]
                    then
#                       echo "6"
                        q=0.9
                    elif [[ $g == *12340849735* ]]
                    then
#                       echo "7"
                        q=0.95
                    elif [[ $g == *12342126885* ]]
                    then
#                       echo "8"
                        q=0.95
                    elif [[ $g == *12342401264* ]]
                    then
#                       echo "9"
                        q=0.95
                    else
#                       echo "10"
                        q=$t
                    fi
                    echo "Splitting $g"
#                   echo "Size: $size"
#                   echo "Q: $q"
#                   echo "-t: $t"
#                   echo "________________________________________"
                    perl ../../splitter.pl -t $q -i $g
                    for FILENAME in *,*.png
                    do
                        FILESIZE=$(stat -f%z $FILENAME)
#                       echo "$FILENAME: $FILESIZE"
                        if (( FILESIZE > size ))
                        then
                        mv $FILENAME ./$y
                        else
                        rm $FILENAME
                        fi
                    done
                done
#           echo "----------------------------"
            done
#       echo "================================"
        cd ..
        done
    cd ..
    fi
done

clear
echo "Done"