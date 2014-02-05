clear

python Offlickr.py -p -n -i 113733456@N06 -d .

ls | egrep xml > a

python get_tags.py

rm a *.xml

echo "Done"

mkdir Round Oblong Spur LRound LOblong LSpur
clear

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
    if [[ $content == *Look-a-Like_round* ]]
    then
        mv $var.png  LRound/$var.png
        mv $tags LRound/$tags
    elif [[ $content == *Look-a-Like_oblong* ]]
    then
        mv $var.png LOblong/$var.png
        mv $tags LOblong/$tags
    elif [[ $content == *Look-a-Like_spur* ]]
    then
        mv $var.png LSpur/$var.png
        mv $tags LSpur/$tags
    elif [[ $content == *Round* ]]
    then
        mv $var.png Round/$var.png
        mv $tags Round/$tags
    elif [[ $content == *Oblong* ]]
    then
        mv $var.png Oblong/$var.png
        mv $tags Oblong/$tags
    else
        mv $var.png Spur/$var.png
        mv $tags Spur/$tags
    fi
done

rm *.jpg

echo "Done"

clear

for f in ./LOblong/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.65 -i $f
    for FILENAME in *,*.png
    do
        FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 50000 ]
        then
            mv $FILENAME ./LOblong
        else
            rm $FILENAME
        fi
    done
done

clear

for f in ./LRound/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.65 -i $f
    for FILENAME in *,*.png
    do
        FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 10000 ]
        then
            mv $FILENAME ./LRound
        else
            rm $FILENAME
        fi
    done
done

clear

for f in ./LSpur/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.65 -i $f
    for FILENAME in *,*.png
    do
        FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 50000 ]
        then
            mv $FILENAME ./LSpur
        else
            rm $FILENAME
        fi
    done
done

clear

for f in ./Oblong/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.65 -i $f
    for FILENAME in *,*.png
    do
        FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 10000 ]
        then
            mv $FILENAME ./Oblong
        else
            rm $FILENAME
        fi
    done
done

clear

for f in ./Round/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.6 -i $f
    for FILENAME in *,*.png
    do
        FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 10000 ]
        then
            mv $FILENAME ./Round
        else
            rm $FILENAME
        fi
    done
done

clear

for f in ./Spur/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.65 -i $f
    for FILENAME in *,*.png
    do
        FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 10000 ]
        then
            mv $FILENAME ./Spur
        else
            rm $FILENAME
        fi
    done
done

for i in $(ls -d */)
do
    echo ${i%%/} >> directories
done

python create_traindata.py

mkdir traindata
mv *.tsv ./traindata
rm directories

clear

echo "Done"