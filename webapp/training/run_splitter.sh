clear

for f in ./LOblong/*.png
do
    echo "Splitting $f"
    perl splitter.pl -t 0.65 -i $f
    for FILENAME in *,*.png
    do
    FILESIZE=$(stat -f%z $FILENAME)
        if [ $FILESIZE -gt 10000 ]
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
        if [ $FILESIZE -gt 10000 ]
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
    perl splitter.pl -t 0.65 -i $f
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