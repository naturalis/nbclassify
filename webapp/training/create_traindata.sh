for i in $(ls -d */)
do
    y=${i%%/}
    echo ${i%%/}
    if [[ $y == T* ]]
    then
        cd $y
        for x in $(ls -d */)
        do
            catagory=0
            a=${x%%/}
            echo "run traindata.p for $a"
            if [[ $a == L* ]]
            then
                catagory=-1
            else
                catagory=1
            fi
            pwd
            echo "perl ../traindata.pl -d ./$a -c $catagory > $a.tsv"
            perl ../traindata.pl -d ./$a -c $catagory > $a.tsv
        done
    cd ..
    elif [[ $y == F* ]]
    then
        cd $y
        for x in $(ls -d */)
        do
            cd $x
            echo "X: $x"
            for z in $(ls -d */)
            do
                b=${z%%/}
                echo "Z: $z"
                pwd
                echo "perl ../../traindata2.pl -d ./$b > $b.tsv"
                perl ../../traindata2.pl -d ./$b > $b.tsv
            done
            cd ..
        done
    cd ..
    fi
done
pwd

clear

echo "Done"