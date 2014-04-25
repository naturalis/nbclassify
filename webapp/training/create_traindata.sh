#Loop throug the directories
for i in $(ls -d */)
do
    y=${i%%/}
#    echo ${i%%/}
    #The tuber pictures use another Perl script for training the neural network
    #than the flower pictures. So check the name of the directory.
    if [[ $y == T* ]]
    #If it starts with T (=Tuber)
    then
        #Go into the directory
        cd $y
        #Loop through the directories
        for x in $(ls -d */)
        do
            #Create variables
            catagory=0
            a=${x%%/}
            echo "run traindata.pl for $a"
            #Give the Look-a-like tubers catagory -1
            if [[ $a == L* ]]
            then
                catagory=-1
            #Give the salep tubers catagory 1
            else
                catagory=1
            fi
#pwd
#           echo "perl ../traindata.pl -d ./$a -c $catagory > $a.tsv"
            #Run the traindata script and redirect the output to a tsv file.
            #Give this file the name of the current directory
            perl ../traindata.pl -d ./$a -c $catagory > $a.tsv
        done
    #After looping through the directories go out of the Tuber directory
    cd ..
    elif [[ $y == F* ]]
    #If it starts with F (=Flower)
    then
        #Go into the directory
        cd $y
        #Loop through the genus directories
        for x in $(ls -d */)
        do
            #Go into the genus directory
            cd $x
#            echo "X: $x"
            #Loop through the species directories
            for z in $(ls -d */)
            do
                #Create variables
                b=${z%%/}
#                echo "Z: $z"
#               pwd
#               echo "perl ../../traindata2.pl -d ./$b > $b.tsv"
				echo "run traindata2.pl for $b"
                #Run the traindata2 script and redirect the output to a tsv file.
                #Give this file the name of the current directory
                perl ../../traindata2.pl -d ./$b -c 0 > $b.tsv
            done
            #After loopging through the species directories go back to the Flower directory
            cd ..
        done
    #After looping through the genus directories gou out of the Flower directory
    cd ..
    fi
done
#pwd

clear

echo "Done"