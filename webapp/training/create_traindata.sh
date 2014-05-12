#!/usr/bin/env bash
#Loop throug the directories
for directories in $(ls -d */)
do
    directory=${directories%%/}
    #The tuber pictures use another Perl script for training the neural network
    #than the flower pictures. So check the name of the directory.
    if [[ $directory == T* ]]
    #If it starts with T (=Tuber)
    then
        #Go into the directory
        cd $directory
        #Loop through the directories
        for directories2 in $(ls -d */)
        do
            #Create variables
            catagory=0
            shape=${directories2%%/}
            echo "run traindata.pl for $shape"
            #Give the Look-a-like tubers catagory -1
            if [[ $shape == L* ]]
            then
                catagory=-1
            #Give the salep tubers catagory 1
            else
                catagory=1
            fi

            #Run the traindata script and write the output to a tsv file.
            #Give this file the name of the current directory
            perl ../traindata.pl -d ./$shape -c $catagory > $shape.tsv
        done
    #After looping through the directories go out of the Tuber directory
    cd ..
    elif [[ $directory == F* ]]
    #If it starts with F (=Flower)
    then
        #Go into the directory
        cd $directory
        #Loop through the section directories
        for section in $(ls -d */)
        do
            #Go into the section directory
            cd $section
            #Loop through the species directories
            for species_dir in $(ls -d */)
            do
                #Create variables
                species=${species_dir%%/}
				echo "run traindata2.pl for $species"
                #Run the traindata2 script and redirect the output to a tsv file.
                #Give this file the name of the current directory
                perl ../../traindata2.pl -d ./$species -c 0 > $species.tsv
            done
            #After loopging through the species directories go back to the Flower directory
            cd ..
        done
    #After looping through the section directories go out of the Flower directory
    cd ..
    fi
done

#Clear the screen
clear

#Print a finish message
echo "Done"
echo "Please modify the flower data:"
echo "Run: sh modify_flower_data.sh"