#Go into the Flower directory
cd Flower

#Loop through the directories
for d in $(ls -d */)
do
#    echo "D: $d"
    #Go into the directory
    cd $d
    #Run add_columns.py
    python ../../add_columns.py
    #Runt combine_files.py
    python ../../combine_files.py
    #Go back to the Flower directory
    cd ..
done

#run complete_columns.py
python ../complete_columns.py
#Remove the txt files with the length of the tsv files
rm *.txt