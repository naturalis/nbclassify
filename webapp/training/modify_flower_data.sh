#Go into the Flower directory
cd Flower

#Loop through the directories
for directory in $(ls -d */)
do
    #Go into the directory
    cd $directory
    #Run combine_files.py
    echo "Run combine_files.py"
    python ../../combine_files.py
    #Go back to the Flower directory
    cd ..
done

#Run add_columns.py
echo "Run add_columns.py"
python ../add_columns.py

#Print a finising message
echo "Finished"

echo "To train a neural network use the folowing command:"
echo "perl trainai.pl -d <directory_with_traindata> -c <number_of_catogories> -o <output>"
echo "To change the Desired Error add -t <desired_error>"
echo "To change the number of epoch add -e <epochs>"
echo "To follow the run time add date; before and ;date after the command"