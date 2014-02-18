cd Flower

for d in $(ls -d */)
do
    echo "D: $d"
    cd $d
    python ../../add_columns.py
    python ../../combine_files.py
    cd ..
done