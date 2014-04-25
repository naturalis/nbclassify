echo "Arg: $1"

cat "$1_filename.txt"

#Loop through every jpg file
cd static/uploaded_files/
mkdir $1
for i in $1*.jpg
do
#echo "File: $i"
    var=(${i//./ }$0)
   echo "Var: $var"
    #Conver from jpg to png
    echo "Convert $i to $var.png"
    convert $i $var.png
    rm $i
    sed -i -e 's/.jpg/.png/g' "../../$1_filename.txt"
    rm ../../*-e
    echo "---------------------------------------------"
done

for a in $1*.png
do
	mkdir $1
	mv $a ./$1
done