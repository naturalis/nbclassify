ip="$1"

#Loop through every jpg file
cd static/uploaded_files/
mkdir $ip
for i in $ip*.jpg
do
#echo "File: $i"
    IFS=.
    set $i
    var="$1.png"
    x="$1.jpg"
    #Conver from jpg to png
    convert "$x" "$var"
    rm "$x"
    sed -i -e 's/.jpg/.png/g' "../../$ip.filename.txt"
    rm ../../*-e
    echo "---------------------------------------------"
done

for a in $ip*.png
do
	mv "$a" ./$ip
done