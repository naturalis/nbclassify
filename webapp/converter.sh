echo "Arg: $1"

ip="$1"
echo "IP: $ip"

cat "$ip.filename.txt"

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
   	echo "Var: $var"
   	echo "1: $1"
   	echo "2:$2"
   	echo "I: $i"
   	echo "X: $x"
   	echo "IP: $ip"
    #Conver from jpg to png
    echo "Convert $x to $var"
    convert "$x" "$var"
    rm "$x"
    echo "sed -i -e 's/.jpg/.png/g' \"../../$ip.filename.txt\""
    sed -i -e 's/.jpg/.png/g' "../../$ip.filename.txt"
    rm ../../*-e
    echo "---------------------------------------------"
done

for a in $ip*.png
do
	echo "mv $a ./$ip"
	mv "$a" ./$ip
done