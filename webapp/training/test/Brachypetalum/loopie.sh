for file in  *.png
do
	echo "perl ../../splitter.pl -t 0.85 -i $file"
	perl ../../splitter.pl -t 0.85 -i $file
done
