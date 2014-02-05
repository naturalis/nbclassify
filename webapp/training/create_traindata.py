import os

directories = open("directories", 'r').readlines()

catagory = 0
for directory in directories:
    directory = directory.strip()
    print "run traindata.pl for %s" %(directory)
    if directory[0] == "L":
        catagory = -1
    else:
        catagory = 1
    os.system("perl traindata.pl -d %s -c %s > %s.tsv" %(directory, catagory, directory))
