import os

os.system("ls | egrep '\.png' > files.txt")
os.system("mkdir Oblong; mkdir Round; mkdir Spur")
os.system("clear")

files = open('files.txt', 'r')

print "Moving the files to its own directory (Oblong, Round or Spur)"
for x in files:
    ID = x.split(".")[0]
    #print ID + "_tags.txt"
    tags = ID + "_tags.txt"
    tag = open(tags, 'r').readlines()
    if "Round\n" in tag:
        os.system("mv %s Round/%s" %(x.strip(), x.strip()))
        os.system("mv %s Round/%s" %(tags, tags))
    elif "Oblong\n" in tag:
        os.system("mv %s Oblong/%s" %(x.strip(), x.strip()))
        os.system("mv %s Oblong/%s" %(tags, tags))
    elif "Spur\n" in tag:
        os.system("mv %s Spur/%s" %(x.strip(), x.strip()))
        os.system("mv %s Spur/%s" %(tags, tags))
    else:
        tag.close()
        print y, "NO"
os.system("rm files.txt")
print "Done"
