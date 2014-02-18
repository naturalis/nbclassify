import os

os.system("ls | egrep 'PW' > files.txt")

files = open('files.txt', 'r').readlines()

aantal = len(files)
counter = 0

bla = os.path.dirname(os.path.realpath("test2.py"))

name = bla.split("/")[-1] + ".tsv"
print name

output = open(name, 'a')

##for x in range (aantal):
##    y = files[x].strip()
##    z = open(y, 'r').readlines()
##    for a in range(len(z)):
##        b = z[a].split("\t")
##        if counter == 0:
##            print "Ja"
##            print b
##            for c in b:
##                output.write(c.strip() + "\t")
##            output.write("\n")
##            counter += 1
##    break
for x in range(aantal):
    y = files[x].strip()
    print "file:", y
    z = open(y, 'r').readlines()
    for a in range(len(z)):
        b = z[a].split("\t")
        if counter == 0:
            for d in b:
                output.write(d.strip() + "\t")
            output.write("\n")
        else:
            if a == 0:
                pass
            else:
                for d in b:
                    output.write(d.strip() + "\t")
                output.write("\n")
    counter += 1
    #os.system("rm %s"%(y))
output.close()
os.system("rm files.txt")
os.system("mv %s ../"%(name))
