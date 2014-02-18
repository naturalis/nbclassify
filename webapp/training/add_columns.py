import os

os.system("ls | egrep '.tsv' > files.txt")

files = open("files.txt", 'r').readlines()

aantal = len(files)
header = []
out = []
counter = 0

for q in range(aantal):
    out.append(0)
    header.append("C%s"%(q+1))


for x in range(aantal):
    print "file:", files[x].strip()
    y = files[x].strip()
    name = y.split(".")[0] + "_PW.tsv"
    output = open(name, 'a')
    z = open(y, 'r').readlines()
    for a in range(len(z)):
        b = z[a].split("\t")
        i = b
        if a == 0:
            for c in range(aantal):
                i.append(header[c])
            for d in i:
                output.write(d.strip() + "\t")
            output.write("\n")
        else:
            for e in range(aantal):
                if counter == x:
                    out[x] = 1
                else:
                    pass
            #Hier Wegschrijven
            b[-1] = ''
            for f in range(aantal):
                i.append(out[f])
            for g in i:
                output.write("%s\t" %(g))
            output.write("\n")
    out[x] = 0
    counter += 1
    #os.system("rm %s"%(y))

output.close()
os.system("rm files.txt")
