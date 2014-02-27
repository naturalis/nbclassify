import os

#Collect all tsv files and save the names in files.txt
os.system("ls | egrep '.tsv' > files.txt")

#Read the names of the tsv files and save them as a list in python
files = open("files.txt", 'r').readlines()

#Create variables
number = len(files)
header = []
out = []
counter = 0
maxi = 0
#Get the path working directory
path = os.path.dirname(os.path.realpath("add_columns.py"))
#The output file for the lenght will named as the directory
file = path.split("/")[-1] + ".txt"
lenght = open(file, 'w')

#add information to the output lists
for q in range(number):
    out.append(0)
    header.append("C%s"%(q+1))

#Loop through the list of files
for x in range(number):
    #print "file:", files[x].strip()
    #Create an output file
    y = files[x].strip()
    name = y.split(".")[0] + "_new.tsv"
    output = open(name, 'a')
    #Read the content of the file, save it as a list
    z = open(y, 'r').readlines()
    #Loop through the list of content
    for a in range(len(z)):
        #Create variables, b is the content list, i is the output list
        b = z[a].split("\t")
        i = b
        #When a is 0, it is the header
        if a == 0:
            #Add the header list to the output list
            for c in range(number):
                i.append(header[c])
            #Write the content of the output list to the output file
            for d in i:
                output.write(d.strip() + "\t")
            #After looping through the output list write an enter to the output file.
            output.write("\n")
        #When a is not 0 it is a normal line
        else:
            #When the counter (=file number) is equal to x, column x gets an 1
            if counter == x:
                out[x] = 1
            #Otherwise it keeps a 0
            else:
                pass
            #Change the last column (=catagory) from a zero with an enter to a zero without an enter
            i[-1] = 0
            #Add the list of 0's (and 1's for column x == counter) to the output list
            for f in range(number):
                i.append(out[f])
            #Write the contentn of the output list to the output file
            for g in i:
                output.write("%s\t" %(g))
            #After looping through the output list write an entr to the output file.
            output.write("\n")
    #print "lengte: ", len(i)
    #Search to the hights number of columns
    if maxi < len(i):
        maxi = len(i)
    #Set the column with 1 back to 0 for the next file
    out[x] = 0
    #Add 1 to the counter.
    counter += 1
    #os.system("rm %s"%(y))

#Close the output file
output.close()
#Remove the temporary file
os.system("rm files.txt")
#print "Maxi:", maxi
#Writhe the highest number of columns to the lenght output file
lenght.write("%s"% (maxi))
#Close the lenght output file
lenght.close()
#Move the output file out the Flower directory
os.system("mv %s ../"%(file))
