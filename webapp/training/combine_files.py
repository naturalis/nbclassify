import os

#Save all tsv files created with add_columns.py in files.txt
os.system("ls | egrep '.tsv' > files.txt")

#Read the content of files.txt and save it as a list
files = open('files.txt', 'r').readlines()

#Create variables
number = len(files)
counter = 0
#Get the path working directory
paht = os.path.dirname(os.path.realpath("combine_files.py"))
#The output file will named as the directory
name = paht.split("/")[-1] + ".tsv"
#print name
output = open(name, 'a')

#Loop through the files list
for x in range(number):
    y = files[x].strip()
    #print "file:", y
    #Read the content of the file, save it as a list.
    z = open(y, 'r').readlines()
    #Loop through the lines
    for a in range(len(z)):
        #create a list, every columns is an entry of the list
        b = z[a].split("\t")
        #When counter is 0, this is the first file.
        if counter == 0:
            #The whole content of the file is written to the output file
            for d in b:
                output.write(d.strip() + "\t")
            #After looping through the content, write an enter
            output.write("\n")
        #When counter isn't 0, it is not the first file
        else:
            #When a is 0 it is the header
            if a == 0:
                #The header will not be written to the output file, because it is already there
                pass
            #When a isn't 0 it is not the header
            else:
                #The content will be written to the output file
                for d in b:
                    output.write(d.strip() + "\t")
                #After looping through the content, write an enter to the output file.
                output.write("\n")
    #Add 1 to the counter
    counter += 1
    #os.system("rm %s"%(y))
#Close the output file
output.close()
#Remove the temporary files.txt file
os.system("rm files.txt")
#Move the output file out of the Flower directory
os.system("mv %s ../"%(name))
