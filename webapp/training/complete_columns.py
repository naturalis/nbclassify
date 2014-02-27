import os

#Save a list of the length files in len_files
os.system("ls | egrep '.txt' > len_files")

#Read the content of len_files and save it as a list
len_files = open('len_files', 'r').readlines()

#Create variables
name_list = []
names = []
max = 0
infile = ''

##print len_files

#Loop through the len_files and add the content to the list.
for x in len_files:
    name_list.append(x.strip())

##print name_list

#Loop through the name_list
for y in name_list:
    #Read the lenth of from the file
    file = int(open(y, 'r').readline())
##    print file
    #Find the maximum lenth
    if max < file:
        max = file
        #Save the name of the file with the maximum lenth
        infile = y

#print "Max:", max
#print "infile:", infile
##print "1:", name_list
#Remove the file wiht the maximum lenght from the name_list
name_list.remove(infile)
# name_list contains only the files that needs to be extended
##print "2:", name_list

#Loop through the name list
for z in name_list:
    #Add the names of the tsv files to the names list.
    name = z.split(".")[0] + ".tsv"
    names.append(name)
# Dit zijn de echte files die aangevuld moeten worden.
#print names

#Loop through the names list
for a in names:
##    print "----------------------------------------------"
    #Read the content of the file and save it as a list
    content = open(a, 'r').readlines()
    #Loop through the content
    for c in range(len(content)):
        #When c is zero it is the header line
        if c == 0:
            #Save the header as list
            header = content[c].split("\t")
            #Remove the empty field and the enter from the header
            header.pop(-1)
            header.pop(-1)
            #Save the lenth of the header line (= number of columns)
            file_length = int(len(header))
        ##    print file_length
            #Calculate the difference between the max lenght and the lenth of the current file
            difference = max-file_length
        ##    print difference
        ##    print header[-1]
            #Save the C-number of the last column
            number = int(header[-1].split("C")[1])
            #Create an output file
            outname = "%s_new.tsv" %(a.split(".")[0])
            #print outname
            output = open(outname, 'w')
            #Use the difference to add information to the header list
            for q in range(difference):
                adding = "C" + str(number + q + 1)
                header.append(adding)
            #Write the content of the header list to the output file
            for i in header:
                output.write("%s\t"%(i))
            #After looping through the header list, write an enter to the output file
            output.write("\n")
            #Close the output file
            output.close()
        #When c is not zero, it is not the header line
        else:
            # save the content of the line as a list.
            line = content[c].split("\t")
            #Remove the empty field and the enter from the header
            line.pop(-1)
            line.pop(-1)
            #Save the lenth of the line (= number of columns)
            file_length = int(len(line))
            #Calculate the difference between the max lenght and the lenth of the current file
            difference = max-file_length
            #Open the output file again, using the append mode
            outname = "%s_new.tsv" %(a.split(".")[0])
##            print outname
            output = open(outname, 'a')
            #Use a the difference to add enough 0's
            for v in range(difference):
                line.append(0)
            #Write the content of the line list to the output file
            for w in line:
                output.write("%s\t"%(w))
            #After looping through the line list, write an enter to the output file
            output.write("\n")
            #Close the output file
            output.close()
    #print "mv %s %s"%(outname, a)
    #Rename the output file
    os.system("mv %s %s"%(outname, a))
#Remove the temporary len_files file
os.system("rm len_files")
