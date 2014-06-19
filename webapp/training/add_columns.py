import os

#Collect all tsv files and save the names in files.txt
os.system("ls | egrep '.tsv' > files.txt")

#Function for creating the variables
def create_variables():
    #Read the names of the tsv files and save them as a list in python
    files = open("files.txt", 'r').readlines()

    #Create variables
    number = len(files)
    header = []
    out_list = []
    counter = 0
    maxi = 0
    
    #Run the create_output_list function
    create_output_list(number, header, out_list ,files, counter, maxi)

#Function for creating the output lists
def create_output_list(number, header, out_list, files, counter, maxi):
    #add information to the output lists
    for q in range(number):
        out_list.append(-1)
        header.append("C%s"%(q+1))
    #Run the change_files function
    change_files(header, out_list, number ,files, counter, maxi)

#Functin for changing the files. This function will add the columns to the files.
def change_files(header, out_list, number, files, counter, maxi):
    #Loop through the list of files
    for x in range(number):
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
            i = b[:-2]
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
                    out_list[x] = 1
                #Otherwise it keeps a 0
                else:
                    pass
                #Add the list of -1's (and 1's for column x == counter) to the output list
                for f in range(number):
                    i.append(out_list[f])
                #Write the contentn of the output list to the output file
                for g in i:
                    output.write("%s\t" %(g))
                #After looping through the output list write an entr to the output file.
                output.write("\n")
        #Search to the hights number of columns
        if maxi < len(i):
            maxi = len(i)
        #Set the column with 1 back to 0 for the next file
        out_list[x] = -1
        #Add 1 to the counter.
        counter += 1
        os.system("mv %s %s"%(y, y.split(".")[0]))
    clean_up(output)

def clean_up(output):
    #Close the output file
    output.close()
    #Remove the temporary file
    os.system("rm files.txt")

#Run the create_variables function
create_variables()
