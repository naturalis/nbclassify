import os

#function for creating the variables
def create_variables():
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
    #Run the process_files function
    process_files(number, files, counter, output)
    #Run the clean_up function
    clean_up(output, name)

#Function for combining the files
def process_files(number, files, counter, output):
    #Loop through the files list
    for file_index in range(number):
        file = files[file_index].strip()
        #Read the content of the file, save it as a list.
        content = open(file, 'r').readlines()
        #Loop through the lines
        for row_number in range(len(content)):
            #create a list, every columns is an entry of the list
            row = content[row_number].split("\t")
            #When counter is 0, this is the first file.
            if counter == 0:
                #The whole content of the file is written to the output file
                for entry in row:
                    output.write(entry.strip() + "\t")
                #After looping through the content, write an enter
                output.write("\n")
            #When counter isn't 0, it is not the first file
            else:
                #When row_number is 0 it is the header
                if row_number == 0:
                    #The header will not be written to the output file, because it is already there
                    pass
                #When row_number isn't 0 it is not the header
                else:
                    #The content will be written to the output file
                    for entry in row:
                        output.write(entry.strip() + "\t")
                    #After looping through the content, write an enter to the output file.
                    output.write("\n")
        #Add 1 to the counter
        counter += 1
#Function for removing temporary files.
def clean_up(output, name):
    #Close the output file
    output.close()
    #Remove the temporary files.txt file
    os.system("rm files.txt")
    #Move the output file out of the Flower directory
    os.system("mv %s ../"%(name))

#Run the create variables function
create_variables()
