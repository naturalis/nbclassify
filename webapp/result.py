#Import the required module
import sys

#Function for creating the variables
def create_variables():
    #Save the ip-adress which is the second commandline argument
    ip = sys.argv[1]

    #Create a list with the possible sections.
    #Sort them so form Z-A. The index of the section
    #Now is corresponding the index of the list from the classifly.pl script
    sections = ["Parvisepalum", "Pardalopetalum", "Paphiopedilum", "Coryopedilum", "Cochlopetalum", "Brachypetalum", "Barbata"]

    #Open the output file from the classifly.pl script.
    infile = open("%s_out.txt"%(ip), 'r')

    #Read in the content of the output file
    file1 = infile.readlines()

    #Create a variable for the index
    index = -1

    #Create an empty list. This list will be used
    #To save the values of the output list form classify.pl
    values = []

    #Create a counter
    counter = -1

    #Create an output file
    output = open("%s_result.txt"%(ip) , 'w')

    #Run the get_section function, give it all the created variables.
    get_section(sections, file1, index, values, counter, output)

def get_section(sections, file1, index, values, counter, output):
    #Loop through the lines of the file
    for line in file1:
        #For every loop add 1 to the counter
        counter += 1
        #To accept error use a try-except
        try:
            #Find the lines which contains the values
            if counter >= 1 and counter <= 7:
                #Add these values to the values list
                values.append(float(line.strip().strip(",").strip("'")))
        #If an error occur, except this and continue
        except:
            continue		

    #Loop through the values
    for number in values:
        #When the value is bigger than 0, the picture
        #Is classified to this section.
        if number > 0:
            #Get the index of this value
            index = values.index(number)
        #Otherwise
        else:
            #Go on
            continue

    #The result of the cassification script is the section
    #Which index corresponds to the index of the positive value
    result = sections[index]


    #Write the result to this file and close the file
    output.write("%s"%(result))
    output.close()

#Run the create_variables function to create the variables
create_variables()