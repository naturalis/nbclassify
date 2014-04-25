import os

#Open a in read mode
files = open("a", 'r')

#Create a list of all the xml files
infiles = files.readlines()

#Close and remove the temporary file
files.close()
#Loop through the list
for x in infiles:
    #Remove all enters at the back of the filename
    infile = x.strip()
    #Get the picture id, to save the tags with the same number as the picture
    #Example the name of the meta file is 123456789.xml so the id is 123456789
    number = infile.split(".")[0]
    #Print a message
    print "Collecting the tags of file %s"%(infile)
    #Open the meta data file in read mode
    open_file = open(infile, 'r')
    #Make a list of the meta data
    read_file = open_file.readlines()
    #Try to find the tags
    try:
        '''One line befor the first tag you can find "/t<tags>\n"
So the first tag will be the index of "/t<tags>\n" +1'''
        #Get the index of the first tag
        start = read_file.index("\t<tags>\n") +1
        '''One line after the last tag you can find "\t</tags>\n"
So the last tag will be the index of "/t</tags>\n" -1. Since a for-loop
loops from start to end, EXCLUDING the end, you use the index of "/t</tags>\n"'''
        #Get the index of the end of the tags
        end = read_file.index("\t</tags>\n")
        #Get the original name of the picture
        title = read_file[2].split(">")[1].split("<")[0]
        #print title
        #Save the output name (using the id of the picture)
        out_name = "%s_tags.txt"%(number)
        #Open the output file in write mode
        output = open(out_name, 'w')
        print "The tags will be saved in %s"%(out_name)
        #Write the name of the picture and a white line to the output file
        #The title will always be the first line of the output file
        output.write("%s\n\n"%(title))
        #Loop through the tags
        for y in range(start, end):
            #print read_file[y].strip().split(" ")[3].split('"')[1]
            #Write the text between <tag> and </tag> to the output file
            output.write(read_file[y].strip().split(" ")[4].split('"')[1])
            #Write an enter to the output file
            output.write("\n")
        #When the loop ends, close the output file
        output.close()
        '''If there are no tags, a ValueError arise. Except this Error and print
a message that the file has no tags'''
    except ValueError:
        print "%s has no tags"%(infile)
    #Close the output file
    open_file.close()
    #break
