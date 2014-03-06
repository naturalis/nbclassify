# Import required modules
import sys, random

# The second argument will be the input file
file1 = sys.argv[1]

# The third argument will be the ip
number = sys.argv[2]

# The standard path to the uploaded files
path = "static/uploaded_files/"

''' The full path to the file is the path to the directory
which contain the file + the name of the file'''
full_path = path + file1


''' THE NEXT PART IS ONLY FOR TESTING! THIS WILL BE REMOVED LATER!!!'''

# Open the uploaded file (picture) in read modus
infile = open(full_path, 'r')

# Read the content of the file
read_infile = infile.read()

# Close the input file
infile.close()

''' THIS PART NEEDS TO BE DONE YET!'''
#analyze
#analyze
#analyze
#result in variable result

# Fake result to test the program!
result = random.choice(["test sentence", "fake sentance", "stupid sentence"])

# Open an output file, named <ip>_test.txt
outfile = open("%s_test.txt" %(number), 'w')

# Write the result to the output file and close this file
outfile.write("This is probably a %s" %(result))
outfile.close()

