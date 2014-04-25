# Import required modules
import sys, random, os

# The second argument will be the input file
file1 = sys.argv[1]

# The third argument will be the ip
number = sys.argv[2]

ann = "ann/flower41.ann"

# The standard path to the uploaded files
path = "static/uploaded_files/%s/"%(number)

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
outfile = "%s_out.txt" %(number)
os.system("perl classify.pl -d %s -a %s > %s"%(path, ann, outfile))
#result in variable result

# Fake result to test the program!
#result = random.choice(["test sentence", "fake sentance", "stupid sentence"])

# Open an output file, named <ip>_test.txt


# Write the result to the output file and close this file
#outfile.write("This is probably a %s" %(result))
#outfile.close()

