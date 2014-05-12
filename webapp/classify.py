# Import required modules
import sys, random, os

#Function for creating the variables
def create_variables():
    # The second argument will be the input file
    file1 = sys.argv[1]

    # The third argument will be the ip
    ip = sys.argv[2]

    #Set ann to the ann that will be used.
    ann = "ann/flower51.ann"

    # The standard path to the uploaded files
    path = "static/uploaded_files/%s/"%(ip)

    #Create an output file
    outfile = "%s_out.txt" %(ip)

    #Run the classify function
    classify(path, ann, outfile)

#Function for classifying the uploaded picture
def classify(path, ann, outfile):
    #Run classify.pl to classify the uploaded picture.
    os.system("perl classify.pl -d %s -a %s > %s"%(path, ann, outfile))

#Run the create_variables function
create_variables()
