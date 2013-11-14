# import the required modules
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template.loader import get_template
from django.shortcuts import render_to_response
from django.template import Context
from django.views.generic.base import TemplateView
from forms import UploadPictureForm
from django.core.context_processors import csrf
import threading
from time import time
import models
import os


# Welcome view (homepage)
def welcome(request):
    # remove filename.txt and all files in static/uploaded_files
    os.system("rm filename.txt")
    os.system("rm static/uploaded_files/*")
    
    # Call the html for de welcome page.
    return render_to_response('welcome.html')


#Function to give the uploaded file a variable part in front of the filename
def processUpload(request, filename):
    # Create an output file named filename.txt
    outfile = open('filename.txt', 'w')
    
    ''' Create the variable part for the filename using a timestamp.
    replace all . in _ to prevent errors for the extension '''
    var_part = str(time()).replace('.', '_')
    
    # Place the variable part in front of the filename of the uploaded file
    os.system("mv static/uploaded_files/%s static/uploaded_files/%s_%s"%(filename, var_part, filename))
    
    # Write the new filename to the outputfile
    outfile.write("%s_%s" %(var_part, filename))
    
    # Close the outputfile
    outfile.close()
    
    # Return the variable part of the filename
    return(var_part)
    

# The upload view (choice file and upload it)
def upload(request):
    
    # Check if the method is POST
    if request.method == 'POST':
        
        # Save the user input from the form
        form = UploadPictureForm(request.POST, request.FILES)
        
        # Check if the form is valid
        if form.is_valid():
            
            # Save the form
            form.save()
            
            # Call the variable part back from processUpload
            vari_part = processUpload(request, request.FILES["picture"]) # zie hier nog een extra regel
            
            ''' save the filename and path in python variables
            use the variable part to create the path'''
            filename = request.FILES["picture"]
            path = ("static/assets/uploaded_files/%s_%s" % (vari_part, filename))
            
            # Create the args dictionary and save the csrf in this dictonary
            args = {}
            args.update(csrf(request))
            
            # save the filename and path in the dictionary
            args['filename'] = filename
            args['path'] = path
            
            # Call the upload_succes html and give it the args dictonary
            return render_to_response('upload_succes.html', args)
    
    # When the method is not POST    
    else:
        # Create a form to upload a picture
        form = UploadPictureForm()
        
    # Create the args dictionary and save the csrf in this dictonary    
    args = {}
    args.update(csrf(request))
    
    # Save the empty form in the dictionary
    args['form'] = UploadPictureForm()
    
    # Call the upload html and give it the args dictionary
    return render_to_response('upload.html', args)

# The result view (to display the result of the analisis)
def result(request):
    # Read in the filename from filename.txt
    infile = open('filename.txt', 'r')
    filename = infile.read().strip()
    
    # Close the infile
    infile.close()
    
    # Run the program to identify the orchid
    # Warning: The program now used is only a test program!
    os.system("python resultaat.py %s" % (filename))
    
    # Open the file with the results from the identify program
    result = open('test.txt', 'r')
    
    # Read in the results
    read_result = result.read()
    
    # Close the file
    result.close()
    

    # Create the args dictionary and save the csrf in this dictonary   
    args = {}
    args.update(csrf(request))
    
    # Save the filename and the result in the args dictionary
    args['filename'] = filename
    args['result'] = read_result
    
    # Call the result html with the args dictionary
    return render_to_response('result.html', args)

# The exit view (to "close" the app and remove all created temporary files)
def exit(request):
    # Read in the filename, save it and close the file
    infile = open('filename.txt', 'r')
    filename = infile.read().strip()
    infile.close()
    
    # Remove all temporary files
    os.system("rm filename.txt")
    os.system("rm static/uploaded_files/%s" %(filename))
    os.system("rm test.txt")
    os.system("rm abc.txt")
    
    # Go back to the welcome page
    return HttpResponseRedirect('/welcome')