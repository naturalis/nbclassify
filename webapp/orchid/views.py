# import the required modules
from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from forms import UploadPictureForm
from django.core.context_processors import csrf
from django.contrib import auth
from time import time
from django.contrib.auth.decorators import login_required
import os

# Welcome view (homepage)
def welcome(request):    
    # Call the html for de welcome page.
    return render_to_response('welcome.html')


#Function to give the uploaded file a variable part in front of the filename
def processUpload(request, filename):
    
    # Get the IP-adres of the computer
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
        
    # Replace the '.' in the ip-adres to '_'
    ip = ip.replace('.', '_')
    
    # Create an output file named <ip>_filename.txt
    outfile = open('%s_filename.txt' %(ip), 'w')
    
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
    try:
        # Get the IP-adres of the computer
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
            
        # Replace the '.' in the ip-adres to '_'
        ip = ip.replace('.', '_') 
        
        # Read in the filename from <ip>_filename.txt
        infile = open('%s_filename.txt' %(ip), 'r')
        filename = infile.read().strip()
        
        # Close the infile
        infile.close()
        
        # Run the program to identify the orchid
        # Warning: The program now used is only a test program!
        os.system("python resultaat.py %s %s" % (filename, ip))
        
        # Open the file with the results from the identify program
        result = open('%s_test.txt' %(ip), 'r')
        
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
    except IOError:
        return HttpResponseRedirect('/sorry')
    
# if the picuter is removed during a calculation, the sorry function will be called    
def sorry(request):
    # Go to the sorry html
    return render_to_response('sorry.html')

# The exit view (to "close" the app and remove all created temporary files)
def exit(request):
    # Get the IP-adres of the computer
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
        
    # Replace the '.' in the ip-adres to '_'
    ip = ip.replace('.', '_')
    
    # Read in the filename from <ip>_filename.txt, save it and close the file
    infile = open('%s_filename.txt' %(ip), 'r')
    filename = infile.read().strip()
    infile.close()
    
    # Remove all temporary files
    os.system("rm %s_filename.txt" %(ip))
    os.system("rm static/uploaded_files/%s" %(filename))
    os.system("rm %s_test.txt" %(ip))
    os.system("rm abc.txt")
    
    # Go back to the welcome page
    return HttpResponseRedirect('/welcome')


# To remove all leftover files, login is required
def login(request):
    # Create a dictionary and put the csrf in it
    c = {}
    c.update(csrf(request))
    
    #Go to the login html, give it the dictionary
    return render_to_response('login.html', c)

# Function to check the username and password
def auth_view(request):
    # Get the username and password
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    ''' If the user and password are incorrect it user will be None
    Otherwise it will be the user '''
    user = auth.authenticate(username=username, password=password)
    
    ''' Go to the correct page (loggedin for correct login, invalid for
    invalid login)'''
    if user is not None:
        auth.login(request, user)
        return HttpResponseRedirect('/accounts/loggedin')
    else:
        return HttpResponseRedirect('/accounts/invalid')

# Function for after login   
def loggedin(request):
    # After login go to the remove page
    return HttpResponseRedirect('/admin/remove')

# function for logout
def logout(request):
    #Log the user out
    auth.logout(request)
    #Go back to the welcome page
    return HttpResponseRedirect('/welcome/')

# Function for invalid login
def invalid_login(request):
    # Go to the invalid login html
    return render_to_response('invalid_login.html')

@login_required
#User need to be registreded. Even when the user is not active this user can login and remove the files.
def remove(request):
    # Remove all the files in the static/uploaded_files folder
    os.system("rm static/uploaded_files/*")
    # Remove all .txt files
    os.system("rm *.txt")
    # Go to the remove html
    return render_to_response('remove.html')