import os
from time import time

from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from django.core.context_processors import csrf
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import UploadedFile

from forms import UploadPictureForm

def home(request):
    data = {}
    data.update(csrf(request))

    return render_to_response("orchid/home.html", data)

def upload(request):
    data = {}
    data['style'] = ""

    if request.method == 'POST':
        # This is a model form with file field.
        form = UploadPictureForm(request.POST, request.FILES)

        if form.is_valid():
            # Save the file to the location specified in the Photos model.
            form_instance = form.save()

            data.update(csrf(request))
            data['photo_url'] = form_instance.photo.url

            return render_to_response("orchid/upload_succes.html", data)
        else:
            data.update(csrf(request))
            data['form'] = UploadPictureForm()
            data['message'] = "Please select a valid image file."
            data['style'] = "color:red"

            return render_to_response("orchid/upload.html", data)

    # Set the template data.
    data.update(csrf(request))
    data['form'] = UploadPictureForm()

    return render_to_response("orchid/upload.html", data)

# The result view (to display the result of the analysis)
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

        #Run converter.sh to convert jpg files to png
        os.system("sh converter.sh %s"%(ip))

        # Read in the filename from <ip>.filename.txt
        infile = open('%s.filename.txt' %(ip), 'r')
        filename = infile.read().strip()

        # Close the infile
        infile.close()

        # Run the program to classify the orchid
        os.system("python classify.py %s %s" % (filename, ip))
        #After the previous step a list with numbers is created.
        #Runt result.py to translate this list to a readable result.
        os.system("python result.py %s" % (ip))

        # Open the file with the result from the result.py program
        result = open('%s_result.txt' %(ip), 'r')

        # Read in the result
        read_result = result.read()

        # Close the file
        result.close()

        # Create the args dictionary and save the csrf in this dictonary
        args = {}
        args.update(csrf(request))

        # Save the filename, the result and the ip in the args dictionary
        args['filename'] = filename
        args['section'] = read_result
        args['genus'] = "Paphiopedilum"
        args['ip'] = ip

        # Call the result html, for the correct device, with the args dictionary
        return render_to_response("orchid/result.html", args)
    except IOError:
        '''If an IOError occure, the picture is uploaded just when the administrator removed all
        unused files. So the uploaded picture is also removed. Send the user to the sorry page,
        which tells the user to try uploading again.'''
        return render_to_response("orchid/sorry.html")

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

    ''' Create the variable part for the filename using a timestamp.
    replace all . into _ to prevent errors for the extension '''
    var_part = str(time()).replace('.', '_')

    # Read in the filename from <ip>.filename.txt, save it and close the file
    infile = open('%s.filename.txt' %(ip), 'r')
    filename = infile.read().strip()
    infile.close()

    # Remove the temporary file <ip>.filename.txt
    # Move the uploaded picture and its result to the result directory,
    # Save it as timestamp_ip.png, timestamp_ip_result.txt and timestamp_ip_section.txt
    os.system("rm %s.filename.txt" %(ip))
    os.system("mv static/uploaded_files/%s/%s results/%s_%s" %(ip, filename, var_part, filename))
    os.system("rm -r static/uploaded_files/%s" %(ip))
    os.system("mv %s_out.txt results/%s_%s_result.txt" %(ip, var_part, ip))
    os.system("mv %s_result.txt results/%s_%s_section.txt" %(ip, var_part, ip))

    # Go back to the welcome page
    return HttpResponseRedirect('/')


# To remove all leftover files, login is required
def login(request):
    # Create a dictionary and put the csrf in it
    args = {}
    args.update(csrf(request))

    #Go to the login html, for the correct device, give it the dictionary
    return render_to_response("orchid/login.html", args)

# Function to check the username and password
def auth_view(request):
    # Get the username and password
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    ''' If the username and password are incorrect user will be None
    Otherwise it will be the user '''
    user = auth.authenticate(username=username, password=password)

    ''' Go to the correct page (admin/remove for correct login, accounts/invalid for
    invalid login)'''
    if user is not None:
        #Login the user
        auth.login(request, user)
        return HttpResponseRedirect('cleanup/')
    else:
        return HttpResponseRedirect('orchid/accounts/invalid/')

# function for logout
def logout(request):
    #Log the user out
    auth.logout(request)
    #Go back to the welcome page
    return HttpResponseRedirect('/')

# Function for invalid login
def invalid_login(request):
    return render_to_response("orchid/invalid_login.html")

@login_required(login_url='/orchid/accounts/login/')
def cleanup(request):
    # List all the files that will be removed using a command line command (ls)
    '''Save the name(s) of the picture(s) that will be removed in uploads.txt and the
     name(s) of the temporary file(s) in temps.txt'''
    os.system("ls static/uploaded_files > uploads.txt")
    os.system("ls | egrep *.filename.txt > temps.txt")

    #Remove all the unused pictures and their temporary files
    os.system("rm -r static/uploaded_files/*")
    os.system("rm *filename.txt")

    #Read the content of the uploads.txt file and the temps.txt file and save the content in
    # Python variables
    uploads_in = open("uploads.txt", 'r')
    temps_in = open("temps.txt", 'r')
    uploads = uploads_in.read()
    temps = temps_in.read()

    # Create the args dictionary and save the csrf in this dictonary
    args = {}
    args.update(csrf(request))

    #Save the list of the pictures that will be removed in the dictionary
    args['uploads'] = uploads
    #Save the list of the temporary files that will be removed in the dictionary
    args['temps'] = temps

    # Remove the text files wich contain the lists
    os.system("rm uploads.txt temps.txt")

    # Call the html, for the correct device, and give it the args directory
    return render_to_response("orchid/remove.html", args)
