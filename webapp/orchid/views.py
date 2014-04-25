# import the required modules
from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from forms import UploadPictureForm
from django.core.context_processors import csrf
from django.contrib import auth
from time import time
from django.contrib.auth.decorators import login_required
import os

# Function to get the used devise.
def get_device( request ):
    """ Redirect to the servers list. """
    #Initiate the device variable
    device = ""
    #If the used device is in the list, the device is a mobile phone
    '''I have test both html-styles on the iPad. The results shows that the iPad can
    better show the computer style'''
    if 'HTTP_USER_AGENT' in request.META and (
      request.META['HTTP_USER_AGENT'].startswith( 'BlackBerry' ) or \
      "Opera Mobi" in request.META.get('HTTP_USER_AGENT') or \
      "Opera Mini" in request.META.get('HTTP_USER_AGENT') or \
      "Windows CE" in request.META.get('HTTP_USER_AGENT') or \
      "MIDP"       in request.META.get('HTTP_USER_AGENT') or \
      "Palm"       in request.META.get('HTTP_USER_AGENT') or \
      "NetFront"   in request.META.get('HTTP_USER_AGENT') or \
      "Nokia"      in request.META.get('HTTP_USER_AGENT') or \
      "Symbian"    in request.META.get('HTTP_USER_AGENT') or \
      "UP.Browser" in request.META.get('HTTP_USER_AGENT') or \
      "UP.Link"    in request.META.get('HTTP_USER_AGENT') or \
      "WinWAP"     in request.META.get('HTTP_USER_AGENT') or \
      "Android"    in request.META.get('HTTP_USER_AGENT') or \
      "DoCoMo"     in request.META.get('HTTP_USER_AGENT') or \
      "KDDI-"      in request.META.get('HTTP_USER_AGENT') or \
      "Softbank"   in request.META.get('HTTP_USER_AGENT') or \
      "J-Phone"    in request.META.get('HTTP_USER_AGENT') or \
      "IEMobile"   in request.META.get('HTTP_USER_AGENT') or \
      "iPod"       in request.META.get('HTTP_USER_AGENT') or \
      "iPhone"     in request.META.get('HTTP_USER_AGENT') ):
        device = "mobile"
    #Otherwise it is a computer.
    else:
        device = "computer"
    #Return the device
    return device

def check_upload(upload):
    picture = ["jpg","tif","bmp","gif","png","jpeg","psd","pspimage","thm","yuv"]
    
    name = str(upload)
    extension = name.lower().split(".")[-1]
    
    if extension in picture:
        return True
    else:
        name = name.replace(" ","\ ")
        os.system("rm static/uploaded_files/%s"%(name))        
        return False


# Welcome view (homepage)
def welcome(request):
    #Get the used device, using the get_device function
    device = get_device(request)
    
    # Create the args dictionary and save the csrf in this dictonary
    args = {}
    args.update(csrf(request))
    # ONLY FOR TESTING! Save the device in the args dictionary
    args['device']=device
    # Save the html name, with the used device
    html = device+"_welcome.html"
    
    # Call the html, for the correct device, for de welcome page.
    return render_to_response(html, args)


#Function to give the uploaded file a variable part in front of the filename
def processUpload(request, filename):
    
    filename2 = str(filename).replace(" ","_")
    filename = str(filename).replace(" ","\ ")
    
    
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
    
    # Place the variable part (the ip) in front of the filename of the uploaded file
    os.system("mv static/uploaded_files/%s static/uploaded_files/%s_%s"%(filename, ip, filename2))
    
    # Write the new filename to the outputfile
    outfile.write("%s_%s" %(ip, filename2))
    
    # Close the outputfile
    outfile.close()
    

# The upload view (choice file and upload it)
def upload(request):
    #Get the used device, using the get_device function
    device = get_device(request)  
    
    # Get the IP-adres of the computer
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')

    # Replace the '.' in the ip-adres to '_'
    ip = ip.replace('.', '_')
    
    message = ""
    style = ""
    # Check if the method is POST
    if request.method == 'POST':
        
        message = "You didn't select a picture"
        style = "color:red"
        
        # Save the user input from the form
        form = UploadPictureForm(request.POST, request.FILES)
        
        # Check if the form is valid
        if form.is_valid():
            
            # Save the form
            form.save()
            
            is_picture = check_upload(request.FILES["picture"])
            
            if is_picture:
            
                # run the processUpload function to place the ip in front of the name of the uploaded file
                processUpload(request, request.FILES["picture"]) # zie hier nog een extra regel
                
                ''' save the filename and path in python variables
                use the variable part (the ip) to create the path'''
                filename = str(request.FILES["picture"]).replace(" ", "_")
                path = ("static/assets/uploaded_files/%s_%s" % (ip, filename))
                
                # Create the args dictionary and save the csrf in this dictonary
                args = {}
                args.update(csrf(request))
                
                # save the filename and path in the dictionary
                args['filename'] = filename
                args['path'] = path
                
                # Save the html name, whit the used device
                html = device+"_upload_succes.html"              
                
                # Call the upload_succes html, for the correct device and give it the args dictonary
                return render_to_response(html, args)
            
            else:
                # Create the args dictionary and save the csrf in this dictonary    
                args = {}
                args.update(csrf(request))
                
                # Save the empty form in the dictionary
                args['form'] = UploadPictureForm()
                args['message'] = message
                args['style'] = style
                
                # Save the html name, with the used device
                html = device+"_upload.html"
                # Call the upload html, for the correct device and give it the args dictionary
                return render_to_response(html, args)                
                
    
    # When the method is not POST    
    else:
        # Create a form to upload a picture
        form = UploadPictureForm()
        
    # Create the args dictionary and save the csrf in this dictonary    
    args = {}
    args.update(csrf(request))
    
    # Save the empty form in the dictionary
    args['form'] = UploadPictureForm()
    args['message'] = message
    args['style'] = style
    
    # Save the html name, with the used device
    html = device+"_upload.html"
    # Call the upload html, for the correct device and give it the args dictionary
    return render_to_response(html, args)

# The result view (to display the result of the analysis)
def result(request):
    #Get the used device, using the get_device function
    device = get_device(request)
    try:
        # Get the IP-adres of the computer
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
            
        # Replace the '.' in the ip-adres to '_'
        ip = ip.replace('.', '_') 
        
        os.system("sh converter.sh %s"%(ip))
        
        # Read in the filename from <ip>_filename.txt
        infile = open('%s_filename.txt' %(ip), 'r')
        filename = infile.read().strip()
        
        # Close the infile
        infile.close()
        
        # Run the program to identify the orchid
        # Warning: The program now used is only a test program!
        os.system("python classify.py %s %s" % (filename, ip))
        os.system("python result.py %s" % (ip))
        
        # Open the file with the results from the identify program
        result = open('%s_result.txt' %(ip), 'r')
        
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
        args['ip'] = ip
        
        # Save the html name with the used device
        html = device+"_result.html"
        # Call the result html, for the correct device, with the args dictionary
        return render_to_response(html, args)
    except IOError:
        '''If an IOError occure, the picture is uploaded just when the administrator removed all
        unused files. So the uploaded picture is also removed. Send the user to the sorry page,
        which tells the user to try uploading again.'''
        # Save the html name with the used device
        html = device+"_sorry.html"
        # Go to the sorry html, for the correct device
        return render_to_response(html)


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
    replace all . in _ to prevent errors for the extension '''
    var_part = str(time()).replace('.', '_')    
    
    # Read in the filename from <ip>_filename.txt, save it and close the file
    infile = open('%s_filename.txt' %(ip), 'r')
    filename = infile.read().strip()
    infile.close()
    
    # Remove the temporary file <ip>_filename.txt
    # Move the uploaded picture and its result to the result directory,
    # Save it as timestamp_ip.jpg and timestamp_ip_result.txt
    os.system("rm %s_filename.txt" %(ip))
    os.system("mv static/uploaded_files/%s/%s results/%s_%s" %(ip, filename, var_part, filename))
    os.system("rm -r static/uploaded_files/%s" %(ip))
    os.system("mv %s_out.txt results/%s_%s_result1.txt" %(ip, var_part, ip))
    os.system("mv %s_result.txt results/%s_%s_section.txt" %(ip, var_part, ip))
    
    # Go back to the welcome page
    return HttpResponseRedirect('/welcome')


# To remove all leftover files, login is required
def login(request):
    #Get the used device, using the get_device function
    device = get_device(request)
    # Create a dictionary and put the csrf in it
    args = {}
    args.update(csrf(request))
    
    #Save the html name with the used device
    html=device+"_login.html"
    #Go to the login html, for the correct device, give it the dictionary
    return render_to_response(html, args)

# Function to check the username and password
def auth_view(request):
    # Get the username and password
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    ''' If the username and password are incorrect user will be None
    Otherwise it will be the user '''
    user = auth.authenticate(username=username, password=password)
    
    ''' Go to the correct page (admin/remove for correct login, invalid for
    invalid login)'''
    if user is not None:
        #Login the user
        auth.login(request, user)
        return HttpResponseRedirect('/admin/remove')
    else:
        return HttpResponseRedirect('/accounts/invalid')

# function for logout
def logout(request):
    #Log the user out
    auth.logout(request)
    #Go back to the welcome page
    return HttpResponseRedirect('/welcome/')

# Function for invalid login
def invalid_login(request):
    #Get the used device, using the get_device function
    device = get_device(request)
    # Go to the invalid login html, for the correct device
    html = device+"_invalid_login.html"
    return render_to_response(html)

@login_required
#User need to be registreded. Even when the user is not active this user can login and remove the files.
def remove(request):
    #Get the used device, using the get_device function
    device = get_device(request)
    # List all the files that will be removed using a command line command (ls)
    '''Save the name(s) of the picture(s) that will be removed in uploads.txt and the
     name(s) of the temporary file(s) in temps.txt'''
    os.system("ls static/uploaded_files > uploads.txt")
    os.system("ls | egrep *_filename.txt > temps.txt")
    
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
    
    # Save the html name with the used device
    html = device+"_remove.html"
    # Call the html, for the correct device, and give it the args directory
    return render_to_response(html, args)