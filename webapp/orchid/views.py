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


# Create your views here.
def welcome(request):
    name = "Mirna"
    age = 21
    return render_to_response('welcome.html')

####### Adjustment Benjamin 5-11-13 #########
def processUpload(request, filename):
    def wrapper():
        outfile = open("abc.txt", 'w')
        outfile.write("filename: %s\n" %(filename))
        with open("static/uploaded_files/%s_%s" % (models.var_part, filename.replace(' ','_'))) as f:
            for line in f: # doe iets met iedere bestandsregel
                outfile.write(line)
            outfile.close()
    threading.Thread(target = wrapper).start()


def upload(request):
    if request.method == 'POST':
        form = UploadPictureForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            processUpload(request, request.FILES["picture"]) # zie hier nog een extra regel
            #return HttpResponseRedirect('/upload_success')
            
            path = ("static/assets/uploaded_files/%s_%s" % (models.var_part, request.FILES["picture"])).replace(' ','_')
            
            args = {}
            args.update(csrf(request))
            
            args['filename'] = request.FILES["picture"]
            args['path'] = path
            return render_to_response('upload_succes.html', args)
        
    else:
        form = UploadPictureForm()    
    args = {}
    args.update(csrf(request))
    
    args['form'] = UploadPictureForm()
    print args
    return render_to_response('upload.html', args)

def upload_success(request):
    return render_to_response('upload_succes.html')
